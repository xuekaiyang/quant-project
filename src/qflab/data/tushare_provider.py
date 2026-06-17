"""Tushare 数据源接入。

需要在 .env 中配置 TUSHARE_TOKEN，并安装可选依赖：``pip install '.[data]'``。

拉取策略（按交易日，不按股票）：
- ``pro.daily(trade_date=YYYYMMDD)`` 一次返回当天全市场行情；
- ``pro.adj_factor(trade_date=...)`` / ``pro.daily_basic(trade_date=...)`` 同理按日拉；
- ``pro.stock_basic(list_status='L,D,P')`` 取上市+退市+暂停全样本，消除幸存者偏差。

落库走 raw 日分区（``data/raw/daily/<YYYYMMDD>.parquet``），fetch 与 normalize 解耦，
天然支持断点续传。最终由 ``build_daily_bar_from_raw`` 合并成 ``daily_bar.parquet``。
"""

from __future__ import annotations

import os
import time

import pandas as pd

from ..utils.config import load_config
from ..utils.logger import get_logger
from .base import DataProvider
from .calendar import TradingCalendar
from .storage import (
    build_daily_bar_from_raw,
    list_raw_dates,
    load_daily_bar,
    save_raw_daily,
)

logger = get_logger(__name__)

# tushare 列名 -> 内部 schema
_DAILY_RENAME = {"vol": "volume"}
_BASIC_KEEP = ["ts_code", "trade_date", "total_mv", "circ_mv"]


class TushareProvider(DataProvider):
    name = "tushare"

    def __init__(
        self,
        token: str | None = None,
        token_env: str = "TUSHARE_TOKEN",
        request_sleep_sec: float | None = None,
        max_retries: int = 5,
        list_status: str = "L,D,P",
    ):
        self.token = token or os.environ.get(token_env, "")
        if not self.token:
            logger.warning("TushareProvider: token is empty. Set %s in .env", token_env)
        self._pro = None
        self._stock_info: pd.DataFrame | None = None

        cfg = load_config().data_source.get("tushare", {}) if load_config().data_source else {}
        self.request_sleep_sec = (
            request_sleep_sec if request_sleep_sec is not None
            else float(cfg.get("request_sleep_sec", 0.35))
        )
        self.max_retries = max_retries
        self.list_status = list_status

    # ------------------------------------------------------------------ api
    def _api(self):
        if self._pro is not None:
            return self._pro
        try:
            import tushare as ts
        except ImportError as e:
            raise ImportError(
                "tushare not installed. Install via: pip install '.[data]'"
            ) from e
        self._pro = ts.pro_api(self.token)
        return self._pro

    def _call_with_retry(self, method: str, **kwargs) -> pd.DataFrame:
        """调用 pro.<method>(**kwargs)，带限频退避重试。

        Tushare 超频会抛异常（消息含 '每分钟' / 'limit' 等）。这里统一指数退避重试，
        每次调用后固定 sleep ``request_sleep_sec`` 平滑请求速率。
        """
        pro = self._api()
        fn = getattr(pro, method)
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                df = fn(**kwargs)
                if self.request_sleep_sec:
                    time.sleep(self.request_sleep_sec)
                return df
            except Exception as e:  # noqa: BLE001 - tushare 抛通用 Exception
                last_err = e
                wait = min(60.0, 2.0 ** attempt)
                logger.warning(
                    "tushare %s failed (attempt %d/%d): %s; retry in %.1fs",
                    method, attempt + 1, self.max_retries, e, wait,
                )
                time.sleep(wait)
        raise RuntimeError(
            f"tushare {method} failed after {self.max_retries} retries: {last_err}"
        )

    # -------------------------------------------------------------- metadata
    def get_stock_list(self) -> pd.DataFrame:
        """全样本股票列表（上市+退市+暂停），列：instrument/name/industry/list_date/...

        用 ``list_status='L,D,P'`` 同时拉三种状态，逐段请求后拼接，避免漏掉已退市股。
        """
        frames = []
        for status in self.list_status.split(","):
            status = status.strip()
            if not status:
                continue
            df = self._call_with_retry(
                "stock_basic",
                exchange="",
                list_status=status,
                fields="ts_code,name,industry,list_date,delist_date,list_status",
            )
            frames.append(df)
        out = pd.concat(frames, ignore_index=True)
        out = out.rename(columns={"ts_code": "instrument"})
        out["instrument"] = out["instrument"].astype(str)
        # ST 由当前名称推断（MVP：未追溯历史曾用名）
        out["is_st"] = out["name"].fillna("").str.contains("ST", case=False)
        return out

    def _stock_info_cached(self) -> pd.DataFrame:
        if self._stock_info is None:
            self._stock_info = self.get_stock_list()
        return self._stock_info

    def get_industry(self) -> pd.DataFrame:
        """行业分类（静态，来自 stock_basic）。columns: instrument, industry。"""
        info = self._stock_info_cached()
        return info[["instrument", "industry"]].dropna(subset=["instrument"]).copy()

    # ----------------------------------------------------------- per-day raw
    def fetch_one_day(self, trade_date: str) -> pd.DataFrame:
        """拉取单个交易日全市场原始行情，合并 daily + adj_factor + daily_basic。

        返回未规范化的长表，含 REQUIRED_COLUMNS + adj_factor/market_cap/industry/
        is_st/is_suspended。无数据（非交易日/未来日）返回空 DataFrame。
        """
        d = pd.Timestamp(trade_date).strftime("%Y%m%d")
        daily = self._call_with_retry("daily", trade_date=d)
        if daily is None or daily.empty:
            return pd.DataFrame()

        adj = self._call_with_retry("adj_factor", trade_date=d)
        basic = self._call_with_retry("daily_basic", trade_date=d, fields=",".join(_BASIC_KEEP))

        df = daily.rename(columns=_DAILY_RENAME)
        if adj is not None and not adj.empty:
            df = df.merge(adj[["ts_code", "trade_date", "adj_factor"]],
                          on=["ts_code", "trade_date"], how="left")
        if basic is not None and not basic.empty:
            df = df.merge(basic[["ts_code", "trade_date", "total_mv"]],
                          on=["ts_code", "trade_date"], how="left")
            df = df.rename(columns={"total_mv": "market_cap"})

        df = df.rename(columns={"ts_code": "instrument"})
        df["instrument"] = df["instrument"].astype(str)
        df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")

        # 行业 / ST 来自静态 stock_basic
        info = self._stock_info_cached()[["instrument", "industry", "is_st"]]
        df = df.merge(info, on="instrument", how="left")
        # 停牌：当日有行情即视为可交易；volume 为 0/缺失视为停牌
        df["is_suspended"] = df["volume"].fillna(0).le(0)

        keep = [
            "trade_date", "instrument", "open", "high", "low", "close",
            "volume", "amount", "adj_factor", "market_cap", "industry",
            "is_st", "is_suspended",
        ]
        return df[[c for c in keep if c in df.columns]]

    def get_daily_bar(self, start_date: str, end_date: str) -> pd.DataFrame:
        """按交易日历逐日拉取并拼接 [start, end] 全市场行情（未规范化）。"""
        cal = TradingCalendar.from_tushare(self._api(), start_date, end_date)
        frames = []
        for dt in cal.dates:
            day = self.fetch_one_day(dt)
            if not day.empty:
                frames.append(day)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def get_market_cap(self, start_date: str, end_date: str) -> pd.DataFrame:
        """按交易日拉取市值。columns: trade_date, instrument, market_cap。"""
        cal = TradingCalendar.from_tushare(self._api(), start_date, end_date)
        frames = []
        for dt in cal.dates:
            d = pd.Timestamp(dt).strftime("%Y%m%d")
            basic = self._call_with_retry(
                "daily_basic", trade_date=d, fields="ts_code,trade_date,total_mv"
            )
            if basic is not None and not basic.empty:
                frames.append(basic)
        if not frames:
            return pd.DataFrame(columns=["trade_date", "instrument", "market_cap"])
        out = pd.concat(frames, ignore_index=True).rename(
            columns={"ts_code": "instrument", "total_mv": "market_cap"}
        )
        out["instrument"] = out["instrument"].astype(str)
        out["trade_date"] = pd.to_datetime(out["trade_date"], format="%Y%m%d")
        return out[["trade_date", "instrument", "market_cap"]]

    # --------------------------------------------------------- orchestration
    def update_daily_data(
        self,
        start_date: str,
        end_date: str,
        skip_existing: bool = True,
        build: bool = True,
        adjust: str | None = None,
    ) -> pd.DataFrame | None:
        """按交易日历循环拉取 → raw 日分区落库 → 合并构建 daily_bar。

        Parameters
        ----------
        skip_existing : 已存在 raw 日文件的交易日跳过（断点续传）。
        build : 拉取完成后是否合并构建 normalized/daily_bar.parquet。
        adjust : 透传 normalize（None | 'qfq'）。

        Returns
        -------
        构建后的 daily_bar DataFrame（build=False 时返回 None）。
        """
        cal = TradingCalendar.from_tushare(self._api(), start_date, end_date)
        all_days = [pd.Timestamp(d).strftime("%Y%m%d") for d in cal.dates]
        existing = set(list_raw_dates()) if skip_existing else set()
        todo = [d for d in all_days if d not in existing]
        logger.info(
            "update_daily_data: %d trading days in range, %d already cached, %d to fetch",
            len(all_days), len(all_days) - len(todo), len(todo),
        )

        for i, d in enumerate(todo, 1):
            day = self.fetch_one_day(d)
            if day.empty:
                logger.info("  [%d/%d] %s: no data (skip)", i, len(todo), d)
                continue
            save_raw_daily(day, d)
            logger.info("  [%d/%d] %s: rows=%d", i, len(todo), d, len(day))

        if build:
            build_daily_bar_from_raw(adjust=adjust)
            return load_daily_bar()
        return None
