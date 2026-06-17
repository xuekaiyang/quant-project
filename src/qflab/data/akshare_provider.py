"""Akshare 数据源骨架。

第一版仅给出清晰的接口签名与 TODO，以避免不可运行的伪代码。当实际接入时：

- get_stock_list: ak.stock_info_a_code_name()
- get_daily_bar: ak.stock_zh_a_hist(symbol=..., period='daily', adjust='qfq')
- get_market_cap: ak.stock_zh_a_spot_em() 仅给当前快照；历史市值需要拼接
- get_industry: ak.stock_board_industry_name_em() / 申万分类

注意：akshare 接口经常变动，请以 akshare 当前文档为准。
"""

from __future__ import annotations

import pandas as pd

from ..utils.logger import get_logger
from .base import DataProvider
from .normalizer import normalize_daily_bar

logger = get_logger(__name__)


class AkshareProvider(DataProvider):
    name = "akshare"

    def __init__(self, request_sleep_sec: float = 0.3):
        self.request_sleep_sec = request_sleep_sec

    def _import(self):
        try:
            import akshare as ak  # noqa: F401
            return ak
        except ImportError as e:
            raise ImportError(
                "akshare not installed. Install via: pip install '.[data]'"
            ) from e

    def get_stock_list(self) -> pd.DataFrame:
        """TODO: 调用 ak.stock_info_a_code_name() 并规范化为 instrument 列。"""
        raise NotImplementedError("AkshareProvider.get_stock_list: TODO")

    def get_daily_bar(self, start_date: str, end_date: str) -> pd.DataFrame:
        """TODO: 遍历 stock_list 调用 ak.stock_zh_a_hist 并 normalize_daily_bar。"""
        raise NotImplementedError("AkshareProvider.get_daily_bar: TODO")

    def get_market_cap(self, start_date: str, end_date: str) -> pd.DataFrame:
        raise NotImplementedError("AkshareProvider.get_market_cap: TODO")

    def get_industry(self) -> pd.DataFrame:
        raise NotImplementedError("AkshareProvider.get_industry: TODO")

    def update_daily_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        df = self.get_daily_bar(start_date, end_date)
        return normalize_daily_bar(df)
