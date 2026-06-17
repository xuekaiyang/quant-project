"""tests for TushareProvider（mock pro，不打真实网络）。"""

from __future__ import annotations

import pandas as pd
import pytest

from qflab.data.tushare_provider import TushareProvider


class FakePro:
    """模拟 tushare pro_api，记录调用并返回固定数据。"""

    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    def stock_basic(self, exchange="", list_status="L", fields=""):
        self.calls.append(("stock_basic", {"list_status": list_status}))
        if list_status == "L":
            data = [("000001.SZ", "平安银行", "银行", "19910403", None, "L"),
                    ("600000.SH", "浦发银行", "银行", "19991110", None, "L")]
        elif list_status == "D":
            data = [("000003.SZ", "ST国农", "综合", "19910129", "20020909", "D")]
        else:  # P
            data = []
        return pd.DataFrame(
            data,
            columns=["ts_code", "name", "industry", "list_date", "delist_date", "list_status"],
        )

    def daily(self, trade_date=""):
        self.calls.append(("daily", {"trade_date": trade_date}))
        return pd.DataFrame(
            [("000001.SZ", trade_date, 10.0, 11.0, 9.5, 10.5, 1000.0, 10500.0),
             ("600000.SH", trade_date, 20.0, 21.0, 19.0, 20.5, 0.0, 0.0)],
            columns=["ts_code", "trade_date", "open", "high", "low", "close", "vol", "amount"],
        )

    def adj_factor(self, trade_date=""):
        self.calls.append(("adj_factor", {"trade_date": trade_date}))
        return pd.DataFrame(
            [("000001.SZ", trade_date, 1.5), ("600000.SH", trade_date, 2.0)],
            columns=["ts_code", "trade_date", "adj_factor"],
        )

    def daily_basic(self, trade_date="", fields=""):
        self.calls.append(("daily_basic", {"trade_date": trade_date}))
        return pd.DataFrame(
            [("000001.SZ", trade_date, 1.0e6, 8.0e5),
             ("600000.SH", trade_date, 2.0e6, 1.6e6)],
            columns=["ts_code", "trade_date", "total_mv", "circ_mv"],
        )

    def trade_cal(self, exchange="SSE", start_date="", end_date=""):
        self.calls.append(("trade_cal", {}))
        return pd.DataFrame(
            [("SSE", "20240102", 1), ("SSE", "20240103", 1), ("SSE", "20240104", 0)],
            columns=["exchange", "cal_date", "is_open"],
        )


@pytest.fixture
def provider():
    p = TushareProvider(token="fake", request_sleep_sec=0.0)
    p._pro = FakePro()  # 直接注入，绕过真实 ts.pro_api
    return p


def test_get_stock_list_includes_delisted(provider):
    """list_status='L,D,P' 应拉全三种状态并拼接，含已退市股。"""
    df = provider.get_stock_list()
    assert set(df["instrument"]) == {"000001.SZ", "600000.SH", "000003.SZ"}
    # ts_code 已改名为 instrument
    assert "ts_code" not in df.columns
    # ST 推断
    assert bool(df.loc[df["instrument"] == "000003.SZ", "is_st"].iloc[0]) is True


def test_fetch_one_day_column_mapping(provider):
    """单日数据：vol->volume、ts_code->instrument、合并 adj_factor/market_cap/行业/停牌。"""
    day = provider.fetch_one_day("20240102")
    assert "volume" in day.columns and "vol" not in day.columns
    assert "instrument" in day.columns and "ts_code" not in day.columns
    assert "adj_factor" in day.columns
    assert "market_cap" in day.columns
    assert "industry" in day.columns
    # 600000 当日 vol=0 → 停牌
    row = day[day["instrument"] == "600000.SH"].iloc[0]
    assert bool(row["is_suspended"]) is True
    row2 = day[day["instrument"] == "000001.SZ"].iloc[0]
    assert bool(row2["is_suspended"]) is False


def test_get_daily_bar_uses_trade_cal(provider):
    """按 trade_cal 仅拉 is_open 的交易日（2 天），跳过休市日。"""
    df = provider.get_daily_bar("20240101", "20240131")
    # 2 个交易日 × 2 股
    assert len(df) == 4
    assert df["trade_date"].nunique() == 2


def test_update_daily_data_skip_existing(provider, tmp_path, monkeypatch):
    """断点续传：已缓存的交易日应被跳过。"""
    import qflab.data.tushare_provider as tp

    raw = tmp_path / "daily"

    def fake_list_raw_dates():
        return ["20240102"]

    saved = []

    def fake_save_raw_daily(df, trade_date):
        saved.append(trade_date)

    monkeypatch.setattr(tp, "list_raw_dates", fake_list_raw_dates)
    monkeypatch.setattr(tp, "save_raw_daily", fake_save_raw_daily)

    provider.update_daily_data("20240101", "20240131", skip_existing=True, build=False)
    # 20240102 已缓存被跳过，只保存 20240103
    assert saved == ["20240103"]


def test_call_with_retry_succeeds_after_failures(provider, monkeypatch):
    """前两次抛错、第三次成功，应返回成功结果而非抛出。"""
    monkeypatch.setattr("time.sleep", lambda *a, **k: None)
    calls = {"n": 0}

    def flaky(**kwargs):
        calls["n"] += 1
        if calls["n"] < 3:
            raise Exception("抱歉，您每分钟最多访问该接口")
        return pd.DataFrame({"ok": [1]})

    provider._pro.daily = flaky
    out = provider._call_with_retry("daily", trade_date="20240102")
    assert calls["n"] == 3
    assert out["ok"].iloc[0] == 1
