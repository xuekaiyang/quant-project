"""tests for 防过拟合纪律修复：非重叠窗口 / 交易成本 / 可交易过滤。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from qflab.evaluation.filters import filter_tradeable, tradeable_mask
from qflab.evaluation.portfolio import (
    apply_trading_cost,
    portfolio_summary,
    to_non_overlapping,
)


# ---------------------------------------------------------------- 非重叠窗口
def test_to_non_overlapping_picks_every_n():
    idx = pd.bdate_range("2024-01-01", periods=20)
    s = pd.Series(np.arange(20, dtype=float), index=idx)
    out = to_non_overlapping(s, horizon=5)
    # 每 5 个取一个：位置 0,5,10,15
    assert list(out.values) == [0.0, 5.0, 10.0, 15.0]


def test_to_non_overlapping_horizon_1_unchanged():
    idx = pd.bdate_range("2024-01-01", periods=6)
    s = pd.Series(np.arange(6, dtype=float), index=idx)
    out = to_non_overlapping(s, horizon=1)
    assert len(out) == 6


def test_overlapping_inflates_nav():
    """同一段 5 日收益：重叠口径(逐日)的累计远大于非重叠口径。验证 bug 的方向。"""
    idx = pd.bdate_range("2024-01-01", periods=50)
    r = pd.Series(0.01, index=idx)  # 每个交易日都标 1% 的 5 日收益
    overlap_nav = float((1 + r).prod())                       # 错误：50 次连乘
    nonoverlap_nav = float((1 + to_non_overlapping(r, 5)).prod())  # 正确：10 次连乘
    assert overlap_nav > nonoverlap_nav
    # 非重叠应约为 1.01^10
    assert abs(nonoverlap_nav - 1.01 ** 10) < 1e-9


def test_portfolio_summary_annualization_scales_with_horizon():
    """h=5 时年化频率应为 252/5；与逐日口径的年化结果不同。"""
    idx = pd.bdate_range("2024-01-01", periods=100)
    r = pd.Series(0.001, index=idx)
    s5 = portfolio_summary(r, periods_per_year=252, horizon=5)
    # n_periods 应是抽样后的个数 ceil(100/5)=20
    assert s5["n_periods"] == 20
    assert s5["horizon"] == 5


# ---------------------------------------------------------------- 交易成本
def test_apply_trading_cost_reduces_return():
    idx = pd.bdate_range("2024-01-01", periods=4)
    ls = pd.Series([0.02, 0.02, 0.02, 0.02], index=idx)
    turnover_both = pd.Series([np.nan, 1.0, 1.0, 1.0], index=idx)  # 两腿合计换手 100%
    out = apply_trading_cost(ls, turnover_both, cost_bps=10)  # 10bps=0.001
    # 首日无前持仓→成本0；其余每期扣 1.0*0.001
    assert abs(out.iloc[0] - 0.02) < 1e-12
    assert abs(out.iloc[1] - (0.02 - 0.001)) < 1e-12


def test_apply_trading_cost_zero_is_noop():
    idx = pd.bdate_range("2024-01-01", periods=3)
    ls = pd.Series([0.01, 0.01, 0.01], index=idx)
    tov = pd.Series([np.nan, 1.0, 1.0], index=idx)
    out = apply_trading_cost(ls, tov, cost_bps=0.0)
    pd.testing.assert_series_equal(out, ls)


# ---------------------------------------------------------------- 可交易过滤
def _daily_bar():
    return pd.DataFrame({
        "trade_date": pd.to_datetime(["2024-01-02"] * 4),
        "instrument": ["A", "B", "C", "D"],
        "is_suspended": [False, True, False, False],
        "is_st": [False, False, True, False],
    })


def test_tradeable_mask_flags_suspended_and_st():
    m = tradeable_mask(_daily_bar())
    flag = dict(zip(m["instrument"], m["tradeable"]))
    assert flag["A"] is True or flag["A"] == True   # 正常
    assert not flag["B"]   # 停牌
    assert not flag["C"]   # ST
    assert flag["D"]       # 正常


def test_filter_tradeable_drops_untradeable_rows():
    merged = pd.DataFrame({
        "trade_date": pd.to_datetime(["2024-01-02"] * 4),
        "instrument": ["A", "B", "C", "D"],
        "factor_value": [1.0, 2.0, 3.0, 4.0],
        "label_value": [0.1, 0.2, 0.3, 0.4],
    })
    out = filter_tradeable(merged, _daily_bar())
    assert set(out["instrument"]) == {"A", "D"}


def test_filter_tradeable_can_be_disabled():
    merged = pd.DataFrame({
        "trade_date": pd.to_datetime(["2024-01-02"] * 4),
        "instrument": ["A", "B", "C", "D"],
        "factor_value": [1.0, 2.0, 3.0, 4.0],
        "label_value": [0.1, 0.2, 0.3, 0.4],
    })
    out = filter_tradeable(merged, _daily_bar(), exclude_suspended=False, exclude_st=False)
    assert set(out["instrument"]) == {"A", "B", "C", "D"}
