"""tests for forward returns。"""

from __future__ import annotations

import pandas as pd

from qflab.labels.forward_returns import compute_forward_return


def _toy_bar() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=6)
    rows = []
    for d, c in zip(dates, [10, 11, 12, 13, 14, 15]):
        rows.append((d, "A", c))
    for d, c in zip(dates, [20, 19, 18, 17, 16, 15]):
        rows.append((d, "B", c))
    return pd.DataFrame(rows, columns=["trade_date", "instrument", "close"])


def test_forward_return_value_T_to_T_plus_n():
    """label[T] = close[T+n] / close[T] - 1。"""
    bar = _toy_bar()
    out = compute_forward_return(bar, horizon=2)
    a = out[out["instrument"] == "A"].sort_values("trade_date").reset_index(drop=True)

    expect_first = 12 / 10 - 1.0
    assert abs(a.loc[0, "label_value"] - expect_first) < 1e-12

    expect_4 = 15 / 13 - 1.0
    assert abs(a.iloc[3]["label_value"] - expect_4) < 1e-12


def test_forward_return_drops_tail_no_future_leak():
    """horizon=2 时，序列尾部最后 2 行应被丢弃（label 为 NaN）。"""
    bar = _toy_bar()
    out = compute_forward_return(bar, horizon=2)
    a = out[out["instrument"] == "A"].sort_values("trade_date")
    assert len(a) == 6 - 2
    assert a["trade_date"].max() == pd.Timestamp("2024-01-04")


def test_forward_return_horizon1_equals_pct_change_shift_neg1():
    """horizon=1 时与 close.pct_change().shift(-1) 等价。"""
    bar = _toy_bar()
    out = compute_forward_return(bar, horizon=1)
    b = out[out["instrument"] == "B"].sort_values("trade_date").reset_index(drop=True)
    assert abs(b.loc[0, "label_value"] - (19 / 20 - 1.0)) < 1e-12
