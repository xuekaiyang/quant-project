"""tests for IC 计算。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from qflab.evaluation.ic import compute_daily_ic, compute_ic_full, ic_summary, merge_factor_label


def _toy(n_days: int = 30, n_inst: int = 50, seed: int = 0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    rows_f, rows_l = [], []
    for d in dates:
        for i in range(n_inst):
            x = rng.normal()
            y = 0.6 * x + rng.normal(scale=0.5)
            rows_f.append((d, f"S{i:03d}", x))
            rows_l.append((d, f"S{i:03d}", y))
    f = pd.DataFrame(rows_f, columns=["trade_date", "instrument", "factor_value"])
    l = pd.DataFrame(rows_l, columns=["trade_date", "instrument", "label_value"])
    return f, l


def test_pearson_ic_positive_when_correlated():
    f, l = _toy()
    pack = compute_ic_full(f, l)
    assert pack["summary_pearson"]["ic_mean"] > 0.3
    assert pack["summary_rank"]["ic_mean"] > 0.3
    assert 0 <= pack["summary_rank"]["ic_pos_ratio"] <= 1


def test_ic_summary_handles_empty():
    s = ic_summary(pd.DataFrame({"ic": [np.nan, np.nan]}))
    assert s["n"] == 0
    assert np.isnan(s["ic_mean"])


def test_min_obs_threshold():
    """每日有效样本数过少时不计算 IC。"""
    f = pd.DataFrame({
        "trade_date": pd.to_datetime(["2024-01-02"] * 3),
        "instrument": ["A", "B", "C"],
        "factor_value": [1.0, 2.0, 3.0],
    })
    l = pd.DataFrame({
        "trade_date": pd.to_datetime(["2024-01-02"] * 3),
        "instrument": ["A", "B", "C"],
        "label_value": [0.1, 0.2, 0.3],
    })
    merged = merge_factor_label(f, l)
    ic = compute_daily_ic(merged, "pearson", min_obs=10)
    assert np.isnan(ic.iloc[0]["ic"])
