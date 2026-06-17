"""tests for quantile 分组。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from qflab.evaluation.quantile import (
    assign_quantiles,
    long_short_return,
    quantile_returns,
)


def test_quantile_assignment_groups_consistent():
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2024-01-01", periods=10)
    rows = []
    for d in dates:
        xs = rng.normal(size=50)
        for i, x in enumerate(xs):
            rows.append((d, f"S{i:03d}", x, 0.1 * x + rng.normal(scale=0.5)))
    df = pd.DataFrame(rows, columns=["trade_date", "instrument", "factor_value", "label_value"])
    qa = assign_quantiles(df, n_quantiles=5)

    counts = qa.groupby(["trade_date", "quantile"]).size().reset_index(name="n")
    # 每天每组应大致相等（10 个）
    assert counts["n"].between(9, 11).all()


def test_long_short_return_top_minus_bottom():
    """构造一个因子值=label 的强 IC 场景，top-bot 应为正。"""
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2024-01-01", periods=15)
    rows = []
    for d in dates:
        v = rng.normal(size=80)
        for i, x in enumerate(v):
            rows.append((d, f"S{i:03d}", x, x))  # label == factor
    df = pd.DataFrame(rows, columns=["trade_date", "instrument", "factor_value", "label_value"])
    qa = assign_quantiles(df, n_quantiles=5)
    qret = quantile_returns(qa, n_quantiles=5)
    ls = long_short_return(qret, n_quantiles=5).dropna()
    assert (ls > 0).all()


def test_quantile_with_nan_does_not_crash():
    df = pd.DataFrame({
        "trade_date": pd.to_datetime(["2024-01-02"] * 6),
        "instrument": list("ABCDEF"),
        "factor_value": [1.0, np.nan, 2.0, 3.0, np.nan, 4.0],
        "label_value": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    })
    qa = assign_quantiles(df, n_quantiles=4)
    assert "quantile" in qa.columns
    assert qa["factor_value"].notna().all()
