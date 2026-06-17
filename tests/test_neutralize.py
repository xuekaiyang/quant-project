"""tests for neutralization。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from qflab.preprocess.neutralize import neutralize


def test_neutralize_removes_size_exposure():
    """构造 factor = a*log_mc + noise，中性化后残差与 log_mc 的相关性应接近 0。"""
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2024-01-01", periods=8)
    industries = ["A", "B", "C"]
    rows_bar, rows_factor = [], []
    for d in dates:
        for i in range(120):
            mc = float(np.exp(rng.normal(loc=22, scale=1.0)))
            ind = industries[i % 3]
            log_mc = np.log(mc)
            noise = rng.normal(scale=0.5)
            factor_val = 0.8 * log_mc + noise
            rows_bar.append((d, f"S{i:03d}", 10.0, mc, ind))
            rows_factor.append((d, f"S{i:03d}", factor_val))
    daily = pd.DataFrame(rows_bar, columns=["trade_date", "instrument", "close", "market_cap", "industry"])
    factor = pd.DataFrame(rows_factor, columns=["trade_date", "instrument", "factor_value"])

    raw_corr = abs(_corr_with_log_mc(factor, daily))
    out = neutralize(factor, daily, by_industry=True, by_log_market_cap=True)
    new_corr = abs(_corr_with_log_mc(out, daily))

    assert raw_corr > 0.5
    assert new_corr < 0.05


def _corr_with_log_mc(factor: pd.DataFrame, daily: pd.DataFrame) -> float:
    m = factor.merge(daily[["trade_date", "instrument", "market_cap"]], on=["trade_date", "instrument"])
    m["log_mc"] = np.log(m["market_cap"])
    m = m.dropna(subset=["factor_value", "log_mc"])
    return float(m[["factor_value", "log_mc"]].corr().iloc[0, 1])


def test_neutralize_handles_thin_day_gracefully():
    """某天样本不足 min_obs 时该日置 NaN，不应抛错。"""
    rng = np.random.default_rng(1)
    dates = pd.bdate_range("2024-01-01", periods=2)
    rows_bar, rows_factor = [], []
    for d in dates:
        n = 5 if d == dates[0] else 80
        for i in range(n):
            mc = float(np.exp(rng.normal(loc=22, scale=1.0)))
            rows_bar.append((d, f"S{i:03d}", 10.0, mc, "A"))
            rows_factor.append((d, f"S{i:03d}", rng.normal()))
    daily = pd.DataFrame(rows_bar, columns=["trade_date", "instrument", "close", "market_cap", "industry"])
    factor = pd.DataFrame(rows_factor, columns=["trade_date", "instrument", "factor_value"])
    out = neutralize(factor, daily, min_obs=30)
    thin_day = out[out["trade_date"] == dates[0]]
    assert thin_day["factor_value"].isna().all()
    fat_day = out[out["trade_date"] == dates[1]]
    assert fat_day["factor_value"].notna().any()
