"""tests for IC 衰减。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from qflab.evaluation.decay import ic_decay


def _make_decaying_factor(n_days=120, n_stocks=80, seed=0):
    """构造信号随 horizon 衰减的因子：
    次日收益 = 0.03*因子 + 噪声，之后每日收益独立噪声。
    → 1 日 IC 最强，horizon 越长（因子对应的那次冲击被稀释）IC 越弱。"""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-01", periods=n_days)
    insts = [f"S{i:03d}" for i in range(n_stocks)]
    fac_by_day = {d: dict(zip(insts, rng.normal(size=n_stocks))) for d in dates}
    price = {s: 100.0 for s in insts}
    bar_rows, fac_rows = [], []
    for di, d in enumerate(dates):
        for s in insts:
            fac_rows.append((d, s, fac_by_day[d][s]))
            bar_rows.append((d, s, price[s], price[s], price[s], price[s], 1e6, 1e8))
        if di < len(dates) - 1:
            for s in insts:
                ret = 0.03 * fac_by_day[d][s] + rng.normal(scale=0.01)
                price[s] *= (1 + ret)
    bar = pd.DataFrame(bar_rows, columns=[
        "trade_date", "instrument", "open", "high", "low", "close", "volume", "amount"])
    fac = pd.DataFrame(fac_rows, columns=["trade_date", "instrument", "factor_value"])
    fac["factor_name"] = "decaying"
    return fac, bar


def test_ic_decay_columns_and_horizons():
    fac, bar = _make_decaying_factor()
    out = ic_decay(fac, bar, horizons=[1, 3, 5, 10])
    assert list(out["horizon"]) == [1, 3, 5, 10]
    assert set(["ic_mean", "icir", "ic_pos_ratio", "n"]).issubset(out.columns)


def test_ic_decays_with_horizon():
    """1 日 IC 应显著强于 10 日 IC（单调衰减方向）。"""
    fac, bar = _make_decaying_factor()
    out = ic_decay(fac, bar, horizons=[1, 3, 5, 10]).set_index("horizon")
    assert abs(out.loc[1, "ic_mean"]) > abs(out.loc[10, "ic_mean"])
    # 1 日应为明显正相关
    assert out.loc[1, "ic_mean"] > 0.1


def test_ic_decay_handles_missing_gracefully():
    fac, bar = _make_decaying_factor(n_days=30, n_stocks=40)
    # horizon 超过样本长度时该行 n 应很小/为 0，但不报错
    out = ic_decay(fac, bar, horizons=[1, 100])
    assert len(out) == 2
