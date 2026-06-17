"""Winsorize（缩尾）。支持 MAD 与 quantile 两种方法。每日横截面执行。"""

from __future__ import annotations

import numpy as np
import pandas as pd


def winsorize_by_quantile(
    df: pd.DataFrame,
    lower: float = 0.01,
    upper: float = 0.99,
    value_col: str = "factor_value",
) -> pd.DataFrame:
    """每日横截面按分位数缩尾。"""
    if not 0 <= lower < upper <= 1:
        raise ValueError(f"invalid quantile bounds: lower={lower}, upper={upper}")

    out = df.copy()

    def _clip(s: pd.Series) -> pd.Series:
        if s.dropna().empty:
            return s
        lo, hi = s.quantile([lower, upper])
        return s.clip(lower=lo, upper=hi)

    out[value_col] = out.groupby("trade_date", group_keys=False)[value_col].apply(_clip)
    return out


def winsorize_by_mad(
    df: pd.DataFrame,
    n: float = 5.0,
    value_col: str = "factor_value",
) -> pd.DataFrame:
    """每日横截面按 MAD 缩尾。MAD = median(|x - median|), 上下界为 median ± n * 1.4826 * MAD。"""
    out = df.copy()

    def _clip(s: pd.Series) -> pd.Series:
        v = s.values.astype(float)
        mask = ~np.isnan(v)
        if mask.sum() < 3:
            return s
        med = np.median(v[mask])
        mad = np.median(np.abs(v[mask] - med))
        if mad == 0:
            return s
        lo = med - n * 1.4826 * mad
        hi = med + n * 1.4826 * mad
        return s.clip(lower=lo, upper=hi)

    out[value_col] = out.groupby("trade_date", group_keys=False)[value_col].apply(_clip)
    return out
