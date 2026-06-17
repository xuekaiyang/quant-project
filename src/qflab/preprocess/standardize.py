"""标准化（zscore by trade_date）。"""

from __future__ import annotations

import numpy as np
import pandas as pd


def zscore_by_date(df: pd.DataFrame, value_col: str = "factor_value") -> pd.DataFrame:
    """每日横截面 zscore。NaN 跳过。某天非 NaN 数小于 2 时该天置 NaN。"""
    out = df.copy()

    def _z(s: pd.Series) -> pd.Series:
        v = s.values.astype(float)
        mask = ~np.isnan(v)
        if mask.sum() < 2:
            return pd.Series(np.nan, index=s.index)
        mu = v[mask].mean()
        sd = v[mask].std(ddof=0)
        if sd == 0 or np.isnan(sd):
            return pd.Series(np.nan, index=s.index)
        return pd.Series((v - mu) / sd, index=s.index)

    out[value_col] = out.groupby("trade_date", group_keys=False)[value_col].apply(_z)
    return out
