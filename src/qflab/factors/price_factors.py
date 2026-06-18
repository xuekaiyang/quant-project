"""价格类因子。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Factor


def _wide_close(df: pd.DataFrame) -> pd.DataFrame:
    """将 long daily_bar 转为 wide close 矩阵 (index=trade_date, columns=instrument)。"""
    return df.pivot(index="trade_date", columns="instrument", values="close").sort_index()


class _ReturnNDays(Factor):
    """N 日累计收益（不含未来）。"""

    n: int = 5

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        close = _wide_close(df)
        wide = close.pct_change(self.n, fill_method=None)
        return self.to_long(wide)


class Ret5D(_ReturnNDays):
    name = "ret_5d"
    n = 5


class Ret20D(_ReturnNDays):
    name = "ret_20d"
    n = 20


class Ret60D(_ReturnNDays):
    name = "ret_60d"
    n = 60


class Reverse1D(Factor):
    """1 日反转：取负的 1 日收益。"""

    name = "reverse_1d"

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        close = _wide_close(df)
        wide = -close.pct_change(1, fill_method=None)
        return self.to_long(wide)


class Reverse5D(Factor):
    name = "reverse_5d"

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        close = _wide_close(df)
        wide = -close.pct_change(5, fill_method=None)
        return self.to_long(wide)


class _CloseToMA(Factor):
    n: int = 20

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        close = _wide_close(df)
        ma = close.rolling(self.n, min_periods=max(2, self.n // 2)).mean()
        wide = close / ma - 1.0
        wide = wide.replace([np.inf, -np.inf], np.nan)
        return self.to_long(wide)


class CloseToMA20(_CloseToMA):
    name = "close_to_ma20"
    n = 20


class CloseToMA60(_CloseToMA):
    name = "close_to_ma60"
    n = 60
