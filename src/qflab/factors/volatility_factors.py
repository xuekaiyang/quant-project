"""波动率因子。"""

from __future__ import annotations

import pandas as pd

from .base import Factor


class _VolatilityND(Factor):
    n: int = 20

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df.pivot(index="trade_date", columns="instrument", values="close").sort_index()
        ret = close.pct_change()
        wide = ret.rolling(self.n, min_periods=max(5, self.n // 2)).std()
        return self.to_long(wide)


class Volatility20D(_VolatilityND):
    name = "volatility_20d"
    n = 20


class Volatility60D(_VolatilityND):
    name = "volatility_60d"
    n = 60
