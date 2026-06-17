"""成交量与成交额因子。"""

from __future__ import annotations

import pandas as pd

from .base import Factor


class _MeanND(Factor):
    n: int = 20
    field: str = "volume"

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        wide = df.pivot(index="trade_date", columns="instrument", values=self.field).sort_index()
        wide = wide.rolling(self.n, min_periods=max(5, self.n // 2)).mean()
        return self.to_long(wide)


class VolumeMean20D(_MeanND):
    name = "volume_mean_20d"
    n = 20
    field = "volume"


class AmountMean20D(_MeanND):
    name = "amount_mean_20d"
    n = 20
    field = "amount"
