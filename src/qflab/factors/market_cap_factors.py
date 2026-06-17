"""市值类因子。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Factor


class LogMarketCap(Factor):
    """对数市值。"""

    name = "log_market_cap"

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        if "market_cap" not in df.columns:
            raise ValueError("LogMarketCap requires 'market_cap' column in daily_bar.")
        wide = df.pivot(index="trade_date", columns="instrument", values="market_cap").sort_index()
        wide = wide.where(wide > 0)
        wide = np.log(wide)
        return self.to_long(wide)
