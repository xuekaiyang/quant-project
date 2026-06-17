"""因子预处理：winsorize / standardize / neutralize，以及 pipeline 串联。"""

from __future__ import annotations

from typing import Callable

import pandas as pd

from .neutralize import neutralize
from .standardize import zscore_by_date
from .winsorize import winsorize_by_mad, winsorize_by_quantile


def apply_pipeline(
    factor_df: pd.DataFrame,
    steps: list[str],
    daily_bar: pd.DataFrame | None = None,
    winsor_lower: float = 0.01,
    winsor_upper: float = 0.99,
    mad_n: float = 5.0,
    neutralize_industry: bool = True,
    neutralize_size: bool = True,
) -> pd.DataFrame:
    """按顺序应用预处理步骤。

    可选 step：
      - winsorize_quantile / winsorize_mad
      - zscore
      - neutralize
    """
    out = factor_df.copy()
    for step in steps:
        s = step.strip().lower()
        if s in ("winsorize", "winsorize_quantile"):
            out = winsorize_by_quantile(out, lower=winsor_lower, upper=winsor_upper)
        elif s == "winsorize_mad":
            out = winsorize_by_mad(out, n=mad_n)
        elif s == "zscore":
            out = zscore_by_date(out)
        elif s == "neutralize":
            if daily_bar is None:
                raise ValueError("neutralize step requires daily_bar.")
            out = neutralize(
                out,
                daily_bar,
                by_industry=neutralize_industry,
                by_log_market_cap=neutralize_size,
            )
        else:
            raise ValueError(f"Unknown preprocess step: {step}")
    return out


__all__ = [
    "winsorize_by_quantile",
    "winsorize_by_mad",
    "zscore_by_date",
    "neutralize",
    "apply_pipeline",
]
