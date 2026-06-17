"""换手率：相邻调仓日两个组合的成分变化比例。"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_turnover(quantile_assign: pd.DataFrame, n_quantiles: int = 5) -> pd.DataFrame:
    """计算 top / bottom / long-short 的每日换手率。

    Parameters
    ----------
    quantile_assign : pd.DataFrame
        来自 quantile.assign_quantiles，包含 trade_date, instrument, quantile。
    n_quantiles : int

    Returns
    -------
    pd.DataFrame
        index=trade_date, columns=['turnover_top','turnover_bottom','turnover_long_short']
    """
    top_q = n_quantiles
    bot_q = 1
    rows = []
    prev_top: set[str] | None = None
    prev_bot: set[str] | None = None
    for d, g in quantile_assign.groupby("trade_date", sort=True):
        cur_top = set(g.loc[g["quantile"] == top_q, "instrument"])
        cur_bot = set(g.loc[g["quantile"] == bot_q, "instrument"])
        if prev_top is None:
            t_top = np.nan
            t_bot = np.nan
        else:
            t_top = _turnover(prev_top, cur_top)
            t_bot = _turnover(prev_bot, cur_bot)
        ls = np.nan if np.isnan(t_top) or np.isnan(t_bot) else 0.5 * (t_top + t_bot)
        rows.append((d, t_top, t_bot, ls))
        prev_top, prev_bot = cur_top, cur_bot
    out = pd.DataFrame(
        rows, columns=["trade_date", "turnover_top", "turnover_bottom", "turnover_long_short"]
    ).set_index("trade_date")
    return out


def _turnover(prev: set[str], cur: set[str]) -> float:
    """单边换手率：1 - |prev ∩ cur| / max(|prev|, |cur|)。空集返回 NaN。"""
    if not prev and not cur:
        return np.nan
    inter = len(prev & cur)
    denom = max(len(prev), len(cur))
    if denom == 0:
        return np.nan
    return 1.0 - inter / denom
