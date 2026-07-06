"""IC 衰减：同一因子在多个 horizon 上的 IC，刻画信号的持续性与最优持有期。

一个因子的预测力通常随持有期变化：有的信号快、只在短 horizon 有效，有的慢、
需要更长持有期才显现。把 IC 沿 horizon 画出来，能回答"信号能维持多久""最优持有期"。
"""

from __future__ import annotations

import pandas as pd

from ..labels.forward_returns import compute_forward_return_panel
from ..preprocess import apply_pipeline
from .ic import compute_daily_ic, ic_summary, merge_factor_label

DEFAULT_HORIZONS = [1, 3, 5, 10, 20]


def ic_decay(
    factor_df: pd.DataFrame,
    daily_bar: pd.DataFrame,
    horizons: list[int] | None = None,
    preprocess: list[str] | None = None,
    method: str = "spearman",
    min_obs: int = 10,
) -> pd.DataFrame:
    """计算因子在多个 horizon 上的 IC 摘要。

    Parameters
    ----------
    factor_df : long format 因子（trade_date/instrument/factor_value）。
    daily_bar : 行情（用于算 forward return 及可能的中性化）。
    horizons : 持有期列表，默认 [1,3,5,10,20]。
    preprocess : 预处理步骤（与单因子评价口径一致）；None 或空表示不预处理。
    method : "spearman"(rank IC) | "pearson"。
    min_obs : 每日截面最小样本数。

    Returns
    -------
    pd.DataFrame
        columns: horizon, ic_mean, icir, ic_pos_ratio, ic_t_stat, n
        按 horizon 升序。
    """
    hs = sorted(horizons or DEFAULT_HORIZONS)
    f = factor_df
    if preprocess:
        f = apply_pipeline(f, preprocess, daily_bar=daily_bar)

    panel = compute_forward_return_panel(daily_bar, hs)
    rows = []
    for h in hs:
        merged = merge_factor_label(f, panel[h])
        ic_df = compute_daily_ic(merged, method=method, min_obs=min_obs)
        s = ic_summary(ic_df)
        rows.append({
            "horizon": h,
            "ic_mean": s["ic_mean"],
            "icir": s["icir"],
            "ic_pos_ratio": s["ic_pos_ratio"],
            "ic_t_stat": s["ic_t_stat"],
            "n": s["n"],
        })
    return pd.DataFrame(rows, columns=["horizon", "ic_mean", "icir", "ic_pos_ratio", "ic_t_stat", "n"])
