"""分组分析：按因子值横截面分位分组，计算每组未来收益。"""

from __future__ import annotations

import numpy as np
import pandas as pd


def assign_quantiles(merged: pd.DataFrame, n_quantiles: int = 5, value_col: str = "factor_value") -> pd.DataFrame:
    """每日横截面分位分组。返回带 'quantile' 列（1=最低，n=最高）的 DataFrame。

    使用 pd.qcut 等频分组；某天有效样本不足 n_quantiles 时跳过该日。
    """
    if n_quantiles < 2:
        raise ValueError(f"n_quantiles must be >=2, got {n_quantiles}")

    out_chunks: list[pd.DataFrame] = []
    for d, g in merged.groupby("trade_date", sort=False):
        v = g[value_col].values
        mask = ~np.isnan(v)
        if mask.sum() < n_quantiles:
            continue
        sub = g.loc[mask].copy()
        try:
            ranks = pd.qcut(sub[value_col], n_quantiles, labels=False, duplicates="drop")
        except ValueError:
            continue
        if ranks is None or ranks.isna().all():
            continue
        sub = sub.loc[ranks.notna()].copy()
        sub["quantile"] = ranks.dropna().astype(int).values + 1
        out_chunks.append(sub)
    if not out_chunks:
        return pd.DataFrame(columns=list(merged.columns) + ["quantile"])
    return pd.concat(out_chunks, ignore_index=True)


def quantile_returns(quantile_df: pd.DataFrame, n_quantiles: int = 5) -> pd.DataFrame:
    """计算每日每组等权未来收益。

    Returns
    -------
    pd.DataFrame
        index=trade_date, columns=quantile (1..n)
    """
    grp = quantile_df.groupby(["trade_date", "quantile"])["label_value"].mean().reset_index()
    wide = grp.pivot(index="trade_date", columns="quantile", values="label_value").sort_index()
    wide.columns.name = None
    return wide


def long_short_return(qret: pd.DataFrame, n_quantiles: int = 5) -> pd.Series:
    """top - bottom 多空收益。"""
    top = qret.get(n_quantiles)
    bot = qret.get(1)
    if top is None or bot is None:
        return pd.Series(dtype=float, name="long_short")
    return (top - bot).rename("long_short")


def quantile_summary(qret: pd.DataFrame, n_quantiles: int = 5) -> dict:
    """各组平均收益、long-short 平均收益与 t-stat。"""
    out = {
        "mean_by_quantile": {int(c): float(qret[c].mean()) for c in qret.columns if not qret[c].dropna().empty},
        "n_periods": int(len(qret)),
    }
    ls = long_short_return(qret, n_quantiles).dropna()
    if not ls.empty and ls.std(ddof=1) > 0:
        out["long_short_mean"] = float(ls.mean())
        out["long_short_t_stat"] = float(ls.mean() / (ls.std(ddof=1) / np.sqrt(len(ls))))
    else:
        out["long_short_mean"] = float(ls.mean()) if not ls.empty else np.nan
        out["long_short_t_stat"] = np.nan
    return out
