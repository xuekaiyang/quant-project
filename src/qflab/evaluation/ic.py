"""IC（Information Coefficient）计算。"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def merge_factor_label(factor_df: pd.DataFrame, label_df: pd.DataFrame) -> pd.DataFrame:
    """对齐因子与标签。返回包含 trade_date/instrument/factor_value/label_value 的长表。"""
    f = factor_df[["trade_date", "instrument", "factor_value"]].copy()
    l = label_df[["trade_date", "instrument", "label_value"]].copy()
    f["trade_date"] = pd.to_datetime(f["trade_date"])
    l["trade_date"] = pd.to_datetime(l["trade_date"])
    merged = f.merge(l, on=["trade_date", "instrument"], how="inner")
    merged = merged.dropna(subset=["factor_value", "label_value"])
    return merged


def compute_daily_ic(merged: pd.DataFrame, method: str = "pearson", min_obs: int = 10) -> pd.DataFrame:
    """每日横截面 IC。method ∈ {'pearson', 'spearman'}.

    Returns: DataFrame[trade_date, ic, n_obs]。
    """
    if method not in ("pearson", "spearman"):
        raise ValueError(f"method must be pearson|spearman, got {method}")

    rows = []
    for d, g in merged.groupby("trade_date", sort=True):
        if len(g) < min_obs:
            rows.append((d, np.nan, len(g)))
            continue
        x = g["factor_value"].values
        y = g["label_value"].values
        try:
            if method == "pearson":
                r, _ = stats.pearsonr(x, y)
            else:
                r, _ = stats.spearmanr(x, y)
        except Exception:
            r = np.nan
        rows.append((d, float(r) if r is not None else np.nan, len(g)))
    return pd.DataFrame(rows, columns=["trade_date", "ic", "n_obs"])


def ic_summary(ic_series: pd.Series | pd.DataFrame) -> dict:
    """汇总 IC 序列。输入可为 Series 或含 'ic' 列的 DataFrame。"""
    s = ic_series["ic"] if isinstance(ic_series, pd.DataFrame) else ic_series
    s = pd.Series(s).dropna()
    if s.empty:
        return {
            "n": 0,
            "ic_mean": np.nan,
            "ic_std": np.nan,
            "icir": np.nan,
            "ic_t_stat": np.nan,
            "ic_pos_ratio": np.nan,
        }
    mean = float(s.mean())
    std = float(s.std(ddof=1)) if len(s) > 1 else np.nan
    icir = mean / std if std and not np.isnan(std) and std > 0 else np.nan
    t_stat = mean / (std / np.sqrt(len(s))) if std and std > 0 else np.nan
    return {
        "n": int(len(s)),
        "ic_mean": mean,
        "ic_std": float(std) if not np.isnan(std) else np.nan,
        "icir": float(icir) if not np.isnan(icir) else np.nan,
        "ic_t_stat": float(t_stat) if not np.isnan(t_stat) else np.nan,
        "ic_pos_ratio": float((s > 0).mean()),
    }


def compute_ic_full(
    factor_df: pd.DataFrame, label_df: pd.DataFrame, min_obs: int = 10
) -> dict:
    """同时返回 pearson IC、rank IC 序列及 summary。"""
    merged = merge_factor_label(factor_df, label_df)
    ic_p = compute_daily_ic(merged, "pearson", min_obs=min_obs)
    ic_r = compute_daily_ic(merged, "spearman", min_obs=min_obs)
    return {
        "ic_pearson": ic_p,
        "ic_rank": ic_r,
        "summary_pearson": ic_summary(ic_p),
        "summary_rank": ic_summary(ic_r),
    }
