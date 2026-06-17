"""中性化：每日横截面对行业哑变量与 log 市值做 OLS，取残差。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..utils.logger import get_logger

logger = get_logger(__name__)


def neutralize(
    factor_df: pd.DataFrame,
    daily_bar: pd.DataFrame,
    by_industry: bool = True,
    by_log_market_cap: bool = True,
    min_obs: int = 30,
    value_col: str = "factor_value",
) -> pd.DataFrame:
    """对因子做行业 + 对数市值中性化。

    Parameters
    ----------
    factor_df : pd.DataFrame
        long format：trade_date, instrument, factor_value (可能还有 factor_name)。
    daily_bar : pd.DataFrame
        提供 industry / market_cap 字段。
    by_industry, by_log_market_cap : bool
    min_obs : int
        每日有效样本数低于该值时跳过该日（残差填 NaN，不报错）。
    """
    if not (by_industry or by_log_market_cap):
        return factor_df.copy()

    cols = ["trade_date", "instrument"]
    have_ind = "industry" in daily_bar.columns
    have_mc = "market_cap" in daily_bar.columns
    if by_industry and not have_ind:
        logger.warning("neutralize: industry column missing, skip industry term.")
        by_industry = False
    if by_log_market_cap and not have_mc:
        logger.warning("neutralize: market_cap column missing, skip size term.")
        by_log_market_cap = False
    if not (by_industry or by_log_market_cap):
        return factor_df.copy()

    extra = cols + ([] if not by_industry else ["industry"]) + ([] if not by_log_market_cap else ["market_cap"])
    side = daily_bar[extra].drop_duplicates(subset=cols)
    merged = factor_df.merge(side, on=cols, how="left")

    if by_log_market_cap:
        merged["log_mc"] = np.log(merged["market_cap"].where(merged["market_cap"] > 0))

    out_chunks: list[pd.DataFrame] = []
    for d, g in merged.groupby("trade_date", sort=False):
        g = g.copy()
        y = g[value_col].astype(float).values
        mask = ~np.isnan(y)
        if by_log_market_cap:
            mask &= ~np.isnan(g["log_mc"].values)
        if by_industry:
            mask &= g["industry"].notna().values

        if mask.sum() < min_obs:
            g[value_col] = np.nan
            out_chunks.append(g.drop(columns=[c for c in ["log_mc", "industry", "market_cap"] if c in g.columns]))
            continue

        sub = g.loc[mask].copy()
        x_parts = []
        if by_industry:
            ind_dum = pd.get_dummies(sub["industry"], prefix="ind", drop_first=True, dtype=float)
            x_parts.append(ind_dum.values)
        if by_log_market_cap:
            x_parts.append(sub[["log_mc"]].values)
        X = np.hstack(x_parts) if x_parts else np.empty((mask.sum(), 0))
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        y_sub = sub[value_col].astype(float).values

        try:
            beta, *_ = np.linalg.lstsq(X, y_sub, rcond=None)
            resid = y_sub - X @ beta
        except np.linalg.LinAlgError:
            logger.warning("neutralize: lstsq failed at %s, set NaN.", d)
            resid = np.full_like(y_sub, np.nan, dtype=float)

        new_vals = np.full(len(g), np.nan, dtype=float)
        new_vals[mask] = resid
        g[value_col] = new_vals
        out_chunks.append(g.drop(columns=[c for c in ["log_mc", "industry", "market_cap"] if c in g.columns]))

    return pd.concat(out_chunks, ignore_index=True)
