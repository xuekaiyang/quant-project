"""因子相关性：识别池子里"说同一件事"的冗余因子。

口径：**每日横截面 rank 相关的时序均值**。
即每个交易日在截面上算因子两两(秩)相关，再对所有交易日取平均。这样避免不同日期
样本规模/量纲差异干扰，是因子研究的标准做法。相关性高的因子对是去重的候选。
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from ..preprocess import apply_pipeline


def _factor_wide(factor_df: pd.DataFrame) -> pd.DataFrame:
    """long -> wide(index=trade_date, columns=instrument)。"""
    df = factor_df.copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df.pivot_table(index="trade_date", columns="instrument", values="factor_value")


def factor_correlation(
    factor_values: dict[str, pd.DataFrame],
    method: str = "spearman",
    preprocess: list[str] | None = None,
    daily_bar: pd.DataFrame | None = None,
    min_obs: int = 10,
) -> pd.DataFrame:
    """计算多个因子的相关矩阵（每日横截面相关的时序均值）。

    Parameters
    ----------
    factor_values : {因子名 -> long df}。
    method : "spearman"(默认，rank 相关) | "pearson"。
    preprocess : 可选预处理（需 daily_bar 若含 neutralize）。
    min_obs : 每日截面计算相关所需的最小共同样本数。

    Returns
    -------
    pd.DataFrame
        对称相关矩阵，index/columns 为因子名，对角线为 1。
    """
    names = list(factor_values.keys())
    if len(names) < 2:
        raise ValueError("factor_correlation needs at least 2 factors")

    wides: dict[str, pd.DataFrame] = {}
    for name, fdf in factor_values.items():
        f = fdf
        if preprocess:
            f = apply_pipeline(f, preprocess, daily_bar=daily_bar)
        wides[name] = _factor_wide(f)

    corr = pd.DataFrame(np.eye(len(names)), index=names, columns=names)
    for a, b in itertools.combinations(names, 2):
        wa, wb = wides[a], wides[b]
        common_dates = wa.index.intersection(wb.index)
        daily_corrs = []
        for d in common_dates:
            sa = wa.loc[d]
            sb = wb.loc[d]
            pair = pd.concat([sa, sb], axis=1, keys=["a", "b"]).dropna()
            if len(pair) < min_obs:
                continue
            c = pair["a"].corr(pair["b"], method=method)
            if pd.notna(c):
                daily_corrs.append(c)
        mean_c = float(np.mean(daily_corrs)) if daily_corrs else np.nan
        corr.loc[a, b] = mean_c
        corr.loc[b, a] = mean_c
    return corr


def redundant_pairs(corr: pd.DataFrame, threshold: float = 0.8) -> list[tuple[str, str, float]]:
    """列出 |相关性| >= threshold 的因子对，按相关性绝对值降序。"""
    names = list(corr.columns)
    out = []
    for a, b in itertools.combinations(names, 2):
        c = corr.loc[a, b]
        if pd.notna(c) and abs(c) >= threshold:
            out.append((a, b, float(c)))
    out.sort(key=lambda t: abs(t[2]), reverse=True)
    return out
