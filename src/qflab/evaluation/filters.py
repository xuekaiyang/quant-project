"""可交易过滤：在分组/建仓前剔除当日不可成交的股票。

口径：在因子观察日 T（即建仓日）检查可交易性。当日停牌、ST、或上市不足
``min_listed_days`` 的股票不纳入组合——它们要么买不到，要么有制度性风险/流动性问题，
若纳入会高估因子收益。涨跌停过滤需要逐日涨跌幅判断，留待后续（需 pre_close 字段）。
"""

from __future__ import annotations

import pandas as pd


def tradeable_mask(
    daily_bar: pd.DataFrame,
    exclude_suspended: bool = True,
    exclude_st: bool = True,
    min_listed_days: int = 0,
) -> pd.DataFrame:
    """从 daily_bar 计算每个 (trade_date, instrument) 的可交易标记。

    Returns
    -------
    pd.DataFrame
        trade_date, instrument, tradeable(bool)。
    """
    df = daily_bar[["trade_date", "instrument"]].copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    tradeable = pd.Series(True, index=daily_bar.index)

    if exclude_suspended and "is_suspended" in daily_bar.columns:
        tradeable &= ~daily_bar["is_suspended"].fillna(False).astype(bool)
    if exclude_st and "is_st" in daily_bar.columns:
        tradeable &= ~daily_bar["is_st"].fillna(False).astype(bool)
    if min_listed_days > 0 and "list_date" in daily_bar.columns:
        ld = pd.to_datetime(daily_bar["list_date"], errors="coerce")
        listed = (pd.to_datetime(daily_bar["trade_date"]) - ld).dt.days
        tradeable &= listed.fillna(0) >= min_listed_days

    df["tradeable"] = tradeable.values
    return df


def filter_tradeable(
    merged: pd.DataFrame,
    daily_bar: pd.DataFrame,
    exclude_suspended: bool = True,
    exclude_st: bool = True,
    min_listed_days: int = 0,
) -> pd.DataFrame:
    """在因子-标签合表上剔除建仓日不可交易的样本。

    Parameters
    ----------
    merged : 含 trade_date, instrument 的因子-标签合表（建仓日为 trade_date）。
    daily_bar : 含 is_suspended / is_st / (list_date) 的行情表。

    Returns
    -------
    过滤后的 merged（只保留 tradeable=True 的行）。
    """
    mask = tradeable_mask(
        daily_bar,
        exclude_suspended=exclude_suspended,
        exclude_st=exclude_st,
        min_listed_days=min_listed_days,
    )
    m = merged.copy()
    m["trade_date"] = pd.to_datetime(m["trade_date"])
    out = m.merge(mask, on=["trade_date", "instrument"], how="left")
    # daily_bar 中不存在的 (date,inst) 视为不可交易，保守剔除
    out = out[out["tradeable"].fillna(False)]
    return out.drop(columns=["tradeable"])
