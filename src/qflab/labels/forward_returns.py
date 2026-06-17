"""未来收益标签。

约定（已与用户确认）：
    label[T] = close[T+n] / close[T] - 1

也即 T 日收盘观察到因子值之后，假设 T 日收盘成交，持有到 T+n 日收盘平仓。
关键的防未来函数实现：
    label = close.shift(-n) / close - 1
而不是 close.shift(-n+1)。这里的 shift(-n) 是把未来值"拉回"到当前 T 行。

边界处理：
- 序列尾部最后 n 行 close.shift(-n) 为 NaN，这些行的 label 自然为 NaN，会被丢弃。
"""

from __future__ import annotations

import pandas as pd


def compute_forward_return(daily_bar: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """计算 T 到 T+n 的收益作为 T 日的标签。

    Parameters
    ----------
    daily_bar : pd.DataFrame
        long format，包含 trade_date, instrument, close。
    horizon : int
        n（>=1）。

    Returns
    -------
    pd.DataFrame
        long format：trade_date, instrument, horizon, label_value。
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >=1, got {horizon}")

    close = (
        daily_bar.pivot(index="trade_date", columns="instrument", values="close")
        .sort_index()
    )
    fwd = close.shift(-horizon) / close - 1.0

    long_df = (
        fwd.stack(dropna=False)
        .rename("label_value")
        .reset_index()
    )
    long_df.columns = ["trade_date", "instrument", "label_value"]
    long_df["horizon"] = horizon
    long_df = long_df.dropna(subset=["label_value"])
    return long_df[["trade_date", "instrument", "horizon", "label_value"]]


def compute_forward_return_panel(
    daily_bar: pd.DataFrame, horizons: list[int]
) -> dict[int, pd.DataFrame]:
    """批量计算多个 horizon 的 forward return。返回 dict[horizon -> long df]。"""
    return {h: compute_forward_return(daily_bar, h) for h in horizons}
