"""数据规范化：把任意来源的原始 DataFrame 统一成内部 schema。"""

from __future__ import annotations

import pandas as pd

from ..utils.logger import get_logger
from .base import REQUIRED_COLUMNS

logger = get_logger(__name__)


def normalize_daily_bar(
    df: pd.DataFrame, drop_invalid: bool = True, adjust: str | None = None
) -> pd.DataFrame:
    """规范化日线 DataFrame。

    Parameters
    ----------
    df : pd.DataFrame
        原始数据，至少包含 REQUIRED_COLUMNS 中的字段。
    drop_invalid : bool
        是否丢弃 close<=0 或缺失 OHLC 的行。
    adjust : str | None
        复权方式。None 表示不复权（保留原始 OHLC）。'qfq' 表示前复权：
        用 adj_factor 把 OHLC 换算成前复权价，原始价格保留到 open_raw/high_raw/
        low_raw/close_raw，下游价格类因子无需改动即可使用前复权价。
        adjust='qfq' 但缺少 adj_factor 列时报错。

    Returns
    -------
    pd.DataFrame
        规范化后的 DataFrame，按 (trade_date, instrument) 升序排序。
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"normalize_daily_bar: missing required columns: {missing}")

    out = df.copy()
    out["trade_date"] = pd.to_datetime(out["trade_date"]).dt.normalize()
    out["instrument"] = out["instrument"].astype(str)

    for col in ["open", "high", "low", "close", "volume", "amount"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    for col in ["adj_factor", "market_cap"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    for col in ["is_st", "is_suspended"]:
        if col in out.columns:
            out[col] = out[col].astype(bool)

    if drop_invalid:
        n0 = len(out)
        out = out.dropna(subset=["open", "high", "low", "close"])
        out = out[out["close"] > 0]
        n1 = len(out)
        if n0 - n1 > 0:
            logger.info("normalize_daily_bar: dropped %d invalid rows", n0 - n1)

    out = out.drop_duplicates(subset=["trade_date", "instrument"], keep="last")
    out = out.sort_values(["trade_date", "instrument"]).reset_index(drop=True)

    if adjust:
        out = _apply_adjust(out, adjust)
    return out


def _apply_adjust(out: pd.DataFrame, adjust: str) -> pd.DataFrame:
    """对 OHLC 应用复权。当前支持前复权 'qfq'。

    前复权价 = raw_price * adj_factor / 该股票最新一日 adj_factor。
    最新日复权因子归一，越往前价格越被缩放，保证最近价格与真实成交价一致。
    """
    if adjust != "qfq":
        raise ValueError(f"normalize_daily_bar: unsupported adjust={adjust!r} (only 'qfq')")
    if "adj_factor" not in out.columns:
        raise ValueError("normalize_daily_bar: adjust='qfq' requires an 'adj_factor' column")

    price_cols = ["open", "high", "low", "close"]
    # 每只股票按 trade_date 取最新一日的 adj_factor 作为归一基准
    latest_af = out.groupby("instrument")["adj_factor"].transform("last")
    ratio = out["adj_factor"] / latest_af

    for col in price_cols:
        out[f"{col}_raw"] = out[col]
        out[col] = out[col] * ratio
    n_bad = ratio.isna().sum()
    if n_bad:
        logger.info("normalize_daily_bar: qfq ratio NaN on %d rows (missing adj_factor)", n_bad)
    return out
