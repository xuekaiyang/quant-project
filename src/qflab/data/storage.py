"""Parquet 存储 I/O。"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from ..utils.config import load_config
from ..utils.logger import get_logger
from .normalizer import normalize_daily_bar

logger = get_logger(__name__)

_RAW_DATE_RE = re.compile(r"^(\d{8})\.parquet$")


def _raw_daily_dir(dir_path: str | Path | None = None) -> Path:
    """raw 日分区目录，默认 data/raw/daily/。"""
    if dir_path is not None:
        return Path(dir_path)
    return load_config().paths.raw / "daily"


def save_daily_bar(df: pd.DataFrame, path: str | Path | None = None) -> Path:
    """保存日线数据到 parquet。"""
    cfg = load_config()
    p = Path(path) if path else cfg.paths.daily_bar
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False)
    logger.info("Saved daily_bar: %s rows=%d", p, len(df))
    return p


def load_daily_bar(path: str | Path | None = None) -> pd.DataFrame:
    """加载日线数据。"""
    cfg = load_config()
    p = Path(path) if path else cfg.paths.daily_bar
    if not p.exists():
        raise FileNotFoundError(
            f"Daily bar not found at {p}. "
            f"Run `python scripts/init_sample_data.py` first."
        )
    df = pd.read_parquet(p)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df


# ---------------------------------------------------------------------------
# raw 日分区存储：每个交易日一个 parquet，fetch 与 normalize 解耦，支持断点续传
# ---------------------------------------------------------------------------


def _normalize_date_key(trade_date) -> str:
    """把任意日期形态转成 'YYYYMMDD' 文件名键。"""
    return pd.Timestamp(trade_date).strftime("%Y%m%d")


def save_raw_daily(
    df: pd.DataFrame, trade_date, dir_path: str | Path | None = None
) -> Path:
    """保存单个交易日的原始行情（daily + adj_factor + daily_basic 合并后）。

    文件名为 <YYYYMMDD>.parquet。重复写入会覆盖当日文件，便于纠错重拉。
    """
    base = _raw_daily_dir(dir_path)
    base.mkdir(parents=True, exist_ok=True)
    key = _normalize_date_key(trade_date)
    p = base / f"{key}.parquet"
    df.to_parquet(p, index=False)
    logger.info("Saved raw daily %s: rows=%d -> %s", key, len(df), p)
    return p


def list_raw_dates(dir_path: str | Path | None = None) -> list[str]:
    """列出已落库的 raw 交易日（'YYYYMMDD' 升序）。目录不存在返回空列表。"""
    base = _raw_daily_dir(dir_path)
    if not base.exists():
        return []
    keys = []
    for f in base.iterdir():
        m = _RAW_DATE_RE.match(f.name)
        if m:
            keys.append(m.group(1))
    return sorted(keys)


def build_daily_bar_from_raw(
    dir_path: str | Path | None = None,
    out_path: str | Path | None = None,
    adjust: str | None = None,
) -> Path:
    """合并所有 raw 日文件 → normalize → 落库 daily_bar.parquet。

    Parameters
    ----------
    dir_path : raw 日分区目录，默认 data/raw/daily/。
    out_path : 输出路径，默认 cfg.paths.daily_bar。
    adjust : 透传给 normalize_daily_bar（None | 'qfq'）。

    Returns
    -------
    Path : 写出的 daily_bar.parquet 路径。
    """
    base = _raw_daily_dir(dir_path)
    keys = list_raw_dates(dir_path)
    if not keys:
        raise FileNotFoundError(
            f"No raw daily files under {base}. "
            f"Run `python scripts/update_daily_data.py --provider tushare ...` first."
        )
    frames = [pd.read_parquet(base / f"{k}.parquet") for k in keys]
    raw = pd.concat(frames, ignore_index=True)
    logger.info("build_daily_bar_from_raw: merged %d days, %d raw rows", len(keys), len(raw))
    normalized = normalize_daily_bar(raw, adjust=adjust)
    return save_daily_bar(normalized, out_path)


def save_factor(df: pd.DataFrame, factor_name: str, dir_path: str | Path | None = None) -> Path:
    """保存因子（long format: trade_date/instrument/factor_name/factor_value）。"""
    cfg = load_config()
    base = Path(dir_path) if dir_path else cfg.paths.factor
    base.mkdir(parents=True, exist_ok=True)
    p = base / f"{factor_name}.parquet"
    df.to_parquet(p, index=False)
    logger.info("Saved factor %s: rows=%d -> %s", factor_name, len(df), p)
    return p


def load_factor(factor_name: str, dir_path: str | Path | None = None) -> pd.DataFrame:
    """加载因子。"""
    cfg = load_config()
    base = Path(dir_path) if dir_path else cfg.paths.factor
    p = base / f"{factor_name}.parquet"
    if not p.exists():
        raise FileNotFoundError(
            f"Factor {factor_name} not found at {p}. "
            f"Run `python scripts/compute_factors.py --factor {factor_name}` first."
        )
    df = pd.read_parquet(p)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df
