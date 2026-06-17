"""日期工具。"""

from __future__ import annotations

from datetime import date, datetime

import pandas as pd

DATE_FMT = "%Y-%m-%d"


def to_date(x: str | date | datetime | pd.Timestamp | None) -> pd.Timestamp | None:
    """将各种类型转换为 pandas.Timestamp（仅日期部分）。None 透传。"""
    if x is None:
        return None
    return pd.Timestamp(x).normalize()


def to_date_str(x: str | date | datetime | pd.Timestamp) -> str:
    """转为 'YYYY-MM-DD' 字符串。"""
    return to_date(x).strftime(DATE_FMT)


def date_range(start: str | pd.Timestamp, end: str | pd.Timestamp, freq: str = "B") -> pd.DatetimeIndex:
    """生成日期范围。默认工作日（B），仅作为简化的交易日近似。"""
    return pd.date_range(start=to_date(start), end=to_date(end), freq=freq)
