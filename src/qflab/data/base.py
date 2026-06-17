"""DataProvider 抽象 + 公共 schema 常量。"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

REQUIRED_COLUMNS: tuple[str, ...] = (
    "trade_date",
    "instrument",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
)

OPTIONAL_COLUMNS: tuple[str, ...] = (
    "adj_factor",
    "industry",
    "market_cap",
    "is_st",
    "is_suspended",
)


class DataProvider(ABC):
    """统一数据源接口。所有 provider 返回标准化 DataFrame。"""

    name: str = "base"

    @abstractmethod
    def get_stock_list(self) -> pd.DataFrame:
        """返回股票列表。columns: instrument, name(optional), industry(optional), list_date(optional)."""

    @abstractmethod
    def get_daily_bar(self, start_date: str, end_date: str) -> pd.DataFrame:
        """返回日频行情。schema: REQUIRED_COLUMNS + 部分 OPTIONAL_COLUMNS。"""

    @abstractmethod
    def get_market_cap(self, start_date: str, end_date: str) -> pd.DataFrame:
        """返回市值数据。columns: trade_date, instrument, market_cap."""

    @abstractmethod
    def get_industry(self) -> pd.DataFrame:
        """返回行业分类。columns: instrument, industry."""

    @abstractmethod
    def update_daily_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """增量更新本地 daily_bar 数据，返回更新后的 DataFrame。"""
