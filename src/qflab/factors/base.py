"""Factor 抽象基类。

所有因子都基于规范化后的 daily_bar (long format) 计算，输出 long format：
trade_date, instrument, factor_name, factor_value
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class Factor(ABC):
    """因子抽象基类。"""

    name: str = ""
    dependencies: tuple[str, ...] = ()

    @abstractmethod
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """从规范化的 daily_bar 计算因子值。

        Parameters
        ----------
        df : pd.DataFrame
            long format daily bar，已按 (instrument, trade_date) 排序时使用。
            包含字段见 data.base.REQUIRED_COLUMNS / OPTIONAL_COLUMNS。

        Returns
        -------
        pd.DataFrame
            长表：trade_date, instrument, factor_name, factor_value。
        """

    def to_long(self, wide: pd.DataFrame) -> pd.DataFrame:
        """把 wide(date x instrument) 因子矩阵转换为 long format。"""
        long_df = (
            wide.stack(dropna=False)
            .rename("factor_value")
            .reset_index()
            .rename(columns={"level_0": "trade_date", "level_1": "instrument"})
        )
        long_df.columns = ["trade_date", "instrument", "factor_value"]
        long_df["factor_name"] = self.name
        long_df = long_df.dropna(subset=["factor_value"])
        return long_df[["trade_date", "instrument", "factor_name", "factor_value"]]
