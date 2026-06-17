"""交易日历。MVP 版本使用工作日近似，A 股节假日由 provider 实际返回的数据决定。"""

from __future__ import annotations

import pandas as pd

from ..utils.dates import to_date


class TradingCalendar:
    """简化的交易日历。

    第一版用工作日（周一到周五）近似 A 股交易日。真实 A 股节假日通过实际数据
    剔除。当从 provider 拉到真实数据后，可用 ``from_dates`` 构造精确日历。
    """

    def __init__(self, dates: pd.DatetimeIndex | None = None):
        self._dates = dates

    @classmethod
    def from_dates(cls, dates) -> "TradingCalendar":
        """从已观察到的交易日序列构造日历。"""
        idx = pd.DatetimeIndex(sorted(set(pd.to_datetime(dates)))).normalize()
        return cls(idx)

    @classmethod
    def business_days(cls, start: str, end: str) -> "TradingCalendar":
        """构造 [start, end] 内的工作日日历（周一到周五）。"""
        idx = pd.bdate_range(to_date(start), to_date(end))
        return cls(idx)

    @classmethod
    def from_tushare(cls, pro, start: str, end: str, exchange: str = "SSE") -> "TradingCalendar":
        """用 Tushare ``pro.trade_cal`` 构造精确 A 股交易日历。

        Parameters
        ----------
        pro : tushare pro_api 实例（或任何带 ``trade_cal`` 方法的对象）。
        start, end : 'YYYYMMDD' 或可被 pandas 解析的日期。
        exchange : 交易所日历，默认 'SSE'（沪深交易日一致）。

        仅保留 ``is_open == 1`` 的交易日。
        """
        s = pd.Timestamp(start).strftime("%Y%m%d")
        e = pd.Timestamp(end).strftime("%Y%m%d")
        cal = pro.trade_cal(exchange=exchange, start_date=s, end_date=e)
        open_days = cal.loc[cal["is_open"] == 1, "cal_date"]
        idx = pd.DatetimeIndex(pd.to_datetime(open_days, format="%Y%m%d")).normalize()
        return cls(idx.sort_values())

    @property
    def dates(self) -> pd.DatetimeIndex:
        if self._dates is None:
            raise ValueError("Calendar has no dates loaded.")
        return self._dates

    def shift(self, dt: pd.Timestamp, n: int) -> pd.Timestamp | None:
        """返回 dt 后的第 n 个交易日（n 可为负）。越界返回 None。"""
        d = to_date(dt)
        idx = self.dates
        if d not in idx:
            pos = idx.searchsorted(d)
        else:
            pos = idx.get_loc(d)
        new_pos = pos + n
        if new_pos < 0 or new_pos >= len(idx):
            return None
        return idx[new_pos]
