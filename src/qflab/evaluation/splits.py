"""时序切分原语：IS/OOS holdout（带 purge+embargo）与子区间划分。

为什么需要 purge/embargo
------------------------
标签口径 label[T] = close[T+n]/close[T] - 1 用到了未来 n 天价格。若训练集末尾的
建仓日 T 与测试集起点相邻，训练标签会"看到"测试期的价格 → 数据泄露，OOS 评价虚高。

- **purge**：从训练集尾部剔除最后 horizon 个交易日的建仓点，保证训练标签窗口
  [T, T+horizon] 不侵入测试集。
- **embargo**：在 purge 基础上再留 embargo 个交易日缓冲，进一步隔离序列自相关的溢出。

本模块只处理"交易日序列 → 日期区间"，不接触因子/标签数据，便于独立测试。
"""

from __future__ import annotations

import pandas as pd


def _as_sorted_index(dates) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(pd.to_datetime(pd.Index(dates))).normalize()
    return idx.drop_duplicates().sort_values()


def train_test_split_dates(
    dates,
    test_ratio: float = 0.3,
    horizon: int = 1,
    embargo: int = 0,
) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    """按时间把交易日切成 (训练集, 测试集)，中间挖 purge+embargo 的间隔。

    Parameters
    ----------
    dates : 可转为日期的序列（交易日）。
    test_ratio : 测试集占比（0~1），取时间靠后的一段。
    horizon : 标签持有期 n，用于 purge。
    embargo : 额外缓冲交易日数。

    Returns
    -------
    (train_dates, test_dates) : 两个 DatetimeIndex。
        训练集尾部已剔除 horizon+embargo 个交易日，与测试集不重叠且留有 gap。
        样本过短导致训练集为空时，train 返回空 index。
    """
    if not 0.0 < test_ratio < 1.0:
        raise ValueError(f"test_ratio must be in (0,1), got {test_ratio}")
    if horizon < 1:
        raise ValueError(f"horizon must be >=1, got {horizon}")

    idx = _as_sorted_index(dates)
    n = len(idx)
    if n == 0:
        empty = pd.DatetimeIndex([])
        return empty, empty

    n_test = max(1, int(round(n * test_ratio)))
    split = n - n_test  # 测试集从位置 split 开始
    test_dates = idx[split:]

    # 训练集：切点前，再往前 purge horizon + embargo 个交易日
    gap = horizon + embargo
    train_end = split - gap
    train_dates = idx[:train_end] if train_end > 0 else pd.DatetimeIndex([])
    return train_dates, test_dates


def subperiod_ranges(dates, by: str = "year") -> list[tuple[str, pd.Timestamp, pd.Timestamp]]:
    """把交易日划成若干子区间，用于稳定性分析（各段独立统计，无需 purge）。

    Parameters
    ----------
    dates : 交易日序列。
    by : "year" 按自然年切；或传一个整数字符串（如 "3"）表示等分 K 段。

    Returns
    -------
    list of (label, start, end)：每段的标签与起止日期（含端点）。
    """
    idx = _as_sorted_index(dates)
    if len(idx) == 0:
        return []

    ranges: list[tuple[str, pd.Timestamp, pd.Timestamp]] = []
    if by == "year":
        for y, grp in pd.Series(idx).groupby(idx.year):
            ranges.append((str(int(y)), grp.min(), grp.max()))
        return ranges

    # 等分 K 段
    try:
        k = int(by)
    except (TypeError, ValueError):
        raise ValueError(f"subperiods must be 'year' or an integer string, got {by!r}")
    if k < 1:
        raise ValueError(f"n_splits must be >=1, got {k}")
    n = len(idx)
    for i in range(k):
        lo = (n * i) // k
        hi = (n * (i + 1)) // k
        if hi <= lo:
            continue
        seg = idx[lo:hi]
        ranges.append((f"seg{i + 1}", seg.min(), seg.max()))
    return ranges
