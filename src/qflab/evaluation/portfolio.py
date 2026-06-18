"""组合 NAV 与绩效指标。"""

from __future__ import annotations

import numpy as np
import pandas as pd


def to_non_overlapping(returns: pd.Series, horizon: int) -> pd.Series:
    """把 horizon 日 forward-return 序列抽成非重叠序列。

    问题背景：当 horizon=n 时，每个交易日 T 的收益是 T→T+n 的 n 日收益，
    相邻交易日的收益窗口重叠 n-1 天。若直接对**每个交易日**的收益连乘求 NAV，
    等于每天新开一个持有 n 天的仓位，收益被重复计入约 n 倍，NAV/Sharpe/回撤全部失真。

    正确口径：每隔 n 个交易日取一个建仓点（T0, T0+n, T0+2n, ...），
    得到首尾相接、互不重叠的持有期收益序列，再做连乘/年化。

    Parameters
    ----------
    returns : pd.Series
        index 为 trade_date（已排序），值为 n 日 forward return。
    horizon : int
        持有期 n（>=1）。horizon=1 时即逐日不重叠，原样返回。

    Returns
    -------
    pd.Series
        非重叠收益序列（index 为被选中的建仓日）。
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >=1, got {horizon}")
    s = pd.Series(returns).dropna()
    if horizon == 1 or s.empty:
        return s
    s = s.sort_index()
    return s.iloc[::horizon]


def cumulative_nav(returns: pd.Series, init_nav: float = 1.0) -> pd.Series:
    """从每期收益序列构造净值曲线（连乘）。

    注意：传入的 returns 必须已是**非重叠**的持有期收益（见 to_non_overlapping），
    否则重叠窗口会让 NAV 失真。
    """
    r = pd.Series(returns).fillna(0.0)
    nav = (1.0 + r).cumprod() * init_nav
    return nav.rename("nav")


def annualized_return(returns: pd.Series, periods_per_year: int = 252) -> float:
    r = pd.Series(returns).dropna()
    if r.empty:
        return float("nan")
    nav_end = float((1.0 + r).prod())
    n = len(r)
    if nav_end <= 0:
        return float("nan")
    return float(nav_end ** (periods_per_year / n) - 1.0)


def annualized_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    r = pd.Series(returns).dropna()
    if len(r) < 2:
        return float("nan")
    return float(r.std(ddof=1) * np.sqrt(periods_per_year))


def sharpe_ratio(returns: pd.Series, periods_per_year: int = 252, rf: float = 0.0) -> float:
    r = pd.Series(returns).dropna() - rf / periods_per_year
    if len(r) < 2:
        return float("nan")
    sd = r.std(ddof=1)
    if sd == 0:
        return float("nan")
    return float(r.mean() / sd * np.sqrt(periods_per_year))


def max_drawdown(nav: pd.Series) -> float:
    n = pd.Series(nav).dropna()
    if n.empty:
        return float("nan")
    peak = n.cummax()
    dd = n / peak - 1.0
    return float(dd.min())


def win_rate(returns: pd.Series) -> float:
    r = pd.Series(returns).dropna()
    if r.empty:
        return float("nan")
    return float((r > 0).mean())


def portfolio_summary(
    returns: pd.Series,
    periods_per_year: int = 252,
    horizon: int = 1,
    already_non_overlapping: bool = False,
) -> dict:
    """汇总组合绩效。

    Parameters
    ----------
    returns : pd.Series
        每个交易日 T 的 horizon 日 forward return。
    periods_per_year : int
        每年交易日数（默认 252）。
    horizon : int
        持有期 n。年化频率为 periods_per_year / horizon。
    already_non_overlapping : bool
        若为 True，表示 returns 已是按调仓日程抽好的非重叠序列（如已扣成本的净收益），
        函数内部不再重采样，仅用 eff_ppy 做年化。否则内部按 horizon 抽非重叠序列。

    传入 horizon=1 时退化为逐日不重叠口径（向后兼容）。
    """
    ret_no = pd.Series(returns).dropna() if already_non_overlapping else to_non_overlapping(returns, horizon)
    eff_ppy = periods_per_year / horizon
    nav = cumulative_nav(ret_no)
    return {
        "n_periods": int(ret_no.shape[0]),
        "horizon": int(horizon),
        "annual_return": annualized_return(ret_no, eff_ppy),
        "annual_volatility": annualized_volatility(ret_no, eff_ppy),
        "sharpe": sharpe_ratio(ret_no, eff_ppy),
        "max_drawdown": max_drawdown(nav),
        "win_rate": win_rate(ret_no),
        "nav_end": float(nav.iloc[-1]) if not nav.empty else float("nan"),
    }


def apply_trading_cost(
    ls_ret: pd.Series,
    turnover_both_legs: pd.Series,
    cost_bps: float,
) -> pd.Series:
    """对多空收益按换手扣单边交易成本。

    Parameters
    ----------
    ls_ret : pd.Series
        多空持有期收益（index=调仓日）。
    turnover_both_legs : pd.Series
        每个调仓日两腿合计换手率（多腿换手 + 空腿换手），index 对齐 ls_ret。
        首个调仓日通常为 NaN（无前一日），按 0 成本处理（建仓成本可单列，这里从简）。
    cost_bps : float
        单边交易成本（基点）。如 10 表示 10bps = 0.001。

    Returns
    -------
    pd.Series
        扣成本后的净多空收益。
    """
    if cost_bps <= 0:
        return ls_ret
    cost = turnover_both_legs.reindex(ls_ret.index).fillna(0.0) * (cost_bps / 1e4)
    return (ls_ret - cost).rename("long_short_net")
