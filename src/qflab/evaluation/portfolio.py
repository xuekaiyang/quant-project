"""组合 NAV 与绩效指标。"""

from __future__ import annotations

import numpy as np
import pandas as pd


def cumulative_nav(returns: pd.Series, init_nav: float = 1.0) -> pd.Series:
    """从每期收益序列构造净值曲线（连乘）。"""
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


def portfolio_summary(returns: pd.Series, periods_per_year: int = 252) -> dict:
    """汇总组合绩效。"""
    nav = cumulative_nav(returns)
    return {
        "n_periods": int(pd.Series(returns).dropna().shape[0]),
        "annual_return": annualized_return(returns, periods_per_year),
        "annual_volatility": annualized_volatility(returns, periods_per_year),
        "sharpe": sharpe_ratio(returns, periods_per_year),
        "max_drawdown": max_drawdown(nav),
        "win_rate": win_rate(returns),
        "nav_end": float(nav.iloc[-1]) if not nav.empty else float("nan"),
    }
