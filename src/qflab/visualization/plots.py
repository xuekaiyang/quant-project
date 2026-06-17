"""Plotly 图表。返回 plotly.graph_objects.Figure，便于在 Streamlit / HTML 报告中复用。"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go


def ic_timeseries_chart(ic_df: pd.DataFrame, title: str = "IC Time Series") -> go.Figure:
    """日度 IC 柱状图 + 移动平均线。"""
    df = ic_df.copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.sort_values("trade_date")
    fig = go.Figure()
    fig.add_bar(x=df["trade_date"], y=df["ic"], name="IC", marker_color="#5B8FF9", opacity=0.6)
    if len(df) > 20:
        ma = df["ic"].rolling(20, min_periods=5).mean()
        fig.add_scatter(x=df["trade_date"], y=ma, name="IC MA20", line=dict(color="#F6BD16"))
    fig.update_layout(title=title, xaxis_title="trade_date", yaxis_title="IC", height=380)
    return fig


def cumulative_ic_chart(ic_df: pd.DataFrame, title: str = "Cumulative IC") -> go.Figure:
    df = ic_df.copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.sort_values("trade_date")
    cum = df["ic"].fillna(0.0).cumsum()
    fig = go.Figure()
    fig.add_scatter(x=df["trade_date"], y=cum, mode="lines", line=dict(color="#5AD8A6"))
    fig.update_layout(title=title, xaxis_title="trade_date", yaxis_title="Cumulative IC", height=320)
    return fig


def quantile_return_bar(qret: pd.DataFrame, title: str = "Mean Quantile Return") -> go.Figure:
    means = qret.mean()
    fig = go.Figure()
    fig.add_bar(x=[f"Q{int(c)}" for c in means.index], y=means.values, marker_color="#5B8FF9")
    fig.update_layout(title=title, xaxis_title="Quantile", yaxis_title="Mean future return", height=320)
    return fig


def long_short_nav_chart(nav: pd.Series, title: str = "Long-Short Cumulative NAV") -> go.Figure:
    s = nav.copy()
    s.index = pd.to_datetime(s.index)
    fig = go.Figure()
    fig.add_scatter(x=s.index, y=s.values, mode="lines", line=dict(color="#F6BD16"))
    fig.update_layout(title=title, xaxis_title="trade_date", yaxis_title="NAV", height=380)
    return fig


def turnover_chart(turnover_df: pd.DataFrame, title: str = "Turnover") -> go.Figure:
    df = turnover_df.copy()
    df.index = pd.to_datetime(df.index)
    fig = go.Figure()
    for col, color in zip(
        ["turnover_top", "turnover_bottom", "turnover_long_short"],
        ["#5B8FF9", "#E86452", "#5AD8A6"],
    ):
        if col in df.columns:
            fig.add_scatter(x=df.index, y=df[col], mode="lines", name=col, line=dict(color=color))
    fig.update_layout(title=title, xaxis_title="trade_date", yaxis_title="Turnover", height=320)
    return fig
