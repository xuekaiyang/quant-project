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


def subperiod_ic_bar(subperiod: dict, title: str = "Rank IC by Sub-period") -> go.Figure:
    """子区间 Rank IC 柱状图。subperiod: {label: {'ic_rank_mean': ...}}。"""
    labels = list(subperiod.keys())
    vals = [subperiod[k].get("ic_rank_mean") for k in labels]
    colors = ["#5AD8A6" if (v is not None and v >= 0) else "#E86452" for v in vals]
    fig = go.Figure(go.Bar(x=labels, y=vals, marker_color=colors))
    fig.update_layout(title=title, xaxis_title="sub-period", yaxis_title="Rank IC mean", height=320)
    return fig


def ic_decay_chart(decay_df: pd.DataFrame, title: str = "IC Decay") -> go.Figure:
    """IC 衰减曲线：x=horizon, y=ic_mean（附 ICIR 次轴）。"""
    df = decay_df.sort_values("horizon")
    fig = go.Figure()
    fig.add_scatter(x=df["horizon"], y=df["ic_mean"], mode="lines+markers",
                    name="IC mean", line=dict(color="#5B8FF9"))
    fig.add_scatter(x=df["horizon"], y=df["icir"], mode="lines+markers",
                    name="ICIR", line=dict(color="#5AD8A6", dash="dot"), yaxis="y2")
    fig.update_layout(
        title=title, xaxis_title="horizon (days)", yaxis_title="IC mean",
        yaxis2=dict(title="ICIR", overlaying="y", side="right"), height=320,
    )
    return fig


def correlation_heatmap(corr: pd.DataFrame, title: str = "Factor Correlation") -> go.Figure:
    """因子相关性热力图。"""
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=list(corr.columns), y=list(corr.index),
        zmin=-1, zmax=1, colorscale="RdBu", reversescale=True,
        colorbar=dict(title="corr"),
    ))
    fig.update_layout(title=title, height=max(320, 24 * len(corr) + 120))
    return fig
