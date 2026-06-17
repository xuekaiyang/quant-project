"""Streamlit UI：交互式因子评价。"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from qflab.data.storage import load_daily_bar, load_factor
from qflab.evaluation.report import EvaluationConfig, evaluate_factor
from qflab.factors import list_factors
from qflab.utils.config import load_config
from qflab.visualization.plots import (
    cumulative_ic_chart,
    ic_timeseries_chart,
    long_short_nav_chart,
    quantile_return_bar,
    turnover_chart,
)

st.set_page_config(page_title="qflab · Factor Lab", layout="wide")

st.title("Quant Factor Lab")
st.caption("⚠ 仅用于研究，不构成投资建议。")

cfg_global = load_config()


@st.cache_data(show_spinner=False)
def _available_factors() -> list[str]:
    base = cfg_global.paths.factor
    if not base.exists():
        return []
    return sorted([p.stem for p in base.glob("*.parquet")])


@st.cache_data(show_spinner=False)
def _load_bar_dates() -> tuple[pd.Timestamp, pd.Timestamp]:
    db = load_daily_bar()
    return db["trade_date"].min(), db["trade_date"].max()


with st.sidebar:
    st.header("Settings")
    saved = _available_factors()
    if not saved:
        st.warning("尚未发现已计算的因子。请先运行 `python scripts/compute_factors.py --factor all`。")
        candidate = list_factors()
    else:
        candidate = saved
    factor_name = st.selectbox("Factor", candidate)
    horizon = st.selectbox("Horizon (days)", [1, 5, 10, 20], index=1)

    try:
        d_min, d_max = _load_bar_dates()
        start_date = st.date_input("Start date", value=d_min.date(), min_value=d_min.date(), max_value=d_max.date())
        end_date = st.date_input("End date", value=d_max.date(), min_value=d_min.date(), max_value=d_max.date())
    except FileNotFoundError:
        st.error("daily_bar.parquet 不存在，请先运行 init_sample_data.py。")
        st.stop()

    quantiles = st.selectbox("Quantiles", [3, 5, 10], index=1)
    preprocess = st.multiselect(
        "Preprocess",
        ["winsorize_quantile", "winsorize_mad", "zscore", "neutralize"],
        default=["winsorize_quantile", "zscore"],
    )
    run = st.button("Run Evaluation", type="primary", use_container_width=True)

if not run:
    st.info("在左侧选择参数，点击 **Run Evaluation** 开始。")
    st.stop()

with st.spinner("Computing..."):
    try:
        factor_df = load_factor(factor_name)
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()
    cfg = EvaluationConfig(
        factor_name=factor_name,
        horizon=int(horizon),
        start_date=str(start_date),
        end_date=str(end_date),
        n_quantiles=int(quantiles),
        preprocess=list(preprocess),
    )
    res = evaluate_factor(cfg, factor_df=factor_df)

s = res.summary
c1, c2, c3, c4 = st.columns(4)
c1.metric("Rank IC mean", f"{s['ic']['rank']['ic_mean']:.4f}")
c2.metric("Rank ICIR", f"{s['ic']['rank']['icir']:.4f}" if s['ic']['rank']['icir'] == s['ic']['rank']['icir'] else "n/a")
c3.metric("L-S Sharpe", f"{s['long_short_portfolio']['sharpe']:.3f}")
c4.metric("L-S Max DD", f"{s['long_short_portfolio']['max_drawdown']:.3f}")

st.subheader("Summary JSON")
st.json(s)

c1, c2 = st.columns(2)
c1.plotly_chart(ic_timeseries_chart(res.ic_rank, "Rank IC Time Series"), use_container_width=True)
c2.plotly_chart(cumulative_ic_chart(res.ic_rank, "Cumulative Rank IC"), use_container_width=True)

c1, c2 = st.columns(2)
c1.plotly_chart(quantile_return_bar(res.quantile_returns, "Mean Quantile Return"), use_container_width=True)
c2.plotly_chart(long_short_nav_chart(res.nav, "Long-Short NAV"), use_container_width=True)

st.plotly_chart(turnover_chart(res.turnover, "Turnover"), use_container_width=True)
