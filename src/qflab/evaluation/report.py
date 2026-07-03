"""端到端因子评价报告。

汇总 IC、分组、组合、换手，输出：
  - summary json
  - detail parquet（IC 序列 / 分组收益 / NAV / 换手）
  - HTML 报告（plotly 拼接）
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from ..data.storage import load_daily_bar, load_factor
from ..labels.forward_returns import compute_forward_return
from ..preprocess import apply_pipeline
from ..utils.config import load_config
from ..utils.logger import get_logger
from ..visualization.plots import (
    cumulative_ic_chart,
    ic_timeseries_chart,
    long_short_nav_chart,
    quantile_return_bar,
    subperiod_ic_bar,
    turnover_chart,
)
from .ic import compute_ic_full, merge_factor_label
from .filters import filter_tradeable
from .portfolio import apply_trading_cost, cumulative_nav, portfolio_summary, to_non_overlapping
from .quantile import (
    assign_quantiles,
    long_short_return,
    quantile_returns,
    quantile_summary,
)
from .splits import subperiod_ranges, train_test_split_dates
from .turnover import compute_turnover

logger = get_logger(__name__)


@dataclass
class EvaluationConfig:
    factor_name: str
    horizon: int = 5
    start_date: str | None = None
    end_date: str | None = None
    n_quantiles: int = 5
    preprocess: list[str] = field(default_factory=list)
    annualization_factor: int = 252
    # 防过拟合纪律（默认开启正确口径）
    trading_cost_bps: float = 0.0       # 单边交易成本（基点），按换手扣减
    exclude_suspended: bool = True      # 建仓日停牌剔除
    exclude_st: bool = True             # ST 剔除
    min_listed_days: int = 0            # 上市天数过滤（需 list_date 字段，默认关）
    # 时序切分（防过拟合）
    oos_test_ratio: float = 0.0         # >0 时启用 IS/OOS holdout，取靠后一段做 OOS
    embargo: int = 0                    # IS/OOS 之间额外缓冲交易日数（purge 已含 horizon）
    subperiods: str | None = None       # "year" 或整数字符串（等分K段）；None 关闭


@dataclass
class EvaluationResult:
    config: EvaluationConfig
    summary: dict
    ic_pearson: pd.DataFrame
    ic_rank: pd.DataFrame
    quantile_returns: pd.DataFrame
    long_short_returns: pd.Series
    nav: pd.Series
    turnover: pd.DataFrame
    is_oos: dict | None = None          # {'train': {...}, 'test': {...}}
    subperiod: dict | None = None       # {label: {...}} 各子区间精简指标
    output_dir: Path | None = None


def _filter_dates(df: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    out = df.copy()
    out["trade_date"] = pd.to_datetime(out["trade_date"])
    if start:
        out = out[out["trade_date"] >= pd.to_datetime(start)]
    if end:
        out = out[out["trade_date"] <= pd.to_datetime(end)]
    return out


def _evaluate_core(
    cfg: EvaluationConfig,
    factor_df: pd.DataFrame,
    label_df: pd.DataFrame,
    daily_bar: pd.DataFrame,
    entry_dates: pd.DatetimeIndex | None = None,
) -> dict:
    """对给定（已预处理的）因子/标签，在指定建仓日子集上跑完整评价。

    entry_dates 为 None 时用全样本；否则只保留 trade_date ∈ entry_dates 的建仓点。
    返回含 ic_pack / qret / ls_ret_net / nav / tov 及各 summary 分块的 dict。
    """
    f = factor_df
    l = label_df
    if entry_dates is not None:
        ed = pd.DatetimeIndex(entry_dates)
        f = f[pd.to_datetime(f["trade_date"]).isin(ed)]
        l = l[pd.to_datetime(l["trade_date"]).isin(ed)]

    ic_pack = compute_ic_full(f, l)

    merged = merge_factor_label(f, l)
    merged = filter_tradeable(
        merged,
        daily_bar,
        exclude_suspended=cfg.exclude_suspended,
        exclude_st=cfg.exclude_st,
        min_listed_days=cfg.min_listed_days,
    )

    qa = assign_quantiles(merged, n_quantiles=cfg.n_quantiles)
    qret = quantile_returns(qa, n_quantiles=cfg.n_quantiles)
    ls_ret_daily = long_short_return(qret, n_quantiles=cfg.n_quantiles)

    # 非重叠调仓日程，避免重叠窗口重复计入收益
    ls_ret = to_non_overlapping(ls_ret_daily, cfg.horizon)
    rebalance_dates = ls_ret.index

    qa_rebal = qa[pd.to_datetime(qa["trade_date"]).isin(rebalance_dates)]
    tov = compute_turnover(qa_rebal, n_quantiles=cfg.n_quantiles)
    turnover_both = (tov["turnover_top"] + tov["turnover_bottom"]) if not tov.empty else pd.Series(dtype=float)

    ls_ret_net = apply_trading_cost(ls_ret, turnover_both, cfg.trading_cost_bps)
    nav = cumulative_nav(ls_ret_net)

    return {
        "ic_pack": ic_pack,
        "qret": qret,
        "ls_ret_net": ls_ret_net,
        "nav": nav,
        "tov": tov,
        "ic": {"pearson": ic_pack["summary_pearson"], "rank": ic_pack["summary_rank"]},
        "quantile": quantile_summary(qret, n_quantiles=cfg.n_quantiles),
        "long_short_portfolio": portfolio_summary(
            ls_ret_net,
            periods_per_year=cfg.annualization_factor,
            horizon=cfg.horizon,
            already_non_overlapping=True,
        ),
        "turnover": {
            "top_mean": float(tov["turnover_top"].mean()) if not tov.empty else float("nan"),
            "bottom_mean": float(tov["turnover_bottom"].mean()) if not tov.empty else float("nan"),
            "long_short_mean": float(tov["turnover_long_short"].mean()) if not tov.empty else float("nan"),
        },
    }


def _segment_digest(core: dict) -> dict:
    """从 _evaluate_core 结果抽取精简指标，用于 IS/OOS 与子区间对比。"""
    r = core["ic"]["rank"]
    pf = core["long_short_portfolio"]
    return {
        "n_ic_days": r.get("n"),
        "ic_rank_mean": r.get("ic_mean"),
        "icir_rank": r.get("icir"),
        "ic_pos_ratio": r.get("ic_pos_ratio"),
        "ls_sharpe": pf.get("sharpe"),
        "ls_annual_return": pf.get("annual_return"),
        "n_rebalance": pf.get("n_periods"),
    }


def evaluate_factor(
    cfg: EvaluationConfig,
    factor_df: pd.DataFrame | None = None,
    daily_bar: pd.DataFrame | None = None,
) -> EvaluationResult:
    """运行端到端评价。如未传入 factor_df / daily_bar，则从默认存储路径加载。"""
    if daily_bar is None:
        daily_bar = load_daily_bar()
    if factor_df is None:
        factor_df = load_factor(cfg.factor_name)

    daily_bar = _filter_dates(daily_bar, cfg.start_date, cfg.end_date)
    factor_df = _filter_dates(factor_df, cfg.start_date, cfg.end_date)

    if cfg.preprocess:
        factor_df = apply_pipeline(factor_df, cfg.preprocess, daily_bar=daily_bar)

    label_df = compute_forward_return(daily_bar, cfg.horizon)
    label_df = _filter_dates(label_df, cfg.start_date, cfg.end_date)

    # 全样本评价
    full = _evaluate_core(cfg, factor_df, label_df, daily_bar, entry_dates=None)

    summary = {
        "factor_name": cfg.factor_name,
        "horizon": cfg.horizon,
        "start_date": cfg.start_date,
        "end_date": cfg.end_date,
        "n_quantiles": cfg.n_quantiles,
        "preprocess": list(cfg.preprocess),
        "trading_cost_bps": cfg.trading_cost_bps,
        "filters": {
            "exclude_suspended": cfg.exclude_suspended,
            "exclude_st": cfg.exclude_st,
            "min_listed_days": cfg.min_listed_days,
        },
        "ic": full["ic"],
        "quantile": full["quantile"],
        "long_short_portfolio": full["long_short_portfolio"],
        "turnover": full["turnover"],
    }

    # IS/OOS holdout
    is_oos = None
    all_entry_dates = pd.DatetimeIndex(sorted(pd.to_datetime(factor_df["trade_date"]).unique()))
    if cfg.oos_test_ratio > 0.0:
        train_d, test_d = train_test_split_dates(
            all_entry_dates,
            test_ratio=cfg.oos_test_ratio,
            horizon=cfg.horizon,
            embargo=cfg.embargo,
        )
        train_core = _evaluate_core(cfg, factor_df, label_df, daily_bar, entry_dates=train_d)
        test_core = _evaluate_core(cfg, factor_df, label_df, daily_bar, entry_dates=test_d)
        is_oos = {
            "train": {
                "start": str(train_d.min().date()) if len(train_d) else None,
                "end": str(train_d.max().date()) if len(train_d) else None,
                **_segment_digest(train_core),
            },
            "test": {
                "start": str(test_d.min().date()) if len(test_d) else None,
                "end": str(test_d.max().date()) if len(test_d) else None,
                **_segment_digest(test_core),
            },
            "embargo": cfg.embargo,
            "purge_horizon": cfg.horizon,
        }
        summary["is_oos"] = is_oos

    # 子区间稳定性
    subperiod = None
    if cfg.subperiods:
        subperiod = {}
        for label, start, end in subperiod_ranges(all_entry_dates, by=cfg.subperiods):
            seg_dates = all_entry_dates[(all_entry_dates >= start) & (all_entry_dates <= end)]
            seg_core = _evaluate_core(cfg, factor_df, label_df, daily_bar, entry_dates=seg_dates)
            subperiod[label] = {
                "start": str(start.date()),
                "end": str(end.date()),
                **_segment_digest(seg_core),
            }
        summary["subperiods"] = subperiod

    return EvaluationResult(
        config=cfg,
        summary=summary,
        ic_pearson=full["ic_pack"]["ic_pearson"],
        ic_rank=full["ic_pack"]["ic_rank"],
        quantile_returns=full["qret"],
        long_short_returns=full["ls_ret_net"],
        nav=full["nav"],
        turnover=full["tov"],
        is_oos=is_oos,
        subperiod=subperiod,
    )


def save_report(result: EvaluationResult, output_dir: Path | str | None = None) -> Path:
    """把评价结果落盘：summary.json / *.parquet / report.html。"""
    cfg_paths = load_config().paths
    base = Path(output_dir) if output_dir else (
        cfg_paths.evaluation / f"{result.config.factor_name}_h{result.config.horizon}"
    )
    base.mkdir(parents=True, exist_ok=True)

    (base / "summary.json").write_text(
        json.dumps(result.summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    result.ic_pearson.to_parquet(base / "ic_pearson.parquet", index=False)
    result.ic_rank.to_parquet(base / "ic_rank.parquet", index=False)
    result.quantile_returns.to_parquet(base / "quantile_returns.parquet")
    result.long_short_returns.to_frame("long_short").to_parquet(base / "long_short.parquet")
    result.nav.to_frame("nav").to_parquet(base / "nav.parquet")
    result.turnover.to_parquet(base / "turnover.parquet")

    html_path = base / "report.html"
    _render_html(result, html_path)
    result.output_dir = base
    logger.info("Saved evaluation report to %s", base)
    return base


def _fig_to_div(fig: go.Figure) -> str:
    return fig.to_html(full_html=False, include_plotlyjs=False)


def _render_html(result: EvaluationResult, path: Path) -> None:
    cfg = result.config
    s = result.summary
    figs = [
        ic_timeseries_chart(result.ic_rank, "Rank IC Time Series"),
        cumulative_ic_chart(result.ic_rank, "Cumulative Rank IC"),
        quantile_return_bar(result.quantile_returns, "Mean Quantile Return"),
        long_short_nav_chart(result.nav, "Long-Short NAV"),
        turnover_chart(result.turnover, "Turnover"),
    ]
    if result.subperiod:
        figs.append(subperiod_ic_bar(result.subperiod, "Rank IC by Sub-period"))
    body = "\n".join(_fig_to_div(f) for f in figs)
    tbl = _summary_table_html(s)
    is_oos_html = _is_oos_table_html(result.is_oos) if result.is_oos else ""
    subp_html = _subperiod_table_html(result.subperiod) if result.subperiod else ""
    html = f"""<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<title>{cfg.factor_name} - h{cfg.horizon} report</title>
<script src="https://cdn.plot.ly/plotly-2.30.0.min.js"></script>
<style>
body {{ font-family: -apple-system, Segoe UI, Helvetica, Arial; margin: 24px; color:#222; }}
h1 {{ font-size: 22px; }} h2 {{ font-size: 16px; color:#444; margin-top:24px; }}
table {{ border-collapse: collapse; }} td, th {{ border: 1px solid #ddd; padding: 4px 10px; font-size: 13px;}}
.small {{ color:#888; font-size: 12px; }}
</style></head><body>
<h1>Factor Evaluation Report</h1>
<p class="small">factor=<b>{cfg.factor_name}</b> · horizon=<b>{cfg.horizon}</b> ·
quantiles=<b>{cfg.n_quantiles}</b> · preprocess=<b>{', '.join(cfg.preprocess) or '-'}</b> ·
range=<b>{cfg.start_date or 'all'} ~ {cfg.end_date or 'all'}</b></p>
<h2>Summary</h2>
{tbl}
{is_oos_html}
{subp_html}
<h2>Charts</h2>
{body}
<p class="small">⚠ 本报告仅用于研究，不构成投资建议；请注意未来函数、幸存者偏差、停牌/ST/退市/复权与交易成本影响。</p>
</body></html>"""
    path.write_text(html, encoding="utf-8")


def _fmt(v) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "-"
    return f"{v:.4f}" if isinstance(v, float) else str(v)


def _is_oos_table_html(is_oos: dict) -> str:
    tr, te = is_oos["train"], is_oos["test"]
    cols = [("Rank IC", "ic_rank_mean"), ("ICIR", "icir_rank"),
            ("IC pos%", "ic_pos_ratio"), ("L-S Sharpe", "ls_sharpe")]
    head = "".join(f"<th>{c[0]}</th>" for c in cols)
    row_tr = "".join(f"<td>{_fmt(tr.get(c[1]))}</td>" for c in cols)
    row_te = "".join(f"<td>{_fmt(te.get(c[1]))}</td>" for c in cols)
    return (
        f"<h2>IS / OOS（purge={is_oos['purge_horizon']} embargo={is_oos['embargo']}）</h2>"
        f"<table><tr><th>segment</th><th>range</th>{head}</tr>"
        f"<tr><td>IS(train)</td><td>{tr.get('start')}~{tr.get('end')}</td>{row_tr}</tr>"
        f"<tr><td>OOS(test)</td><td>{te.get('start')}~{te.get('end')}</td>{row_te}</tr>"
        f"</table>"
    )


def _subperiod_table_html(subperiod: dict) -> str:
    cols = [("Rank IC", "ic_rank_mean"), ("ICIR", "icir_rank"),
            ("IC pos%", "ic_pos_ratio"), ("L-S Sharpe", "ls_sharpe")]
    head = "".join(f"<th>{c[0]}</th>" for c in cols)
    rows = ""
    for label, d in subperiod.items():
        cells = "".join(f"<td>{_fmt(d.get(c[1]))}</td>" for c in cols)
        rows += f"<tr><td>{label}</td><td>{d.get('start')}~{d.get('end')}</td>{cells}</tr>"
    return f"<h2>子区间稳定性</h2><table><tr><th>period</th><th>range</th>{head}</tr>{rows}</table>"


def _summary_table_html(summary: dict) -> str:
    rows = []
    ic_p = summary["ic"]["pearson"]
    ic_r = summary["ic"]["rank"]
    pf = summary["long_short_portfolio"]
    qs = summary["quantile"]
    tv = summary["turnover"]
    items = [
        ("IC mean (pearson)", ic_p.get("ic_mean")),
        ("IC ICIR (pearson)", ic_p.get("icir")),
        ("IC mean (rank)", ic_r.get("ic_mean")),
        ("IC ICIR (rank)", ic_r.get("icir")),
        ("IC pos ratio (rank)", ic_r.get("ic_pos_ratio")),
        ("L-S annual return", pf.get("annual_return")),
        ("L-S annual vol", pf.get("annual_volatility")),
        ("L-S sharpe", pf.get("sharpe")),
        ("L-S max drawdown", pf.get("max_drawdown")),
        ("L-S win rate", pf.get("win_rate")),
        ("Quantile L-S t-stat", qs.get("long_short_t_stat")),
        ("Turnover top (mean)", tv.get("top_mean")),
        ("Turnover L-S (mean)", tv.get("long_short_mean")),
    ]
    for k, v in items:
        vs = "-" if v is None or (isinstance(v, float) and pd.isna(v)) else (
            f"{v:.4f}" if isinstance(v, float) else str(v)
        )
        rows.append(f"<tr><td>{k}</td><td>{vs}</td></tr>")
    return "<table>" + "".join(rows) + "</table>"
