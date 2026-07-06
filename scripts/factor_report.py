"""批量评价多因子 → leaderboard + 因子相关性矩阵。

对每个因子跑单因子评价，汇总成横向对比榜单，并算两两相关识别冗余因子。

示例：
    python scripts/factor_report.py --factors all --horizon 5
    python scripts/factor_report.py --factors ret_20d,reverse_5d,volatility_20d --horizon 5 --cost-bps 10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from qflab.data.storage import load_daily_bar, load_factor
from qflab.evaluation.correlation import factor_correlation, redundant_pairs
from qflab.evaluation.report import EvaluationConfig, evaluate_factor
from qflab.factors import list_factors
from qflab.utils.config import load_config
from qflab.utils.logger import get_logger
from qflab.visualization.plots import correlation_heatmap

logger = get_logger(__name__)


def _row_from_result(name: str, res) -> dict:
    s = res.summary
    r = s["ic"]["rank"]
    pf = s["long_short_portfolio"]
    row = {
        "factor": name,
        "ic_rank_mean": r.get("ic_mean"),
        "icir_rank": r.get("icir"),
        "ic_pos_ratio": r.get("ic_pos_ratio"),
        "ls_sharpe": pf.get("sharpe"),
        "ls_annual_return": pf.get("annual_return"),
        "ls_max_drawdown": pf.get("max_drawdown"),
        "turnover_ls": s["turnover"].get("long_short_mean"),
    }
    if res.is_oos:
        row["oos_ic_rank"] = res.is_oos["test"].get("ic_rank_mean")
    return row


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--factors", type=str, default="all", help="all 或逗号分隔因子名")
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--quantiles", type=int, default=5)
    p.add_argument("--preprocess", type=str, default="winsorize_quantile,zscore")
    p.add_argument("--cost-bps", type=float, default=None)
    p.add_argument("--oos-ratio", type=float, default=0.0)
    p.add_argument("--corr-threshold", type=float, default=0.8, help="冗余因子相关性阈值")
    p.add_argument("--sort-by", type=str, default="icir_rank", help="leaderboard 排序列(取绝对值)")
    args = p.parse_args()

    steps = [s for s in args.preprocess.split(",") if s.strip()]
    eval_yaml = load_config().evaluation
    default_cost = float(eval_yaml.get("portfolio", {}).get("trading_cost_bps", 0.0))
    cost_bps = args.cost_bps if args.cost_bps is not None else default_cost

    if args.factors == "all":
        names = list_factors()
    else:
        names = [x.strip() for x in args.factors.split(",") if x.strip()]

    daily_bar = load_daily_bar()
    rows = []
    factor_values: dict[str, pd.DataFrame] = {}
    for name in names:
        try:
            fdf = load_factor(name)
        except FileNotFoundError:
            logger.warning("因子 %s 未计算，跳过(先跑 compute_factors.py)", name)
            continue
        factor_values[name] = fdf
        cfg = EvaluationConfig(
            factor_name=name, horizon=args.horizon, n_quantiles=args.quantiles,
            preprocess=steps, trading_cost_bps=cost_bps, oos_test_ratio=args.oos_ratio,
        )
        logger.info("评价因子: %s", name)
        res = evaluate_factor(cfg, factor_df=fdf, daily_bar=daily_bar)
        rows.append(_row_from_result(name, res))

    if not rows:
        logger.error("没有可评价的因子。")
        return

    lb = pd.DataFrame(rows)
    sort_col = args.sort_by if args.sort_by in lb.columns else "icir_rank"
    lb = lb.reindex(lb[sort_col].abs().sort_values(ascending=False).index).reset_index(drop=True)

    out_dir = load_config().paths.evaluation / "_leaderboard"
    out_dir.mkdir(parents=True, exist_ok=True)
    lb.to_csv(out_dir / "leaderboard.csv", index=False)
    (out_dir / "leaderboard.json").write_text(
        json.dumps(lb.to_dict(orient="records"), indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    logger.info("===== Leaderboard (sorted by |%s|) =====", sort_col)
    logger.info("\n%s", lb.to_string(index=False))

    # 因子相关性
    if len(factor_values) >= 2:
        corr = factor_correlation(factor_values, method="spearman")
        corr.to_parquet(out_dir / "correlation.parquet")
        fig = correlation_heatmap(corr, "Factor Correlation (mean daily rank corr)")
        (out_dir / "correlation.html").write_text(
            fig.to_html(full_html=True, include_plotlyjs="cdn"), encoding="utf-8"
        )
        pairs = redundant_pairs(corr, threshold=args.corr_threshold)
        if pairs:
            logger.info("===== 高相关因子对 (|corr|>=%.2f) =====", args.corr_threshold)
            for a, b, c in pairs:
                logger.info("  %s <-> %s : %.3f", a, b, c)
        else:
            logger.info("无 |corr|>=%.2f 的冗余因子对。", args.corr_threshold)

    logger.info("产物目录: %s", out_dir)


if __name__ == "__main__":
    main()
