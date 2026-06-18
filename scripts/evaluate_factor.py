"""评价单个因子并输出报告。"""

from __future__ import annotations

import argparse

from qflab.evaluation.report import EvaluationConfig, evaluate_factor, save_report
from qflab.utils.config import load_config
from qflab.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--factor", type=str, required=True)
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--start", type=str, default=None)
    p.add_argument("--end", type=str, default=None)
    p.add_argument("--quantiles", type=int, default=5)
    p.add_argument(
        "--preprocess",
        type=str,
        default="winsorize_quantile,zscore",
        help="comma list: winsorize_quantile|winsorize_mad|zscore|neutralize",
    )
    p.add_argument("--cost-bps", type=float, default=None, help="单边交易成本(基点)，默认读配置")
    p.add_argument("--no-filter", action="store_true", help="关闭可交易过滤(停牌/ST)")
    p.add_argument("--min-listed-days", type=int, default=0, help="上市天数过滤(需 list_date 字段)")
    args = p.parse_args()

    steps = [s for s in args.preprocess.split(",") if s.strip()] if args.preprocess else []

    eval_cfg_yaml = load_config().evaluation
    default_cost = float(eval_cfg_yaml.get("portfolio", {}).get("trading_cost_bps", 0.0))
    cost_bps = args.cost_bps if args.cost_bps is not None else default_cost

    cfg = EvaluationConfig(
        factor_name=args.factor,
        horizon=args.horizon,
        start_date=args.start,
        end_date=args.end,
        n_quantiles=args.quantiles,
        preprocess=steps,
        trading_cost_bps=cost_bps,
        exclude_suspended=not args.no_filter,
        exclude_st=not args.no_filter,
        min_listed_days=args.min_listed_days,
    )

    res = evaluate_factor(cfg)
    out = save_report(res)
    s = res.summary
    logger.info("===== Summary =====")
    logger.info(
        "IC mean (rank): %.4f  ICIR: %.4f  pos_ratio: %.4f",
        s["ic"]["rank"]["ic_mean"],
        s["ic"]["rank"]["icir"],
        s["ic"]["rank"]["ic_pos_ratio"],
    )
    pf = s["long_short_portfolio"]
    logger.info(
        "L-S annual_return=%.4f sharpe=%.4f max_dd=%.4f",
        pf["annual_return"], pf["sharpe"], pf["max_drawdown"],
    )
    logger.info("Report dir: %s", out)


if __name__ == "__main__":
    main()
