"""评价单个因子并输出报告。"""

from __future__ import annotations

import argparse

from qflab.evaluation.report import EvaluationConfig, evaluate_factor, save_report
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
    args = p.parse_args()

    steps = [s for s in args.preprocess.split(",") if s.strip()] if args.preprocess else []
    cfg = EvaluationConfig(
        factor_name=args.factor,
        horizon=args.horizon,
        start_date=args.start,
        end_date=args.end,
        n_quantiles=args.quantiles,
        preprocess=steps,
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
