"""从本地 daily_bar 计算并保存因子。"""

from __future__ import annotations

import argparse

from qflab.data.storage import load_daily_bar, save_factor
from qflab.factors import get_factor, list_factors
from qflab.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--factor", type=str, required=True, help="factor name or 'all'")
    args = parser.parse_args()

    df = load_daily_bar()
    targets = list_factors() if args.factor == "all" else [args.factor]
    for name in targets:
        factor = get_factor(name)
        logger.info("Computing factor: %s", name)
        out = factor.compute(df)
        save_factor(out, name)
        logger.info("  rows=%d", len(out))


if __name__ == "__main__":
    main()
