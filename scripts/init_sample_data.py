"""生成模拟 A 股日线样本数据。

- 100 只股票，500+ 个工作日
- 价格走 GBM；行业聚类带共同 beta；市值 = 自由流通价 * shares
- 不依赖外部数据源即可跑通整条 pipeline
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from qflab.data.normalizer import normalize_daily_bar
from qflab.data.storage import save_daily_bar
from qflab.utils.config import load_config
from qflab.utils.logger import get_logger

logger = get_logger(__name__)

INDUSTRIES = [
    "TMT", "Finance", "Consumer", "Energy", "Materials",
    "Industrial", "Healthcare", "RealEstate", "Utilities", "Telecom",
]


def generate_sample(
    n_stocks: int = 100,
    n_days: int = 600,
    start: str = "2018-01-02",
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, periods=n_days)
    instruments = [f"S{idx:04d}" for idx in range(n_stocks)]

    industries = rng.choice(INDUSTRIES, size=n_stocks)
    init_price = rng.uniform(5, 80, size=n_stocks)
    init_shares = rng.uniform(1e8, 5e9, size=n_stocks)

    market_ret = rng.normal(0.0003, 0.012, size=n_days)
    industry_ret = {ind: rng.normal(0.0001, 0.008, size=n_days) for ind in INDUSTRIES}

    rows = []
    last_close = init_price.copy()
    for t, dt in enumerate(dates):
        idio = rng.normal(0.0, 0.018, size=n_stocks)
        ind_eff = np.array([industry_ret[ind][t] for ind in industries])
        ret_t = 0.5 * market_ret[t] + 0.3 * ind_eff + idio
        new_close = last_close * (1.0 + ret_t)
        new_close = np.maximum(new_close, 0.5)
        opens = last_close * (1 + rng.normal(0, 0.003, size=n_stocks))
        highs = np.maximum(new_close, opens) * (1 + np.abs(rng.normal(0, 0.004, size=n_stocks)))
        lows = np.minimum(new_close, opens) * (1 - np.abs(rng.normal(0, 0.004, size=n_stocks)))
        volume = rng.lognormal(15, 0.6, size=n_stocks)
        amount = volume * new_close
        market_cap = init_shares * new_close

        for i, inst in enumerate(instruments):
            rows.append((
                dt, inst,
                float(opens[i]), float(highs[i]), float(lows[i]), float(new_close[i]),
                float(volume[i]), float(amount[i]),
                float(market_cap[i]), industries[i], False, False,
            ))
        last_close = new_close

    df = pd.DataFrame(rows, columns=[
        "trade_date", "instrument", "open", "high", "low", "close",
        "volume", "amount", "market_cap", "industry", "is_st", "is_suspended",
    ])
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate sample daily bar data")
    parser.add_argument("--stocks", type=int, default=100)
    parser.add_argument("--days", type=int, default=600)
    parser.add_argument("--start", type=str, default="2018-01-02")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config()
    cfg.paths.ensure()

    logger.info("Generating sample: stocks=%d days=%d start=%s", args.stocks, args.days, args.start)
    df = generate_sample(args.stocks, args.days, args.start, args.seed)
    df = normalize_daily_bar(df)
    out: Path = save_daily_bar(df)
    logger.info("Done. rows=%d  -> %s", len(df), out)


if __name__ == "__main__":
    main()
