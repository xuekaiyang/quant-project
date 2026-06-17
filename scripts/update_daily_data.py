"""增量更新本地 daily_bar 数据（真实 provider）。

按交易日历逐日拉取全市场行情，落 raw 日分区（断点续传），再合并构建 daily_bar。

示例：
    # 小范围验证管线
    python scripts/update_daily_data.py --provider tushare --start 20240101 --end 20240131
    # 仅用已有 raw 重建 daily_bar（不重新拉取），并应用前复权
    python scripts/update_daily_data.py --rebuild --adjust qfq
    # 强制重拉（忽略已缓存日期）
    python scripts/update_daily_data.py --provider tushare --start 20240101 --end 20240131 --no-skip-existing
"""

from __future__ import annotations

import argparse

from qflab.data.storage import build_daily_bar_from_raw
from qflab.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--provider", choices=["akshare", "tushare"], default="tushare")
    p.add_argument("--start", help="起始日 YYYYMMDD / YYYY-MM-DD")
    p.add_argument("--end", help="结束日 YYYYMMDD / YYYY-MM-DD")
    p.add_argument(
        "--rebuild",
        action="store_true",
        help="仅用已有 raw 日分区重建 daily_bar，不拉取网络",
    )
    p.add_argument(
        "--no-skip-existing",
        dest="skip_existing",
        action="store_false",
        help="不跳过已缓存的交易日（强制重拉）",
    )
    p.add_argument(
        "--adjust",
        choices=["qfq"],
        default=None,
        help="构建 daily_bar 时的复权方式（默认不复权）",
    )
    args = p.parse_args()

    if args.rebuild:
        logger.info("Rebuilding daily_bar from raw (adjust=%s)", args.adjust)
        out = build_daily_bar_from_raw(adjust=args.adjust)
        logger.info("Rebuilt daily_bar -> %s", out)
        return

    if not args.start or not args.end:
        p.error("--start 和 --end 在非 --rebuild 模式下必填")

    if args.provider == "akshare":
        from qflab.data.akshare_provider import AkshareProvider
        prov = AkshareProvider()
    else:
        from qflab.data.tushare_provider import TushareProvider
        prov = TushareProvider()

    logger.info(
        "Updating daily data via %s: %s ~ %s (skip_existing=%s, adjust=%s)",
        args.provider, args.start, args.end, args.skip_existing, args.adjust,
    )
    prov.update_daily_data(
        args.start,
        args.end,
        skip_existing=args.skip_existing,
        build=True,
        adjust=args.adjust,
    )
    logger.info("Done.")


if __name__ == "__main__":
    main()
