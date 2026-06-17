"""tests for raw 日分区存储 + build_daily_bar_from_raw（不打网络）。"""

from __future__ import annotations

import pandas as pd

from qflab.data.storage import (
    build_daily_bar_from_raw,
    list_raw_dates,
    save_raw_daily,
)


def _make_day(date_str: str, instruments, close_base: float = 10.0) -> pd.DataFrame:
    rows = []
    for k, inst in enumerate(instruments):
        c = close_base + k
        rows.append((pd.Timestamp(date_str), inst, c, c, c, c, 1000.0, 1000.0 * c))
    return pd.DataFrame(
        rows,
        columns=["trade_date", "instrument", "open", "high", "low", "close", "volume", "amount"],
    )


def test_list_raw_dates_empty(tmp_path):
    assert list_raw_dates(tmp_path / "nope") == []


def test_save_and_list_raw_dates(tmp_path):
    raw = tmp_path / "daily"
    save_raw_daily(_make_day("2024-01-03", ["A", "B"]), "20240103", dir_path=raw)
    save_raw_daily(_make_day("2024-01-02", ["A", "B"]), "20240102", dir_path=raw)
    # 乱序写入，list 应升序返回
    assert list_raw_dates(raw) == ["20240102", "20240103"]


def test_build_merges_dedups_and_sorts(tmp_path):
    raw = tmp_path / "daily"
    out_path = tmp_path / "daily_bar.parquet"
    save_raw_daily(_make_day("2024-01-02", ["A", "B"]), "20240102", dir_path=raw)
    save_raw_daily(_make_day("2024-01-03", ["A", "B"]), "20240103", dir_path=raw)
    # 同日重复写，build 后应只保留去重后的行
    save_raw_daily(_make_day("2024-01-03", ["A", "B"]), "20240103", dir_path=raw)

    build_daily_bar_from_raw(dir_path=raw, out_path=out_path)
    df = pd.read_parquet(out_path)

    # 2 天 × 2 股 = 4 行，无重复
    assert len(df) == 4
    assert df.duplicated(subset=["trade_date", "instrument"]).sum() == 0
    # 升序排序
    assert df["trade_date"].is_monotonic_increasing


def test_build_with_qfq(tmp_path):
    raw = tmp_path / "daily"
    out_path = tmp_path / "daily_bar_qfq.parquet"
    day = _make_day("2024-01-02", ["A"])
    day["adj_factor"] = 2.0
    save_raw_daily(day, "20240102", dir_path=raw)
    day2 = _make_day("2024-01-03", ["A"])
    day2["adj_factor"] = 4.0  # 最新日因子，归一基准
    save_raw_daily(day2, "20240103", dir_path=raw)

    build_daily_bar_from_raw(dir_path=raw, out_path=out_path, adjust="qfq")
    df = pd.read_parquet(out_path).sort_values("trade_date").reset_index(drop=True)

    # 最新日复权价 == 原始价；前一日按 2/4 缩放
    assert "close_raw" in df.columns
    assert df.loc[1, "close"] == df.loc[1, "close_raw"]  # latest day unchanged
    assert df.loc[0, "close"] == df.loc[0, "close_raw"] * 0.5
