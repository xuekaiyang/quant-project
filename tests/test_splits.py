"""tests for 时序切分原语：IS/OOS purge+embargo 与子区间划分。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from qflab.evaluation.report import EvaluationConfig, evaluate_factor
from qflab.evaluation.splits import subperiod_ranges, train_test_split_dates


# ------------------------------------------------------------ train/test split
def test_split_ratio_and_order():
    dates = pd.bdate_range("2024-01-01", periods=100)
    train, test = train_test_split_dates(dates, test_ratio=0.3, horizon=1, embargo=0)
    # 测试集约占 30%，且时间靠后
    assert 28 <= len(test) <= 32
    assert test.min() > train.max()


def test_purge_prevents_label_leakage():
    """核心防泄露断言：训练集最后建仓日 + horizon 必须 < 测试集起点。"""
    dates = pd.bdate_range("2024-01-01", periods=100)
    horizon = 5
    train, test = train_test_split_dates(dates, test_ratio=0.3, horizon=horizon, embargo=0)
    # 训练标签窗口 [T, T+horizon] 不得侵入测试集
    train_last_pos = list(dates).index(train.max())
    test_first_pos = list(dates).index(test.min())
    assert test_first_pos - train_last_pos > horizon


def test_embargo_widens_gap():
    dates = pd.bdate_range("2024-01-01", periods=100)
    horizon, embargo = 5, 3
    train, test = train_test_split_dates(dates, test_ratio=0.3, horizon=horizon, embargo=embargo)
    train_last_pos = list(dates).index(train.max())
    test_first_pos = list(dates).index(test.min())
    # gap 应 >= horizon + embargo
    assert test_first_pos - train_last_pos >= horizon + embargo


def test_split_short_sample_train_empty():
    """样本极短、purge 吃掉全部训练集时，train 为空但不报错。"""
    dates = pd.bdate_range("2024-01-01", periods=6)
    train, test = train_test_split_dates(dates, test_ratio=0.3, horizon=10, embargo=0)
    assert len(train) == 0
    assert len(test) >= 1


def test_split_invalid_ratio():
    dates = pd.bdate_range("2024-01-01", periods=10)
    for bad in (0.0, 1.0, -0.1, 1.5):
        try:
            train_test_split_dates(dates, test_ratio=bad, horizon=1)
            assert False, f"should reject ratio={bad}"
        except ValueError:
            pass


# ------------------------------------------------------------ sub-periods
def test_subperiod_by_year():
    dates = pd.bdate_range("2022-06-01", "2024-06-30")
    ranges = subperiod_ranges(dates, by="year")
    labels = [r[0] for r in ranges]
    assert labels == ["2022", "2023", "2024"]
    # 各段边界落在对应年份内
    for label, start, end in ranges:
        assert str(start.year) == label and str(end.year) == label


def test_subperiod_equal_k():
    dates = pd.bdate_range("2024-01-01", periods=30)
    ranges = subperiod_ranges(dates, by="3")
    assert len(ranges) == 3
    # 段间不重叠、按序
    assert ranges[0][2] < ranges[1][1]
    assert ranges[1][2] < ranges[2][1]


def test_subperiod_empty():
    assert subperiod_ranges([], by="year") == []


# ------------------------------------------------------------ end-to-end
def _make_factor_and_bar(n_days=180, n_stocks=60, seed=0):
    """构造合成数据：前半段因子与未来收益强正相关，后半段无关。
    子区间 IC 应能区分这两段。"""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-01", periods=n_days)
    insts = [f"S{i:03d}" for i in range(n_stocks)]
    bar_rows, fac_rows = [], []
    price = {s: 100.0 for s in insts}
    half = n_days // 2
    # 先按日生成因子值，再据此决定"未来收益"，保证前半段可预测
    fac_by_day = {}
    for di, d in enumerate(dates):
        fv = rng.normal(size=n_stocks)
        fac_by_day[d] = dict(zip(insts, fv))
    for di, d in enumerate(dates):
        for k, s in enumerate(insts):
            fac_rows.append((d, s, fac_by_day[d][s]))
    # 生成价格：前半段次日收益 = 0.02*当日因子 + 噪声；后半段纯噪声
    for di, d in enumerate(dates):
        for s in insts:
            bar_rows.append((d, s, price[s], price[s], price[s], price[s], 1e6, 1e8, False, False))
            if di < len(dates) - 1:
                if di < half:
                    ret = 0.02 * fac_by_day[d][s] + rng.normal(scale=0.005)
                else:
                    ret = rng.normal(scale=0.02)
                price[s] = price[s] * (1 + ret)
    bar = pd.DataFrame(bar_rows, columns=[
        "trade_date", "instrument", "open", "high", "low", "close",
        "volume", "amount", "is_st", "is_suspended"])
    fac = pd.DataFrame(fac_rows, columns=["trade_date", "instrument", "factor_value"])
    fac["factor_name"] = "synthetic"
    return fac, bar


def test_subperiod_ic_distinguishes_regimes():
    """端到端：前半段强 IC、后半段无效，等分两段的 IC 应明显不同。"""
    fac, bar = _make_factor_and_bar()
    cfg = EvaluationConfig(
        factor_name="synthetic", horizon=1, n_quantiles=5,
        preprocess=[], subperiods="2",
    )
    res = evaluate_factor(cfg, factor_df=fac, daily_bar=bar)
    assert res.subperiod is not None and len(res.subperiod) == 2
    ics = [d["ic_rank_mean"] for d in res.subperiod.values()]
    seg1, seg2 = ics[0], ics[1]
    # 第一段 IC 明显为正且远高于第二段
    assert seg1 > 0.1
    assert seg1 - seg2 > 0.1


def test_is_oos_populates_summary():
    fac, bar = _make_factor_and_bar()
    cfg = EvaluationConfig(
        factor_name="synthetic", horizon=1, n_quantiles=5,
        preprocess=[], oos_test_ratio=0.3, embargo=2,
    )
    res = evaluate_factor(cfg, factor_df=fac, daily_bar=bar)
    assert res.is_oos is not None
    assert "train" in res.is_oos and "test" in res.is_oos
    assert res.summary["is_oos"]["test"]["start"] is not None
