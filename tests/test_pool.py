"""tests for 因子池：CRUD 幂等/版本/抽取/相关性/状态。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from qflab.pool.models import FactorRecord, FactorStatus
from qflab.pool.registry import record_from_evaluation, register
from qflab.pool.store import FactorPool


def _pool(tmp_path):
    return FactorPool(db_path=tmp_path / "pool" / "factors.db")


def _summary(name="ret_20d", ic=-0.05, sharpe=-1.1):
    return {
        "factor_name": name,
        "horizon": 5,
        "start_date": None,
        "end_date": None,
        "n_quantiles": 5,
        "preprocess": ["winsorize_quantile", "zscore"],
        "trading_cost_bps": 0.0,
        "filters": {"exclude_suspended": True, "exclude_st": True, "min_listed_days": 0},
        "ic": {"rank": {"ic_mean": ic, "icir": ic * 7, "ic_pos_ratio": 0.36}},
        "long_short_portfolio": {"sharpe": sharpe, "annual_return": -0.2, "max_drawdown": -0.6},
        "turnover": {"long_short_mean": 0.4},
    }


# ------------------------------------------------------------------ CRUD
def test_upsert_and_get(tmp_path):
    pool = _pool(tmp_path)
    rec = FactorRecord(name="f1", definition="f1", category="动量", ic_rank_mean=0.03)
    pool.upsert(rec)
    got = pool.get("f1")
    assert got is not None
    assert got.version == 1
    assert got.category == "动量"
    assert got.created_at is not None
    pool.close()


def test_upsert_idempotent_same_definition(tmp_path):
    """同名同定义再登记 → 更新不新增版本。"""
    pool = _pool(tmp_path)
    pool.upsert(FactorRecord(name="f1", definition="f1", ic_rank_mean=0.01))
    pool.upsert(FactorRecord(name="f1", definition="f1", ic_rank_mean=0.09))
    assert pool.latest_version("f1") == 1
    assert abs(pool.get("f1").ic_rank_mean - 0.09) < 1e-9  # 指标被更新
    pool.close()


def test_definition_change_bumps_version(tmp_path):
    pool = _pool(tmp_path)
    pool.upsert(FactorRecord(name="f1", definition="v1_formula"))
    pool.upsert(FactorRecord(name="f1", definition="v2_formula"))
    assert pool.latest_version("f1") == 2
    # 旧版本仍在
    assert pool.get("f1", version=1).definition == "v1_formula"
    assert pool.get("f1", version=2).definition == "v2_formula"
    pool.close()


def test_list_filters(tmp_path):
    pool = _pool(tmp_path)
    pool.upsert(FactorRecord(name="a", definition="a", category="动量"))
    pool.upsert(FactorRecord(name="b", definition="b", category="反转"))
    assert len(pool.list_all()) == 2
    assert len(pool.list_all(category="动量")) == 1
    assert pool.list_all(category="动量")[0].name == "a"
    pool.close()


def test_set_status(tmp_path):
    pool = _pool(tmp_path)
    pool.upsert(FactorRecord(name="a", definition="a"))
    rec = pool.set_status("a", "validated", reason="OOS 稳健")
    assert rec.status == FactorStatus.VALIDATED
    assert pool.get("a").status_reason == "OOS 稳健"
    pool.close()


def test_delete(tmp_path):
    pool = _pool(tmp_path)
    pool.upsert(FactorRecord(name="a", definition="a"))
    assert pool.delete("a") == 1
    assert pool.get("a") is None
    pool.close()


def test_to_dataframe_roundtrip(tmp_path):
    pool = _pool(tmp_path)
    pool.upsert(FactorRecord(name="a", definition="a", ic_rank_mean=0.03))
    pool.upsert(FactorRecord(name="b", definition="b", ic_rank_mean=-0.02))
    df = pool.to_dataframe()
    assert set(df["name"]) == {"a", "b"}
    assert "ic_rank_mean" in df.columns
    pool.close()


# ------------------------------------------------------- record_from_evaluation
def test_record_from_evaluation_extracts_fields():
    rec = record_from_evaluation(_summary("ret_20d", ic=-0.05), source="自研", category="动量")
    assert rec.name == "ret_20d"
    assert rec.category == "动量"
    assert abs(rec.ic_rank_mean + 0.05) < 1e-9
    assert rec.horizon == 5
    assert rec.preprocess == "winsorize_quantile,zscore"
    assert rec.status == FactorStatus.CANDIDATE


def test_record_from_evaluation_reads_oos():
    s = _summary()
    s["is_oos"] = {"test": {"ic_rank_mean": -0.051}}
    rec = record_from_evaluation(s)
    assert abs(rec.oos_ic_rank + 0.051) < 1e-9


# ------------------------------------------------------------- correlation on register
def _long_factor(name, wide):
    long = wide.stack().rename("factor_value").reset_index()
    long.columns = ["trade_date", "instrument", "factor_value"]
    long["factor_name"] = name
    return long


def test_register_computes_max_corr(tmp_path):
    pool = _pool(tmp_path)
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2024-01-01", periods=20)
    insts = [f"S{i:02d}" for i in range(40)]
    base = pd.DataFrame(rng.normal(size=(20, 40)), index=dates, columns=insts)

    # 先登记 a
    fv = {"a": _long_factor("a", base)}
    register(pool, _summary("a"), factor_values=fv, source="自研")

    # 再登记 a 的复制品 a_dup → 应检测到高相关
    fv2 = {"a": _long_factor("a", base), "a_dup": _long_factor("a_dup", base.copy())}
    rec = register(pool, _summary("a_dup"), factor_values=fv2, source="自研", corr_threshold=0.8)
    assert rec.max_corr_with == "a"
    assert rec.max_corr is not None and rec.max_corr > 0.9
    assert "高相关" in rec.status_reason
    pool.close()


def test_register_no_corr_when_alone(tmp_path):
    pool = _pool(tmp_path)
    rng = np.random.default_rng(1)
    dates = pd.bdate_range("2024-01-01", periods=15)
    insts = [f"S{i:02d}" for i in range(30)]
    base = pd.DataFrame(rng.normal(size=(15, 30)), index=dates, columns=insts)
    fv = {"solo": _long_factor("solo", base)}
    rec = register(pool, _summary("solo"), factor_values=fv)
    assert rec.max_corr_with is None
    pool.close()
