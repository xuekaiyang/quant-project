"""tests for 因子相关性。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from qflab.evaluation.correlation import factor_correlation, redundant_pairs


def _long(values: pd.DataFrame, name: str) -> pd.DataFrame:
    """wide(date x inst) -> long factor df。"""
    long = values.stack().rename("factor_value").reset_index()
    long.columns = ["trade_date", "instrument", "factor_value"]
    long["factor_name"] = name
    return long


def _base(seed=0, n_days=20, n_stocks=50):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    insts = [f"S{i:03d}" for i in range(n_stocks)]
    return pd.DataFrame(rng.normal(size=(n_days, n_stocks)), index=dates, columns=insts)


def test_identical_factors_corr_near_one():
    a = _base()
    fv = {"a": _long(a, "a"), "b": _long(a.copy(), "b")}
    corr = factor_correlation(fv, method="spearman")
    assert abs(corr.loc["a", "b"] - 1.0) < 1e-6
    assert abs(corr.loc["a", "a"] - 1.0) < 1e-9  # 对角线


def test_negated_factor_corr_near_minus_one():
    a = _base()
    fv = {"a": _long(a, "a"), "neg": _long(-a, "neg")}
    corr = factor_correlation(fv, method="spearman")
    assert abs(corr.loc["a", "neg"] + 1.0) < 1e-6


def test_independent_factors_corr_near_zero():
    a = _base(seed=1)
    b = _base(seed=999)
    fv = {"a": _long(a, "a"), "b": _long(b, "b")}
    corr = factor_correlation(fv, method="spearman")
    assert abs(corr.loc["a", "b"]) < 0.15


def test_symmetric_matrix():
    fv = {"a": _long(_base(1), "a"), "b": _long(_base(2), "b"), "c": _long(_base(3), "c")}
    corr = factor_correlation(fv)
    assert np.allclose(corr.values, corr.values.T, equal_nan=True)


def test_redundant_pairs_flags_high_corr():
    a = _base()
    b = _base(seed=42)
    fv = {"a": _long(a, "a"), "a_dup": _long(a.copy(), "a_dup"), "b": _long(b, "b")}
    corr = factor_correlation(fv)
    pairs = redundant_pairs(corr, threshold=0.8)
    names = {frozenset([p[0], p[1]]) for p in pairs}
    assert frozenset(["a", "a_dup"]) in names
    assert frozenset(["a", "b"]) not in names


def test_needs_two_factors():
    fv = {"a": _long(_base(), "a")}
    try:
        factor_correlation(fv)
        assert False, "should require >=2 factors"
    except ValueError:
        pass
