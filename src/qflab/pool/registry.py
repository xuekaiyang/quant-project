"""因子登记逻辑：从评价结果 summary 拼装记录 + 落库时算与池中因子的相关性。"""

from __future__ import annotations

import json

from ..evaluation.correlation import factor_correlation
from ..utils.logger import get_logger
from .models import FactorRecord, FactorStatus
from .store import FactorPool

logger = get_logger(__name__)


def record_from_evaluation(
    summary: dict,
    source: str = "",
    category: str = "",
    description: str = "",
    definition: str = "",
    n_trials: int = 1,
    test_id: str = "",
) -> FactorRecord:
    """从 evaluate_factor 的 summary(dict) 抽取指标，拼装 FactorRecord。"""
    name = summary.get("factor_name", "")
    ic_r = summary.get("ic", {}).get("rank", {})
    pf = summary.get("long_short_portfolio", {})
    tv = summary.get("turnover", {})
    oos = summary.get("is_oos", {}).get("test", {}) if summary.get("is_oos") else {}
    filt = summary.get("filters", {})

    return FactorRecord(
        name=name,
        definition=definition or name,   # 默认用因子名作为定义标识(python-class)
        source=source,
        category=category,
        description=description,
        horizon=summary.get("horizon"),
        ic_rank_mean=ic_r.get("ic_mean"),
        icir_rank=ic_r.get("icir"),
        ic_pos_ratio=ic_r.get("ic_pos_ratio"),
        ls_sharpe=pf.get("sharpe"),
        ls_annual_return=pf.get("annual_return"),
        ls_max_drawdown=pf.get("max_drawdown"),
        turnover=tv.get("long_short_mean"),
        oos_ic_rank=oos.get("ic_rank_mean"),
        data_start=summary.get("start_date"),
        data_end=summary.get("end_date"),
        preprocess=",".join(summary.get("preprocess", [])),
        cost_bps=summary.get("trading_cost_bps"),
        eval_config_json=json.dumps(
            {k: summary.get(k) for k in ("horizon", "n_quantiles", "preprocess",
                                         "trading_cost_bps", "filters")},
            ensure_ascii=False, default=str,
        ),
        n_trials=n_trials,
        test_id=test_id,
        status=FactorStatus.CANDIDATE,
    )


def _compute_max_corr(
    pool: FactorPool,
    new_name: str,
    factor_values: dict[str, "object"],
    corr_threshold: float,
) -> tuple[str | None, float | None]:
    """算新因子与池中已有因子(除自己)的最大绝对相关，返回 (对象名, 相关系数)。

    factor_values 需包含 new_name 及池中已有因子的 long df。样本不足/无其他因子返回 (None, None)。
    """
    others = [n for n in factor_values if n != new_name]
    if not others:
        return None, None
    subset = {new_name: factor_values[new_name], **{n: factor_values[n] for n in others}}
    corr = factor_correlation(subset, method="spearman")
    row = corr.loc[new_name].drop(labels=[new_name])
    row = row.dropna()
    if row.empty:
        return None, None
    idx = row.abs().idxmax()
    return str(idx), float(row[idx])


def register(
    pool: FactorPool,
    summary: dict,
    factor_values: dict | None = None,
    source: str = "",
    category: str = "",
    description: str = "",
    definition: str = "",
    n_trials: int = 1,
    test_id: str = "",
    corr_threshold: float = 0.8,
) -> FactorRecord:
    """把一次评价结果登记入池。

    Parameters
    ----------
    pool : FactorPool 实例。
    summary : evaluate_factor 的 summary dict。
    factor_values : {因子名 -> long df}，含新因子及池中已有因子的值。
        提供时自动算相关性填 max_corr；超阈值则状态标 candidate 并注明冗余对象。
    """
    rec = record_from_evaluation(
        summary, source=source, category=category, description=description,
        definition=definition, n_trials=n_trials, test_id=test_id,
    )

    if factor_values and rec.name in factor_values:
        max_with, max_c = _compute_max_corr(pool, rec.name, factor_values, corr_threshold)
        rec.max_corr_with = max_with
        rec.max_corr = max_c
        if max_c is not None and abs(max_c) >= corr_threshold:
            rec.status = FactorStatus.CANDIDATE
            rec.status_reason = f"与 {max_with} 高相关({max_c:+.2f})，疑似冗余，待人工确认"
            logger.warning("因子 %s 与 %s 高相关(%.2f)，标记候选", rec.name, max_with, max_c)

    return pool.upsert(rec)
