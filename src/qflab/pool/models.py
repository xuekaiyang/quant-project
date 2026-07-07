"""因子池数据模型(pydantic)。"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class FactorStatus(str, Enum):
    """因子状态流转。"""

    CANDIDATE = "candidate"   # 候选：刚入池/存在冗余/待验证
    VALIDATED = "validated"   # 已验证：OOS 稳健、决定纳入研究
    RETIRED = "retired"       # 已退役：失效/被更优因子取代


class FactorRecord(BaseModel):
    """因子池中的一条记录。只存元数据，不存因子值。"""

    # ---- 身份 ----
    name: str
    version: int = 1
    created_at: str | None = None    # ISO 字符串，由 store 落库时补
    updated_at: str | None = None

    # ---- 定义 / 出处 ----
    definition: str = ""             # python-class 名 / 未来 DSL 串
    source: str = ""                 # 研报名/作者/URL/自研
    category: str = ""               # 动量/反转/波动/量能/市值...
    description: str = ""

    # ---- 最新回测指标(来自 summary.json，扁平存) ----
    horizon: int | None = None
    ic_rank_mean: float | None = None
    icir_rank: float | None = None
    ic_pos_ratio: float | None = None
    ls_sharpe: float | None = None
    ls_annual_return: float | None = None
    ls_max_drawdown: float | None = None
    turnover: float | None = None
    oos_ic_rank: float | None = None

    # ---- 回测上下文(可复现) ----
    data_start: str | None = None
    data_end: str | None = None
    n_stocks: int | None = None
    preprocess: str = ""             # 逗号串
    cost_bps: float | None = None
    eval_config_json: str = ""       # 完整 EvaluationConfig 的 json 快照

    # ---- 相关性 / 去重 ----
    max_corr_with: str | None = None
    max_corr: float | None = None

    # ---- 状态 ----
    status: FactorStatus = FactorStatus.CANDIDATE
    status_reason: str = ""

    # ---- 多重检验预留(本轮只记录，不做校正算法) ----
    n_trials: int = 1                # 该因子/该批次累计试验次数
    test_id: str = ""                # 试验批次标识

    def key(self) -> tuple[str, int]:
        return (self.name, self.version)
