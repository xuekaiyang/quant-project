"""因子池：带元数据的因子注册库。

只存元数据(定义/出处/版本/最新回测指标/相关性/状态)，因子值与报告继续落 parquet。
入库走显式登记(register)，与评价解耦，用户控制什么入池。
"""

from .models import FactorRecord, FactorStatus
from .registry import record_from_evaluation, register
from .store import FactorPool

__all__ = [
    "FactorRecord",
    "FactorStatus",
    "FactorPool",
    "record_from_evaluation",
    "register",
]
