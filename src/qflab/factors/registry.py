"""因子注册表。"""

from __future__ import annotations

from .base import Factor
from .market_cap_factors import LogMarketCap
from .price_factors import (
    CloseToMA20,
    CloseToMA60,
    Ret5D,
    Ret20D,
    Ret60D,
    Reverse1D,
    Reverse5D,
)
from .volatility_factors import Volatility20D, Volatility60D
from .volume_factors import AmountMean20D, VolumeMean20D

_REGISTRY: dict[str, type[Factor]] = {
    cls.name: cls
    for cls in [
        Ret5D,
        Ret20D,
        Ret60D,
        Reverse1D,
        Reverse5D,
        Volatility20D,
        Volatility60D,
        VolumeMean20D,
        AmountMean20D,
        CloseToMA20,
        CloseToMA60,
        LogMarketCap,
    ]
}


def list_factors() -> list[str]:
    """列出所有已注册因子名。"""
    return sorted(_REGISTRY.keys())


def get_factor(name: str) -> Factor:
    """按名称获取因子实例。"""
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown factor: {name}. Available: {list_factors()}"
        )
    return _REGISTRY[name]()


def register(cls: type[Factor]) -> type[Factor]:
    """装饰器：注册自定义因子。"""
    if not cls.name:
        raise ValueError(f"Factor class {cls.__name__} must define a non-empty 'name'.")
    _REGISTRY[cls.name] = cls
    return cls
