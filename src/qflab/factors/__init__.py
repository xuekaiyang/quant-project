"""qflab.factors 入口。"""

from .base import Factor
from .registry import get_factor, list_factors, register

__all__ = ["Factor", "get_factor", "list_factors", "register"]
