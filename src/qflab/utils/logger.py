"""日志工具。"""

from __future__ import annotations

import logging
import os
import sys

_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_INITIALIZED = False


def setup_logging(level: str | int | None = None, fmt: str = _DEFAULT_FORMAT) -> None:
    """配置全局 root logger。多次调用幂等。"""
    global _INITIALIZED
    if _INITIALIZED:
        return
    if level is None:
        level = os.environ.get("QFLAB_LOG_LEVEL", "INFO")
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(fmt))
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
    _INITIALIZED = True


def get_logger(name: str) -> logging.Logger:
    """获取命名 logger。首次调用会确保 root logger 已配置。"""
    setup_logging()
    return logging.getLogger(name)
