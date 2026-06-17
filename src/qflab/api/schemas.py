"""FastAPI / pydantic 入参与出参 schema。"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class EvaluateRequest(BaseModel):
    factor: str
    horizon: int = 5
    start: str | None = None
    end: str | None = None
    quantiles: int = 5
    preprocess: list[str] = Field(default_factory=lambda: ["winsorize_quantile", "zscore"])


class EvaluateResponse(BaseModel):
    summary: dict[str, Any]


class FactorListResponse(BaseModel):
    factors: list[str]
