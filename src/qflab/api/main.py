"""FastAPI 入口。`uvicorn qflab.api.main:app --reload` 启动。"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException

from ..evaluation.report import EvaluationConfig, evaluate_factor
from ..factors import list_factors
from .schemas import EvaluateRequest, EvaluateResponse, FactorListResponse

app = FastAPI(title="qflab API", version="0.1.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/factors", response_model=FactorListResponse)
def factors() -> FactorListResponse:
    return FactorListResponse(factors=list_factors())


@app.post("/evaluate", response_model=EvaluateResponse)
def evaluate(req: EvaluateRequest) -> EvaluateResponse:
    try:
        res = evaluate_factor(
            EvaluationConfig(
                factor_name=req.factor,
                horizon=req.horizon,
                start_date=req.start,
                end_date=req.end,
                n_quantiles=req.quantiles,
                preprocess=list(req.preprocess),
            )
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(e)) from e
    return EvaluateResponse(summary=res.summary)
