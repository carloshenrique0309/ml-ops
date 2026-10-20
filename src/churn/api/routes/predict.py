"""Rotas de predição."""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from churn.api.deps import get_model_service
from churn.api.schemas import (
    BatchPredictRequest,
    ModelReference,
    PredictRequest,
    PredictResponse,
)
from churn.services.model_service import (
    ModelNotAvailableError,
    ModelService,
    ModelSnapshot,
    PredictionError,
    PredictionResult,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["predição"])

ModelServiceDep = Annotated[ModelService, Depends(get_model_service)]


@router.post("/predict", response_model=PredictResponse, summary="Predição de um cliente")
def predict(payload: PredictRequest, service: ModelServiceDep) -> PredictResponse:
    return _predict(service, [payload.features])


@router.post(
    "/predict/batch",
    response_model=PredictResponse,
    summary="Predição em lote (até 1000 clientes)",
)
def predict_batch(payload: BatchPredictRequest, service: ModelServiceDep) -> PredictResponse:
    return _predict(service, payload.records)


def _predict(service: ModelService, records: list[dict]) -> PredictResponse:
    try:
        result: PredictionResult = service.predict_batch(records)
        snapshot: ModelSnapshot = service.info()
    except ModelNotAvailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except PredictionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Erro inesperado durante a predição")
        raise HTTPException(
            status_code=500, detail="Erro interno ao executar a predição."
        ) from exc

    return PredictResponse(
        predictions=result.predictions,
        probabilities=result.probabilities,
        model=ModelReference(
            name=snapshot.name,
            version=snapshot.version,
            alias=snapshot.alias,
            uri=snapshot.model_uri,
        ),
    )
