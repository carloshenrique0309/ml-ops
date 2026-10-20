"""Rotas de informação e recarga do modelo."""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from starlette.concurrency import run_in_threadpool

from churn.api.deps import get_model_service
from churn.api.schemas import ModelInfoResponse
from churn.services.model_service import ModelNotAvailableError, ModelService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/model", tags=["modelo"])

ModelServiceDep = Annotated[ModelService, Depends(get_model_service)]


@router.get(
    "/info",
    response_model=ModelInfoResponse,
    summary="Informações do melhor modelo em serving",
)
def model_info(service: ModelServiceDep) -> ModelInfoResponse:
    """Retorna metadados do modelo carregado (versão, run, métricas, params e assinatura)."""
    try:
        return ModelInfoResponse.from_snapshot(service.info())
    except ModelNotAvailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post(
    "/reload",
    response_model=ModelInfoResponse,
    summary="Recarrega o melhor modelo do MLflow",
)
async def reload_model(
    request: Request,
    service: ModelServiceDep,
    x_reload_token: Annotated[
        str | None,
        Header(alias="X-Reload-Token", description="Obrigatório quando RELOAD_TOKEN está definido."),
    ] = None,
) -> ModelInfoResponse:
    """Resolve novamente o melhor modelo e substitui o que está em memória.

    Útil depois de promover uma nova versão no Model Registry, sem reiniciar a API.
    """
    settings = request.app.state.settings
    if settings.reload_token and x_reload_token != settings.reload_token:
        raise HTTPException(status_code=401, detail="Token de recarga inválido.")

    try:
        snapshot = await run_in_threadpool(service.load)
    except ModelNotAvailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Falha ao recarregar o modelo")
        raise HTTPException(
            status_code=502,
            detail=f"Não foi possível carregar o modelo do MLflow: {exc}",
        ) from exc

    logger.info("Modelo recarregado: %s", snapshot.model_uri)
    return ModelInfoResponse.from_snapshot(snapshot)
