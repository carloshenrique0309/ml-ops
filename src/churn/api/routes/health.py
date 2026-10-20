"""Healthcheck da API."""

from __future__ import annotations

import time
from typing import Annotated

from fastapi import APIRouter, Depends, Request

from churn.api.deps import get_model_service
from churn.api.schemas import HealthResponse
from churn.services.model_service import ModelService

router = APIRouter(tags=["health"])

ModelServiceDep = Annotated[ModelService, Depends(get_model_service)]


@router.get("/health", response_model=HealthResponse, summary="Estado da API e do modelo")
def health(request: Request, service: ModelServiceDep) -> HealthResponse:
    settings = request.app.state.settings
    started_at = getattr(request.app.state, "started_at", time.monotonic())

    return HealthResponse(
        status="ok" if service.is_loaded else "degraded",
        model_loaded=service.is_loaded,
        model_uri=service.model_uri,
        version=settings.app_version,
        uptime_seconds=round(time.monotonic() - started_at, 3),
    )
