"""Aplicação FastAPI: cria o app, carrega o modelo no startup e expõe as rotas."""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from starlette.concurrency import run_in_threadpool

from churn.api.routes import health, model, predict
from churn.config import Settings, get_settings
from churn.services.model_service import ModelService

logger = logging.getLogger(__name__)


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings: Settings = app.state.settings
    service: ModelService = app.state.model_service

    app.state.started_at = time.monotonic()

    if settings.autoload_model:
        try:
            await run_in_threadpool(service.load)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Não foi possível carregar o modelo no startup: %s", exc)
            if settings.fail_fast_on_startup:
                raise
    else:
        logger.info("AUTOLOAD_MODEL=false: modelo não será carregado no startup.")

    yield

    logger.info("API finalizada.")


def create_app(
    settings: Settings | None = None,
    service: ModelService | None = None,
) -> FastAPI:
    """Factory do app (facilita testes com settings/serviço injetados)."""
    settings = settings or get_settings()
    configure_logging(settings.log_level)

    app = FastAPI(
        title="churn-api",
        version=settings.app_version,
        description=(
            "API de serving do modelo de churn. Descobre o melhor modelo no MLflow "
            "(alias/estágio/registry ou melhor run do experimento) e expõe predições."
        ),
        lifespan=lifespan,
    )

    app.state.settings = settings
    app.state.started_at = time.monotonic()
    app.state.model_service = service or ModelService(settings)

    app.include_router(health.router)
    app.include_router(model.router)
    app.include_router(predict.router)

    @app.get("/", include_in_schema=False)
    def root() -> dict[str, str]:
        return {
            "name": settings.app_name,
            "version": settings.app_version,
            "docs": "/docs",
            "health": "/health",
            "model_info": "/model/info",
        }

    return app


app = create_app()


def run() -> None:
    """Entrypoint do console script ``churn-api``."""
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "churn.api.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        log_level=settings.log_level.lower(),
    )
