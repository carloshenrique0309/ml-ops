"""API + frontend (uma única imagem) para consultar resultados já calculados.

Não carrega modelo e não faz inferência: apenas lê os resultados de uma tabela
PostgreSQL (padrão) ou de um arquivo CSV/Parquet (modo local).
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool

from churn.results.config import ResultsSettings
from churn.results.repository import ResultsError, ResultsRepository, ResultsSchemaError
from churn.results.schemas import ResultsHealth, ResultsPage, ResultsSummary

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"

router = APIRouter()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def get_repository(request: Request) -> ResultsRepository:
    return request.app.state.repository


RepoDep = Annotated[ResultsRepository, Depends(get_repository)]


def _http_error(exc: ResultsError) -> HTTPException:
    status = 409 if isinstance(exc, ResultsSchemaError) else 503
    return HTTPException(status_code=status, detail=str(exc))


@asynccontextmanager
async def lifespan(app: FastAPI):
    repository: ResultsRepository = app.state.repository
    app.state.started_at = time.monotonic()
    try:
        await run_in_threadpool(repository.load)
        app.state.load_error = None
    except Exception as exc:  # noqa: BLE001 - sobe degradado e permite /api/reload
        app.state.load_error = str(exc)
        logger.warning("Não foi possível carregar os resultados: %s", exc)
    yield


@router.get("/health", response_model=ResultsHealth, tags=["health"])
def health(request: Request, repository: RepoDep) -> ResultsHealth:
    return ResultsHealth(
        status="ok" if repository.is_loaded else "degraded",
        source=repository.source_label,
        total_rows=repository.total_rows,
        loaded_at=repository.loaded_at,
        detail=getattr(request.app.state, "load_error", None),
    )


@router.get("/api/summary", response_model=ResultsSummary, tags=["resultados"])
def summary(repository: RepoDep) -> ResultsSummary:
    try:
        return ResultsSummary(**repository.summary())
    except ResultsError as exc:
        raise _http_error(exc) from exc


@router.get("/api/results", response_model=ResultsPage, tags=["resultados"])
def list_results(
    request: Request,
    repository: RepoDep,
    q: Annotated[str | None, Query(description="Busca textual em todas as colunas.")] = None,
    prediction: Annotated[
        Literal["churn", "no_churn"] | None, Query(description="Filtra pelo rótulo.")
    ] = None,
    min_probability: Annotated[
        float | None, Query(ge=0.0, le=1.0, description="Probabilidade mínima de churn.")
    ] = None,
    limit: Annotated[int | None, Query(ge=1, description="Tamanho da página.")] = None,
    offset: Annotated[int, Query(ge=0, description="Registros a pular.")] = 0,
) -> ResultsPage:
    settings: ResultsSettings = request.app.state.settings
    page_size = min(limit or settings.page_size, settings.max_page_size)

    try:
        total, items = repository.query(
            q=q,
            prediction=prediction,
            min_probability=min_probability,
            limit=page_size,
            offset=offset,
        )
    except ResultsError as exc:
        raise _http_error(exc) from exc

    return ResultsPage(total=total, limit=page_size, offset=offset, items=items)


@router.get("/api/results/{customer_id}", tags=["resultados"])
def get_result(customer_id: str, repository: RepoDep) -> dict[str, Any]:
    try:
        item = repository.get(customer_id)
    except ResultsError as exc:
        raise _http_error(exc) from exc
    if item is None:
        raise HTTPException(status_code=404, detail=f"Cliente {customer_id!r} não encontrado.")
    return item


@router.post("/api/reload", response_model=ResultsSummary, tags=["resultados"])
async def reload_results(repository: RepoDep) -> ResultsSummary:
    """Reler a fonte (tabela do banco ou arquivo) após o batch scoring."""
    try:
        await run_in_threadpool(repository.load)
        return ResultsSummary(**repository.summary())
    except ResultsError as exc:
        raise _http_error(exc) from exc


def create_app(
    settings: ResultsSettings | None = None,
    repository: ResultsRepository | None = None,
) -> FastAPI:
    """Factory do app (frontend + API na mesma aplicação)."""
    settings = settings or ResultsSettings()
    configure_logging(settings.log_level)

    app = FastAPI(
        title="churn-results",
        version="0.1.0",
        description=(
            "Consulta dos resultados de churn já calculados. Sem inferência: "
            "lê uma tabela PostgreSQL (ou CSV/Parquet) e expõe a consulta em "
            "JSON e em uma página web."
        ),
        lifespan=lifespan,
    )

    app.state.settings = settings
    app.state.repository = repository or ResultsRepository(
        settings.results_path,
        source=settings.results_source,
        database_url=settings.database_url,
        schema=settings.db_schema,
        table=settings.db_table,
        sslmode=settings.db_sslmode,
        id_column=settings.id_column,
        probability_column=settings.probability_column,
        prediction_column=settings.prediction_column,
        churn_threshold=settings.churn_threshold,
    )

    app.include_router(router)

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    return app


app = create_app()


def run() -> None:
    """Entrypoint do console script ``churn-results``."""
    import uvicorn

    settings = ResultsSettings()
    uvicorn.run(
        "churn.results.main:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
    )
