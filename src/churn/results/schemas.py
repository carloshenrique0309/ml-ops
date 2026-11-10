"""Contratos da API de resultados (somente leitura)."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class ResultsSummary(BaseModel):
    total: int = Field(description="Total de registros consultados.")
    churn: int | None = Field(default=None, description="Registros classificados como churn.")
    not_churn: int | None = Field(default=None, description="Registros classificados como não churn.")
    churn_rate: float | None = Field(default=None, description="Proporção de churn (0 a 1).")
    average_probability: float | None = Field(
        default=None, description="Probabilidade média de churn (0 a 1)."
    )
    churn_threshold: float = Field(description="Limiar usado quando só há probabilidade.")
    id_column: str | None = None
    probability_column: str | None = None
    prediction_column: str | None = None
    columns: list[str] = Field(default_factory=list)
    source: str = Field(description="Arquivo de origem dos resultados.")
    updated_at: datetime | None = Field(
        default=None, description="Última modificação do arquivo (UTC)."
    )


class ResultsPage(BaseModel):
    total: int
    limit: int
    offset: int
    items: list[dict[str, Any]]


class ResultsHealth(BaseModel):
    status: Literal["ok", "degraded"]
    source: str
    total_rows: int
    loaded_at: datetime | None = None
    detail: str | None = Field(
        default=None, description="Motivo quando o arquivo não pôde ser carregado."
    )
