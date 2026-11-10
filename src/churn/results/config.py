"""Configuração do viewer de resultados."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ResultsSettings(BaseSettings):
    """Configuração lida de variáveis de ambiente e/ou ``.env``."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "churn-results"
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"

    # Fonte dos resultados: PostgreSQL (produção) ou CSV/Parquet (local/testes).
    results_source: Literal["postgres", "csv"] = "postgres"

    # Usado apenas quando RESULTS_SOURCE=csv.
    results_path: Path = Path("data/sample_predictions.csv")

    # Conexão PostgreSQL (Railway injeta DATABASE_URL ao referenciar o serviço).
    # Ex.: postgresql://usuario:senha@host:5432/banco
    database_url: str | None = None
    db_schema: str = "public"
    db_table: str = "churn_results"
    # Opcional: "require", "prefer", "disable"... Anexado à URL quando definido.
    db_sslmode: str | None = None

    # Nomes das colunas podem ser sobrescritos; por padrão são detectados.
    id_column: str | None = None
    probability_column: str | None = None
    prediction_column: str | None = None

    churn_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    page_size: int = Field(default=25, ge=1, le=500)
    max_page_size: int = Field(default=500, ge=1, le=5000)
