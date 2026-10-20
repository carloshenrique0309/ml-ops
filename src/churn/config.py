"""Configuração da API, carregada de variáveis de ambiente e/ou arquivo .env."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Todas as configurações da API em um único lugar.

    As variáveis de ambiente usam o nome do campo em MAIÚSCULAS
    (ex.: ``mlflow_tracking_uri`` -> ``MLFLOW_TRACKING_URI``).
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ---- Servidor -------------------------------------------------------
    app_name: str = "churn-api"
    app_version: str = "0.1.0"
    log_level: str = "INFO"
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False
    autoload_model: bool = True
    fail_fast_on_startup: bool = False
    reload_token: str | None = None

    # ---- MLflow ---------------------------------------------------------
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "churn"
    mlflow_model_name: str = "churn"
    mlflow_model_alias: str | None = "champion"
    mlflow_model_stage: str | None = None
    mlflow_model_uri: str | None = None

    # ---- Seleção automática do melhor modelo (fallback) -----------------
    model_selection_metric: str = Field(
        default="roc_auc",
        description="Métrica usada para escolher o melhor run do experimento.",
    )
    model_selection_mode: Literal["max", "min"] = Field(
        default="max",
        description="'max' para métricas em que maior é melhor; 'min' para perdas/erros.",
    )
    model_selection_top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Quantos runs candidatos avaliar até encontrar um com artefato de modelo.",
    )


@lru_cache
def get_settings() -> Settings:
    """Retorna as configurações (cacheadas por processo)."""
    return Settings()
