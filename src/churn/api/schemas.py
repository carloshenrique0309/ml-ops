"""Modelos de request/response da API (Pydantic)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from churn.services.model_service import ModelSnapshot


class ApiModel(BaseModel):
    """Base dos modelos da API (permite campos com prefixo ``model_``)."""

    model_config = ConfigDict(protected_namespaces=(), extra="forbid")


class ModelReference(ApiModel):
    name: str | None = Field(default=None, description="Nome no Model Registry.")
    version: str | None = Field(default=None, description="Versão registrada.")
    alias: str | None = Field(default=None, description="Alias usado no registry.")
    uri: str = Field(description="URI MLflow de onde o modelo foi carregado.")


class ModelInfoResponse(ApiModel):
    """Informações do melhor modelo: apenas métricas e hiperparâmetros."""

    metrics: dict[str, float] = Field(
        default_factory=dict, description="Métricas registradas no run do modelo."
    )
    params: dict[str, str] = Field(
        default_factory=dict, description="Parâmetros registrados no run do modelo."
    )

    @classmethod
    def from_snapshot(cls, snapshot: ModelSnapshot) -> ModelInfoResponse:
        return cls(metrics=snapshot.metrics, params=snapshot.params)


class HealthResponse(ApiModel):
    status: Literal["ok", "degraded"] = Field(
        description="'ok' com modelo carregado; 'degraded' sem modelo."
    )
    model_loaded: bool
    model_uri: str | None = None
    version: str = Field(description="Versão da API.")
    uptime_seconds: float = Field(description="Tempo desde o start da aplicação.")


class PredictRequest(ApiModel):
    features: dict[str, Any] = Field(
        ...,
        min_length=1,
        description="Campos do cliente, no formato usado no treino do modelo.",
        examples=[{"tenure": 12, "MonthlyCharges": 70.35, "Contract": "Month-to-month"}],
    )


class BatchPredictRequest(ApiModel):
    records: list[dict[str, Any]] = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Lista de registros (máximo 1000 por requisição).",
    )


class PredictResponse(ApiModel):
    predictions: list[Any] = Field(description="Predições do modelo (rótulos).")
    probabilities: list[float] | None = Field(
        default=None,
        description="Probabilidade da classe positiva (apenas classificação binária com predict_proba).",
    )
    model: ModelReference
