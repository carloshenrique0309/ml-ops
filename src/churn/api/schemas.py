"""Modelos de request/response da API (Pydantic)."""

from __future__ import annotations

from datetime import datetime
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


class SelectionInfo(ApiModel):
    strategy: str = Field(description="Como o modelo foi escolhido (registry/alias, experiment/best-run, ...).")
    metric: str | None = Field(default=None, description="Métrica usada na seleção.")
    mode: Literal["max", "min"] | None = Field(default=None, description="Critério da métrica.")
    value: float | None = Field(default=None, description="Valor da métrica no modelo escolhido.")


class ModelInfoResponse(ApiModel):
    name: str | None = None
    version: str | None = None
    alias: str | None = None
    stage: str | None = None
    run_id: str | None = None
    status: str | None = None
    description: str | None = None
    model_uri: str = Field(description="URI completo do modelo.")
    loaded_at: datetime = Field(description="Quando o modelo foi carregado em memória (UTC).")
    flavors: list[str] = Field(default_factory=list, description="Flavors MLflow disponíveis.")
    metrics: dict[str, float] = Field(default_factory=dict)
    params: dict[str, str] = Field(default_factory=dict)
    tags: dict[str, str] = Field(default_factory=dict)
    signature: dict[str, Any] | None = Field(
        default=None, description="Assinatura de entrada/saída do modelo, quando disponível."
    )
    input_columns: list[str] = Field(
        default_factory=list, description="Colunas de entrada esperadas pelo modelo."
    )
    selection: SelectionInfo | None = None

    @classmethod
    def from_snapshot(cls, snapshot: ModelSnapshot) -> ModelInfoResponse:
        selection = None
        if snapshot.strategy:
            selection = SelectionInfo(
                strategy=snapshot.strategy,
                metric=snapshot.selection_metric,
                mode=snapshot.selection_mode,  # type: ignore[arg-type]
                value=snapshot.selection_value,
            )

        return cls(
            name=snapshot.name,
            version=snapshot.version,
            alias=snapshot.alias,
            stage=snapshot.stage,
            run_id=snapshot.run_id,
            status=snapshot.status,
            description=snapshot.description,
            model_uri=snapshot.model_uri,
            loaded_at=snapshot.loaded_at,
            flavors=sorted(snapshot.flavors),
            metrics=snapshot.metrics,
            params=snapshot.params,
            tags=snapshot.tags,
            signature=snapshot.signature,
            input_columns=snapshot.input_columns,
            selection=selection,
        )


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
