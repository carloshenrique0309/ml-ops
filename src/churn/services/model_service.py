"""Serviço responsável por descobrir, carregar e executar o modelo de churn."""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from churn.config import Settings, get_settings

logger = logging.getLogger(__name__)


class ModelNotAvailableError(RuntimeError):
    """Nenhum modelo pôde ser carregado com a configuração atual."""


class PredictionError(RuntimeError):
    """O modelo carregado não conseguiu executar a predição."""


@dataclass(slots=True)
class PredictionResult:
    """Saída padronizada de uma (ou várias) predições."""

    predictions: list[Any]
    probabilities: list[float] | None = None


@dataclass(slots=True)
class ModelSnapshot:
    """Modelo carregado em memória + metadados do MLflow."""

    model: Any
    model_uri: str
    loaded_at: datetime
    name: str | None = None
    version: str | None = None
    alias: str | None = None
    stage: str | None = None
    run_id: str | None = None
    status: str | None = None
    description: str | None = None
    strategy: str | None = None
    selection_metric: str | None = None
    selection_mode: str | None = None
    selection_value: float | None = None
    flavors: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    params: dict[str, str] = field(default_factory=dict)
    tags: dict[str, str] = field(default_factory=dict)
    input_columns: list[str] = field(default_factory=list)
    signature: dict[str, Any] | None = None


class ModelService:
    """Descobre o melhor modelo no MLflow, carrega em memória e serve predições.

    Ordem de resolução:

    1. ``MLFLOW_MODEL_URI`` (override explícito);
    2. alias do Model Registry (``MLFLOW_MODEL_ALIAS``);
    3. estágio do Model Registry (``MLFLOW_MODEL_STAGE``);
    4. versão mais recente registrada;
    5. melhor run do experimento pela métrica configurada.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._lock = threading.Lock()
        self._snapshot: ModelSnapshot | None = None

        mlflow.set_tracking_uri(self._settings.mlflow_tracking_uri)
        self._client = MlflowClient(tracking_uri=self._settings.mlflow_tracking_uri)

    # ---- Estado ---------------------------------------------------------
    @property
    def is_loaded(self) -> bool:
        return self._snapshot is not None

    @property
    def model_uri(self) -> str | None:
        return None if self._snapshot is None else self._snapshot.model_uri

    @property
    def snapshot(self) -> ModelSnapshot:
        if self._snapshot is None:
            raise ModelNotAvailableError(
                "Modelo ainda não carregado. Verifique o MLflow ou chame POST /model/reload."
            )
        return self._snapshot

    def info(self) -> ModelSnapshot:
        """Metadados do modelo em memória."""
        return self.snapshot

    # ---- Carga ----------------------------------------------------------
    def load(self) -> ModelSnapshot:
        """Resolve, baixa e carrega o modelo de forma thread-safe."""
        with self._lock:
            model_uri, meta = self._resolve_model_uri()
            logger.info(
                "Carregando modelo de %s (estratégia=%s)",
                model_uri,
                meta.get("strategy"),
            )

            flavors, signature, input_columns = self._inspect_model(model_uri)
            model = self._load_model(model_uri, flavors)

            snapshot = self._build_snapshot(
                model=model,
                model_uri=model_uri,
                meta=meta,
                flavors=flavors,
                signature=signature,
                input_columns=input_columns,
            )
            self._snapshot = snapshot

            logger.info(
                "Modelo carregado: name=%s version=%s run_id=%s",
                snapshot.name,
                snapshot.version,
                snapshot.run_id,
            )
            return snapshot

    def _resolve_model_uri(self) -> tuple[str, dict[str, Any]]:
        settings = self._settings

        if settings.mlflow_model_uri:
            logger.info("Usando MLFLOW_MODEL_URI explícito: %s", settings.mlflow_model_uri)
            return settings.mlflow_model_uri, {"strategy": "settings"}

        if settings.mlflow_model_alias:
            try:
                version = self._client.get_model_version_by_alias(
                    settings.mlflow_model_name, settings.mlflow_model_alias
                )
            except MlflowException as exc:
                logger.warning(
                    "Alias %r não encontrado em %r (%s). Tentando o próximo critério.",
                    settings.mlflow_model_alias,
                    settings.mlflow_model_name,
                    exc,
                )
            else:
                meta = self._version_meta(version, strategy="registry/alias")
                meta["alias"] = settings.mlflow_model_alias
                return (
                    f"models:/{settings.mlflow_model_name}@{settings.mlflow_model_alias}",
                    meta,
                )

        if settings.mlflow_model_stage:
            try:
                versions = self._client.get_latest_versions(
                    settings.mlflow_model_name, stages=[settings.mlflow_model_stage]
                )
            except MlflowException as exc:
                logger.warning("Estágio %r indisponível: %s", settings.mlflow_model_stage, exc)
                versions = []

            if versions:
                version = versions[0]
                return (
                    f"models:/{version.name}/{version.version}",
                    self._version_meta(version, strategy="registry/stage"),
                )

        try:
            registered = self._client.search_model_versions(
                f"name = '{settings.mlflow_model_name}'"
            )
        except MlflowException as exc:
            logger.info("Modelo %r não está registrado: %s", settings.mlflow_model_name, exc)
            registered = []

        if registered:
            version = max(registered, key=lambda item: _version_key(item.version))
            return (
                f"models:/{version.name}/{version.version}",
                self._version_meta(version, strategy="registry/latest"),
            )

        return self._resolve_best_run()

    def _resolve_best_run(self) -> tuple[str, dict[str, Any]]:
        settings = self._settings
        direction = "DESC" if settings.model_selection_mode == "max" else "ASC"
        metric_column = f"metrics.{settings.model_selection_metric}"

        try:
            runs = mlflow.search_runs(
                experiment_names=[settings.mlflow_experiment_name],
                order_by=[f"{metric_column} {direction}"],
                max_results=settings.model_selection_top_k,
            )
        except MlflowException as exc:
            raise ModelNotAvailableError(
                "Nenhum modelo encontrado. O registry não retornou versões e o experimento "
                f"{settings.mlflow_experiment_name!r} não pôde ser consultado: {exc}"
            ) from exc

        if runs.empty:
            raise ModelNotAvailableError(
                f"O experimento {settings.mlflow_experiment_name!r} não possui runs."
            )

        for _, row in runs.iterrows():
            run_id = str(row["run_id"])
            if not self._has_model_artifact(run_id):
                logger.warning(
                    "Run %s não possui artefato 'model'. Tentando o próximo candidato.", run_id
                )
                continue

            value = row.get(metric_column)
            logger.info(
                "Melhor run do experimento: %s (%s=%.4f)",
                run_id,
                settings.model_selection_metric,
                float(value) if pd.notna(value) else float("nan"),
            )
            return f"runs:/{run_id}/model", {
                "strategy": "experiment/best-run",
                "run_id": run_id,
                "selection_metric": settings.model_selection_metric,
                "selection_mode": settings.model_selection_mode,
                "selection_value": float(value) if pd.notna(value) else None,
                "metrics": _row_values(row, "metrics."),
                "params": _row_values(row, "params."),
            }

        raise ModelNotAvailableError(
            "Nenhum dos runs candidatos do experimento "
            f"{settings.mlflow_experiment_name!r} possui o artefato 'model'."
        )

    def _has_model_artifact(self, run_id: str) -> bool:
        try:
            return bool(self._client.list_artifacts(run_id, "model"))
        except MlflowException:
            logger.warning("Não foi possível listar artefatos do run %s", run_id)
            return False

    @staticmethod
    def _version_meta(version: Any, strategy: str) -> dict[str, Any]:
        stage = getattr(version, "current_stage", None)
        if stage in (None, "", "None"):
            stage = None
        return {
            "strategy": strategy,
            "name": version.name,
            "version": str(version.version),
            "stage": stage,
            "run_id": version.run_id,
            "status": version.status,
            "description": version.description,
            "tags": dict(getattr(version, "tags", None) or {}),
        }

    def _build_snapshot(
        self,
        model: Any,
        model_uri: str,
        meta: dict[str, Any],
        flavors: dict[str, Any],
        signature: dict[str, Any] | None,
        input_columns: list[str],
    ) -> ModelSnapshot:
        metrics: dict[str, float] = dict(meta.get("metrics") or {})
        params: dict[str, str] = dict(meta.get("params") or {})
        tags: dict[str, str] = dict(meta.get("tags") or {})

        run_id = meta.get("run_id")
        if run_id:
            run_metrics, run_params, run_tags = self._run_data(run_id)
            metrics = {**run_metrics, **metrics}
            params = {**run_params, **params}
            tags = {**run_tags, **tags}

        return ModelSnapshot(
            model=model,
            model_uri=model_uri,
            loaded_at=datetime.now(timezone.utc),
            name=meta.get("name"),
            version=meta.get("version"),
            alias=meta.get("alias"),
            stage=meta.get("stage"),
            run_id=run_id,
            status=meta.get("status"),
            description=meta.get("description"),
            strategy=meta.get("strategy"),
            selection_metric=meta.get("selection_metric"),
            selection_mode=meta.get("selection_mode"),
            selection_value=meta.get("selection_value"),
            flavors=flavors,
            metrics=metrics,
            params=params,
            tags=tags,
            input_columns=input_columns,
            signature=signature,
        )

    def _run_data(
        self, run_id: str
    ) -> tuple[dict[str, float], dict[str, str], dict[str, str]]:
        try:
            run = self._client.get_run(run_id)
        except MlflowException:
            logger.warning("Não foi possível ler os metadados do run %s", run_id, exc_info=True)
            return {}, {}, {}

        return (
            {key: float(value) for key, value in run.data.metrics.items()},
            {key: str(value) for key, value in run.data.params.items()},
            {key: str(value) for key, value in run.data.tags.items()},
        )

    def _inspect_model(
        self, model_uri: str
    ) -> tuple[dict[str, Any], dict[str, Any] | None, list[str]]:
        """Lê flavors, assinatura e colunas de entrada direto do artefato MLmodel."""
        try:
            from mlflow.models import get_model_info

            model_info = get_model_info(model_uri)
        except Exception:  # noqa: BLE001 - metadados são opcionais, não devem impedir o load
            logger.warning("Não foi possível inspecionar o modelo em %s", model_uri, exc_info=True)
            return {}, None, []

        flavors = dict(getattr(model_info, "flavors", None) or {})
        signature = getattr(model_info, "signature", None)

        signature_dict: dict[str, Any] | None = None
        if signature is not None:
            try:
                signature_dict = _normalize_json(signature.to_dict())
            except Exception:  # noqa: BLE001
                signature_dict = None

        return flavors, signature_dict, _schema_columns(signature)

    @staticmethod
    def _load_model(model_uri: str, flavors: dict[str, Any]) -> Any:
        # O flavor nativo (sklearn) preserva predict_proba, que o pyfunc não expõe.
        if "sklearn" in flavors:
            try:
                import mlflow.sklearn

                return mlflow.sklearn.load_model(model_uri)
            except Exception:  # noqa: BLE001
                logger.warning(
                    "Falha ao carregar o flavor sklearn; caindo para pyfunc.", exc_info=True
                )

        return mlflow.pyfunc.load_model(model_uri)

    # ---- Predição -------------------------------------------------------
    def predict(self, features: dict[str, Any]) -> PredictionResult:
        """Predição para um único registro."""
        return self.predict_batch([features])

    def predict_batch(self, records: list[dict[str, Any]]) -> PredictionResult:
        """Predição em lote (uma única chamada ao modelo)."""
        snapshot = self.snapshot
        if not records:
            return PredictionResult(predictions=[], probabilities=None)

        frame = _prepare_frame(records, snapshot.input_columns)
        try:
            raw_predictions = snapshot.model.predict(frame)
        except Exception as exc:  # noqa: BLE001
            raise PredictionError(
                "O modelo não conseguiu executar a predição. Confira os nomes/tipos dos campos "
                f"enviados. Detalhe: {exc}"
            ) from exc

        predictions = _to_python_list(raw_predictions)
        probabilities = _predict_proba(snapshot.model, frame)

        if len(predictions) != len(records):
            raise PredictionError(
                "Quantidade de predições diferente da quantidade de registros enviados "
                f"({len(predictions)} != {len(records)})."
            )

        return PredictionResult(predictions=predictions, probabilities=probabilities)


# ---- Helpers de módulo --------------------------------------------------
def _version_key(version: str) -> int:
    try:
        return int(version)
    except (TypeError, ValueError):
        return 0


def _normalize_json(value: Any) -> Any:
    """Decodifica strings JSON aninhadas (o MLflow 3 serializa `inputs`/`outputs` assim)."""
    if isinstance(value, dict):
        return {key: _normalize_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_json(item) for item in value]
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith(("[", "{")):
            try:
                return _normalize_json(json.loads(stripped))
            except json.JSONDecodeError:
                return value
    return value


def _schema_columns(signature: Any) -> list[str]:
    """Extrai os nomes das colunas de entrada de uma assinatura MLflow (best-effort)."""
    inputs = getattr(signature, "inputs", None)
    if inputs is None:
        return []

    for attribute in ("inputs", "columns"):
        values = getattr(inputs, attribute, None)
        if isinstance(values, dict):
            return [str(key) for key in values]
        if isinstance(values, list):
            names = [getattr(value, "name", None) for value in values]
            columns = [str(name) for name in names if name]
            if columns:
                return columns

    return []


def _prepare_frame(records: list[dict[str, Any]], columns: list[str]) -> pd.DataFrame:
    """Monta o DataFrame na ordem de colunas esperada pelo modelo (quando conhecida)."""
    frame = pd.DataFrame(records)
    if columns:
        known = [column for column in columns if column in frame.columns]
        if known:
            extra = [column for column in frame.columns if column not in known]
            frame = frame[known + extra]
    return frame


def _to_python_list(values: Any) -> list[Any]:
    if isinstance(values, pd.DataFrame):
        values = values.iloc[:, -1].to_numpy()
    elif isinstance(values, pd.Series):
        values = values.to_numpy()

    array = np.asarray(values)
    if array.ndim > 1:
        array = array.reshape(array.shape[0], -1)[:, -1]

    return [_python_scalar(value) for value in array.tolist()]


def _python_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _predict_proba(model: Any, frame: pd.DataFrame) -> list[float] | None:
    """Probabilidade da classe positiva, quando o modelo expõe ``predict_proba``."""
    predict_proba = getattr(model, "predict_proba", None)
    if not callable(predict_proba):
        return None

    try:
        probabilities = np.asarray(predict_proba(frame))
    except Exception:  # noqa: BLE001 - probabilidade é opcional
        logger.warning("Modelo não expôs predict_proba para este lote.", exc_info=True)
        return None

    if probabilities.ndim == 1:
        return [float(value) for value in probabilities]

    if probabilities.ndim == 2 and probabilities.shape[1] == 2:
        return [float(value) for value in probabilities[:, 1]]

    logger.info(
        "predict_proba com %s colunas: probabilidades só são retornadas para classificação binária.",
        probabilities.shape[-1],
    )
    return None


def _row_values(row: pd.Series, prefix: str) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for key, value in row.items():
        if not isinstance(key, str) or not key.startswith(prefix) or pd.isna(value):
            continue
        name = key[len(prefix) :]
        if prefix == "metrics.":
            try:
                values[name] = float(value)
            except (TypeError, ValueError):
                continue
        else:
            values[name] = str(value)
    return values
