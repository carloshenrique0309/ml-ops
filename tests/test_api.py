"""Testes da API usando um serviço de modelo falso (não dependem do MLflow)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from churn.api.main import create_app
from churn.config import Settings
from churn.services.model_service import (
    ModelNotAvailableError,
    ModelService,
    ModelSnapshot,
    PredictionResult,
)
from datetime import datetime, timezone


class FakeModel:
    def predict(self, frame: pd.DataFrame) -> list[int]:
        return [1] * len(frame)

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        return np.array([[0.25, 0.75]] * len(frame))


def make_snapshot() -> ModelSnapshot:
    return ModelSnapshot(
        model=FakeModel(),
        model_uri="models:/churn@champion",
        loaded_at=datetime.now(timezone.utc),
        name="churn",
        version="3",
        alias="champion",
        run_id="abc123",
        status="READY",
        strategy="registry/alias",
        flavors={"python_function": {}, "sklearn": {}},
        metrics={"roc_auc": 0.9123},
        params={"n_estimators": "300"},
        tags={"task": "churn"},
        input_columns=["tenure", "MonthlyCharges"],
    )


class FakeService(ModelService):
    """Duck-type do ModelService para testar as rotas sem rede/MLflow."""

    def __init__(self, loaded: bool = True) -> None:
        self._loaded = loaded
        self.load_calls = 0
        self._snapshot = make_snapshot() if loaded else None

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def model_uri(self) -> str | None:
        return self._snapshot.model_uri if self._snapshot else None

    def load(self) -> ModelSnapshot:
        self.load_calls += 1
        self._loaded = True
        self._snapshot = make_snapshot()
        return self._snapshot

    def info(self) -> ModelSnapshot:
        if self._snapshot is None:
            raise ModelNotAvailableError("Modelo ainda não carregado.")
        return self._snapshot


def make_client(service: FakeService | None = None, **overrides) -> TestClient:
    settings = Settings(
        autoload_model=False,
        mlflow_tracking_uri="sqlite://",
        **overrides,
    )
    return TestClient(create_app(settings=settings, service=service or FakeService()))


def test_health_ok_with_loaded_model() -> None:
    with make_client() as client:
        response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_uri"] == "models:/churn@champion"


def test_health_degraded_without_model() -> None:
    with make_client(FakeService(loaded=False)) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "degraded"


def test_model_info_exposes_best_model_metadata() -> None:
    with make_client() as client:
        response = client.get("/model/info")

    assert response.status_code == 200
    body = response.json()
    assert body["name"] == "churn"
    assert body["version"] == "3"
    assert body["alias"] == "champion"
    assert body["metrics"]["roc_auc"] == pytest.approx(0.9123)
    assert body["flavors"] == ["python_function", "sklearn"]
    assert body["selection"]["strategy"] == "registry/alias"


def test_model_info_returns_503_without_model() -> None:
    with make_client(FakeService(loaded=False)) as client:
        response = client.get("/model/info")

    assert response.status_code == 503


def test_predict_single_returns_probability() -> None:
    with make_client() as client:
        response = client.post(
            "/predict",
            json={"features": {"tenure": 12, "MonthlyCharges": 70.35}},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["predictions"] == [1]
    assert body["probabilities"] == [0.75]
    assert body["model"]["uri"] == "models:/churn@champion"


def test_predict_batch() -> None:
    with make_client() as client:
        response = client.post(
            "/predict/batch",
            json={"records": [{"tenure": 1}, {"tenure": 24}]},
        )

    assert response.status_code == 200
    assert response.json()["predictions"] == [1, 1]


def test_predict_requires_features() -> None:
    with make_client() as client:
        response = client.post("/predict", json={"features": {}})

    assert response.status_code == 422


def test_predict_returns_503_without_model() -> None:
    with make_client(FakeService(loaded=False)) as client:
        response = client.post("/predict", json={"features": {"tenure": 12}})

    assert response.status_code == 503


def test_reload_requires_token_when_configured() -> None:
    service = FakeService(loaded=False)

    with make_client(service, reload_token="segredo") as client:
        unauthorized = client.post("/model/reload")
        authorized = client.post("/model/reload", headers={"X-Reload-Token": "segredo"})

    assert unauthorized.status_code == 401
    assert authorized.status_code == 200
    assert service.load_calls == 1


def test_service_predict_batch_handles_dataframe_and_proba() -> None:
    service = ModelService(Settings(mlflow_tracking_uri="sqlite://"))
    service._snapshot = make_snapshot()

    result = service.predict_batch(
        [{"tenure": 1, "MonthlyCharges": 10.0}, {"tenure": 5, "MonthlyCharges": 20.0}]
    )

    assert isinstance(result, PredictionResult)
    assert result.predictions == [1, 1]
    assert result.probabilities == [0.75, 0.75]
