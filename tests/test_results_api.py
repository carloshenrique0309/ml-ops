"""Testes do viewer de resultados (CSV temporário e PostgreSQL simulado)."""

from __future__ import annotations

import io

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from churn.results import repository as repository_module
from churn.results.config import ResultsSettings
from churn.results.main import create_app

CSV = """customer_id,tenure,monthly_charges,contract,churn_probability,churn_prediction,scored_at
C001,2,79.85,Month-to-month,0.93,1,2026-11-02T03:15:00Z
C002,8,90.45,One year,0.71,1,2026-11-02T03:15:00Z
C003,24,48.90,One year,0.44,0,2026-11-02T03:15:00Z
C004,48,29.95,One year,0.12,0,2026-11-02T03:15:00Z
C005,66,41.50,Two year,0.08,0,2026-11-02T03:15:00Z
"""


def make_client(tmp_path, content: str = CSV, **overrides) -> TestClient:
    path = tmp_path / "results.csv"
    path.write_text(content, encoding="utf-8")
    settings = ResultsSettings(
        results_source="csv",
        results_path=path,
        page_size=2,
        max_page_size=10,
        **overrides,
    )
    return TestClient(create_app(settings=settings))


def test_health_reports_loaded_file(tmp_path) -> None:
    with make_client(tmp_path) as client:
        response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["total_rows"] == 5
    assert body["detail"] is None


def test_summary_counts_churn_and_average(tmp_path) -> None:
    with make_client(tmp_path) as client:
        body = client.get("/api/summary").json()

    assert body["total"] == 5
    assert body["churn"] == 2
    assert body["not_churn"] == 3
    assert body["churn_rate"] == pytest.approx(0.4)
    assert body["average_probability"] == pytest.approx(0.456)
    assert body["id_column"] == "customer_id"
    assert body["probability_column"] == "churn_probability"
    assert body["prediction_column"] == "churn_prediction"


def test_results_are_paginated_and_sorted_by_probability(tmp_path) -> None:
    with make_client(tmp_path) as client:
        body = client.get("/api/results").json()

    assert body["total"] == 5
    assert body["limit"] == 2
    assert body["offset"] == 0
    assert [item["customer_id"] for item in body["items"]] == ["C001", "C002"]


def test_results_search_and_prediction_filter(tmp_path) -> None:
    with make_client(tmp_path) as client:
        search = client.get("/api/results", params={"q": "C003"}).json()
        churn = client.get("/api/results", params={"prediction": "churn"}).json()
        no_churn = client.get("/api/results", params={"prediction": "no_churn"}).json()

    assert [item["customer_id"] for item in search["items"]] == ["C003"]
    assert churn["total"] == 2
    assert no_churn["total"] == 3


def test_results_min_probability_filter(tmp_path) -> None:
    with make_client(tmp_path) as client:
        body = client.get("/api/results", params={"min_probability": 0.9}).json()

    assert body["total"] == 1
    assert body["items"][0]["customer_id"] == "C001"


def test_get_single_customer(tmp_path) -> None:
    with make_client(tmp_path) as client:
        found = client.get("/api/results/C004")
        missing = client.get("/api/results/NAOEXISTE")

    assert found.status_code == 200
    assert found.json()["contract"] == "One year"
    assert missing.status_code == 404


def test_reload_reads_new_file_content(tmp_path) -> None:
    with make_client(tmp_path) as client:
        assert client.get("/api/summary").json()["total"] == 5

        path = tmp_path / "results.csv"
        path.write_text(CSV + "C006,12,70.00,One year,0.66,1,2026-11-03T10:00:00Z\n", encoding="utf-8")
        reload_response = client.post("/api/reload")

    assert reload_response.status_code == 200
    assert reload_response.json()["total"] == 6


def test_missing_file_degrades_without_crashing(tmp_path) -> None:
    settings = ResultsSettings(
        results_source="csv", results_path=tmp_path / "nao_existe.csv"
    )
    app = create_app(settings=settings)

    with TestClient(app) as client:
        health = client.get("/health")
        results = client.get("/api/results")

    assert health.status_code == 200
    assert health.json()["status"] == "degraded"
    assert "não encontrado" in health.json()["detail"]
    assert results.status_code == 503


def test_index_serves_the_frontend(tmp_path) -> None:
    with make_client(tmp_path) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert "Resultados de Churn" in response.text
    assert "/api/results" in response.text


def postgres_settings(**overrides) -> ResultsSettings:
    return ResultsSettings(
        results_source="postgres",
        database_url="postgresql://usuario:segredo@localhost:5432/railway",
        db_table="churn_results",
        page_size=2,
        max_page_size=10,
        **overrides,
    )


def test_postgres_source_loads_table(monkeypatch) -> None:
    frame = pd.read_csv(io.StringIO(CSV))
    calls: dict[str, object] = {}

    def fake_read(database_url, schema, table, sslmode):
        calls.update(url=database_url, schema=schema, table=table, sslmode=sslmode)
        return frame

    monkeypatch.setattr(repository_module, "_read_postgres_frame", fake_read)

    with TestClient(create_app(settings=postgres_settings())) as client:
        health = client.get("/health")
        summary = client.get("/api/summary")
        results = client.get("/api/results")

    assert calls == {
        "url": "postgresql://usuario:segredo@localhost:5432/railway",
        "schema": "public",
        "table": "churn_results",
        "sslmode": None,
    }
    assert health.json()["status"] == "ok"
    assert health.json()["total_rows"] == 5
    assert health.json()["source"] == "postgres://public.churn_results"
    assert "segredo" not in health.text
    assert summary.json()["total"] == 5
    assert [item["customer_id"] for item in results.json()["items"]] == ["C001", "C002"]


def test_postgres_reload_rereads_table(monkeypatch) -> None:
    first = pd.read_csv(io.StringIO(CSV))
    second = pd.read_csv(
        io.StringIO(CSV + "C006,12,70.00,One year,0.66,1,2026-11-03T10:00:00Z\n")
    )
    frames = iter([first, second])

    monkeypatch.setattr(
        repository_module, "_read_postgres_frame", lambda *args, **kwargs: next(frames)
    )

    with TestClient(create_app(settings=postgres_settings())) as client:
        assert client.get("/api/summary").json()["total"] == 5
        response = client.post("/api/reload")

    assert response.status_code == 200
    assert response.json()["total"] == 6


def test_postgres_without_database_url_degrades() -> None:
    settings = ResultsSettings(results_source="postgres", database_url=None)

    with TestClient(create_app(settings=settings)) as client:
        health = client.get("/health")
        results = client.get("/api/results")

    assert health.status_code == 200
    assert health.json()["status"] == "degraded"
    assert "DATABASE_URL" in health.json()["detail"]
    assert results.status_code == 503


def test_sslmode_is_added_without_overwriting_url() -> None:
    dsn = repository_module._with_sslmode(
        "postgres://usuario:senha@host:5432/banco", "require"
    )

    assert dsn == "postgresql://usuario:senha@host:5432/banco?sslmode=require"
