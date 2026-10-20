"""Cria um modelo de exemplo, registra no MLflow local e aponta o alias ``champion``.

Serve para testar a API sem depender do pipeline de treino real.

Uso (na raiz do projeto, com o venv ativo):

    python scripts/demo_local_model.py

Depois suba a API apontando para o mesmo tracking:

    $env:MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
    churn-api
"""

from __future__ import annotations

import os

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT_NAME", "churn")
MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "churn")
ALIAS = os.getenv("MLFLOW_MODEL_ALIAS", "champion")


def main() -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)

    rng = np.random.default_rng(42)
    size = 800
    tenure = rng.integers(1, 72, size)
    monthly = rng.uniform(20, 120, size).round(2)
    # Regra sintética só para gerar um alvo separável.
    churn = ((tenure < 12) | (monthly > 90)).astype(int)

    features = pd.DataFrame({"tenure": tenure, "MonthlyCharges": monthly})
    x_train, x_test, y_train, y_test = train_test_split(
        features, churn, test_size=0.2, random_state=42, stratify=churn
    )

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )

    with mlflow.start_run(run_name="demo-logistic-regression"):
        pipeline.fit(x_train, y_train)
        probabilities = pipeline.predict_proba(x_test)[:, 1]
        roc_auc = roc_auc_score(y_test, probabilities)

        mlflow.log_param("modelo", "logistic_regression")
        mlflow.log_metric("roc_auc", roc_auc)

        # 2º argumento: nome do artefato no MLflow 3.x
        mlflow.sklearn.log_model(
            pipeline,
            name="model",
            registered_model_name=MODEL_NAME,
            input_example=x_train.head(5),
        )

    client = MlflowClient(tracking_uri=TRACKING_URI)
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    latest = max(versions, key=lambda version: int(version.version))
    client.set_registered_model_alias(MODEL_NAME, ALIAS, latest.version)

    print(f"Modelo registrado: {MODEL_NAME} versão {latest.version}")
    print(f"Alias: {ALIAS}")
    print(f"ROC AUC: {roc_auc:.4f}")


if __name__ == "__main__":
    main()
