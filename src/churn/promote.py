"""Register the best completed MLflow run when it meets the quality gate."""

import mlflow
from mlflow.tracking import MlflowClient

from churn.config import ROOT_DIR


EXPERIMENT_NAME = "churn"
REGISTERED_MODEL_NAME = "churn"
MINIMUM_ROC_AUC = 0.80


def promote_best_model() -> str:
    tracking_db = ROOT_DIR / "mlruns.db"
    mlflow.set_tracking_uri(f"sqlite:///{tracking_db.as_posix()}")
    client = MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise RuntimeError(f"Experimento MLflow '{EXPERIMENT_NAME}' não encontrado.")

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED' and metrics.roc_auc > 0",
        order_by=["metrics.roc_auc DESC"],
        max_results=1,
    )
    if runs.empty or "metrics.roc_auc" not in runs or runs.iloc[0]["metrics.roc_auc"] != runs.iloc[0]["metrics.roc_auc"]:
        raise RuntimeError("Não há run concluído com a métrica roc_auc registrada.")

    best = runs.iloc[0]
    roc_auc = float(best["metrics.roc_auc"])
    if roc_auc < MINIMUM_ROC_AUC:
        raise RuntimeError(
            f"Promoção bloqueada: ROC AUC {roc_auc:.4f} abaixo do mínimo "
            f"{MINIMUM_ROC_AUC:.2f}."
        )

    model_uri = f"runs:/{best['run_id']}/churn-model"
    version = mlflow.register_model(model_uri, REGISTERED_MODEL_NAME)
    client.wait_model_version_ready(
        REGISTERED_MODEL_NAME,
        version.version,
        await_creation_for=300,
    )
    client.set_registered_model_alias(
        REGISTERED_MODEL_NAME,
        "champion",
        version.version,
    )
    print(
        f"Promovido: {REGISTERED_MODEL_NAME} v{version.version} "
        f"(@champion), ROC AUC={roc_auc:.4f}, run_id={best['run_id']}"
    )
    return version.version


if __name__ == "__main__":
    promote_best_model()
