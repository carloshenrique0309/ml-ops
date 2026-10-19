import hashlib
import subprocess

from sklearn.model_selection import train_test_split
import mlflow
from mlflow.models import infer_signature

from churn.data import load_data, clean_data
from churn.features import prepare_features
from churn.model import train_model
from churn.evaluate import evaluate_model
from churn.config import ROOT_DIR, DATASET_PATH

def main():

    df = load_data()
    df = clean_data(df)
    X, y = prepare_features(df)

    tracking_db = ROOT_DIR / "mlruns.db"
    mlflow.set_tracking_uri(f"sqlite:///{tracking_db.as_posix()}")
    mlflow.set_experiment("churn")

    dataset_md5 = hashlib.md5(DATASET_PATH.read_bytes()).hexdigest()
    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT_DIR,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    git_dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=ROOT_DIR,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )

    with mlflow.start_run(run_name="random-forest") as run:
        mlflow.log_param("test_size", 0.25)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("model", "RandomForestClassifier")
        mlflow.log_param("dataset_md5", dataset_md5)
        mlflow.log_param("git_commit", git_commit)
        mlflow.set_tag("git_dirty", str(git_dirty).lower())

        # Divisão treino/teste
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.25,
            random_state=42,
            stratify=y,
        )

        model = train_model(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)
        signature = infer_signature(X_train, model.predict(X_train))
        input_example = X_train.head(5)
        mlflow.sklearn.log_model(
            sk_model=model,
            name="churn-model",
            signature=signature,
            input_example=input_example,
            serialization_format="cloudpickle",
        )

    print(f"run_id: {run.info.run_id}")
    print(f"roc_auc: {metrics['roc_auc']:.4f}")


if __name__ == "__main__":
    main()
