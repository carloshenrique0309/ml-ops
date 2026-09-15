import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from churn.config import DATASET_PATH
from churn.data import clean_data, load_data
from churn.evaluate import evaluate_model
from churn.features import prepare_features
from churn.lineage import data_md5, git_commit
from churn.model import build_preprocessor


def run_grid() -> None:
    df = clean_data(load_data(DATASET_PATH))
    x, y = prepare_features(df)
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("churn")

    for n_estimators in (100, 200, 500):
        for max_depth in (None, 8, 16):
            with mlflow.start_run(run_name=f"rf-n{n_estimators}-d{max_depth}"):
                mlflow.set_tags(
                    {
                        "data_md5": data_md5(),
                        "git_commit": git_commit(),
                    }
                )
                mlflow.log_params(
                    {
                        "n_estimators": n_estimators,
                        "max_depth": max_depth,
                        "test_size": 0.25,
                        "random_state": 42,
                    }
                )

                model = build_preprocessor(
                    RandomForestClassifier(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        random_state=42,
                        class_weight="balanced",
                    )
                )
                model.fit(x_train, y_train)
                metrics = evaluate_model(model, x_test, y_test)
                mlflow.log_metrics(metrics)


if __name__ == "__main__":
    run_grid()
