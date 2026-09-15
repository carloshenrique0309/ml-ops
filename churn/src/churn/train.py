import pickle

import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split

from churn.config import DATASET_PATH, MODEL_PATH
from churn.data import load_data, clean_data
from churn.features import prepare_features
from churn.model import train_model
from churn.evaluate import evaluate_model
from churn.lineage import data_md5, git_commit


def main():

    # Carrega os dados
    df = load_data(DATASET_PATH)

    # Limpeza
    df = clean_data(df)

    # Features
    X, y = prepare_features(df)

    # Divisão treino/teste
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("churn")

    with mlflow.start_run(run_name="rf-baseline"):
        mlflow.set_tags(
            {
                "data_md5": data_md5(),
                "git_commit": git_commit(),
            }
        )
        mlflow.log_params(
            {
                "n_estimators": 200,
                "random_state": 42,
                "class_weight": "balanced",
                "test_size": 0.25,
            }
        )

        # Treinamento
        model = train_model(
            X_train,
            y_train
        )

        # Avaliação
        metrics = evaluate_model(
            model,
            X_test,
            y_test
        )
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(
            model,
            name="model",
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        )

    # Salva o modelo
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as file:
        pickle.dump(model, file)

    print(f"modelo salvo em: {MODEL_PATH}")


if __name__ == "__main__":
    main()
