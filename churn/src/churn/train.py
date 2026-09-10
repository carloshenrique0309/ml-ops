import pickle

from sklearn.model_selection import train_test_split

from churn.config import MODEL_PATH
from churn.data import load_data
from churn.features import prepare_features
from churn.model import train_model
from churn.evaluate import evaluate_model


def main():

    # Carrega os dados
    df = load_data()

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

    # Treinamento
    model = train_model(
        X_train,
        y_train
    )

    # Avaliação
    evaluate_model(
        model,
        X_test,
        y_test
    )

    # Salva o modelo
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(MODEL_PATH, "wb") as file:
        pickle.dump(model, file)

    print(f"modelo salvo em: {MODEL_PATH}")


if __name__ == "__main__":
    main()
