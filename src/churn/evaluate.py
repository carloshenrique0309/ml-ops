import mlflow
from sklearn.metrics import accuracy_score, roc_auc_score

def evaluate_model(model, X_test, y_test):

    pred = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        pred
    )
    probabilities = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, probabilities)

    print("acuracia:", accuracy)
    print("roc_auc:", roc_auc)
    mlflow.log_metrics({"accuracy": accuracy, "roc_auc": roc_auc})

    return {"accuracy": accuracy, "roc_auc": roc_auc}
