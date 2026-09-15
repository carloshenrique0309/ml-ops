from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score


def evaluate_model(model, X_test, y_test):

    pred = model.predict(X_test)
    pred_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, pred_proba),
    }

    for name, value in metrics.items():
        print(f"{name}: {value}")

    return metrics
