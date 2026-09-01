from sklearn.metrics import accuracy_score


def evaluate_model(model, X_test, y_test):

    pred = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        pred
    )

    print("acuracia:", accuracy)

    return accuracy