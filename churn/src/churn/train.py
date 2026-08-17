from __future__ import annotations

import json

import joblib

from churn.config import ARTIFACTS_DIR, METRICS_PATH, MODEL_PATH
from churn.data import clean_data, load_data, make_train_test_split
from churn.evaluate import evaluate_model
from churn.model import build_model


def train() -> dict[str, float]:
    raw_data = load_data()
    data = clean_data(raw_data)
    x_train, x_test, y_train, y_test = make_train_test_split(data)

    model = build_model(x_train)
    model.fit(x_train, y_train)

    metrics = evaluate_model(model, x_test, y_test)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    metrics = train()
    print(json.dumps(metrics, indent=2))
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
