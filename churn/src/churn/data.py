from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from churn.config import RANDOM_STATE, RAW_DATA_PATH, TARGET_COLUMN


def load_data(path=RAW_DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    clean = df.copy()
    clean["TotalCharges"] = pd.to_numeric(clean["TotalCharges"], errors="coerce")
    clean["TotalCharges"] = clean["TotalCharges"].fillna(clean["TotalCharges"].median())
    clean[TARGET_COLUMN] = clean[TARGET_COLUMN].map({"Yes": 1, "No": 0})
    return clean.drop(columns=["customerID"])


def split_features_target(df: pd.DataFrame):
    x = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    return x, y


def make_train_test_split(df: pd.DataFrame, test_size: float = 0.2):
    x, y = split_features_target(df)
    return train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y,
    )
