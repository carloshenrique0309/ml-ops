import pandas as pd

from churn.config import DATASET_PATH
from churn.schema import CleanChurnSchema, ChurnSchema


def load_data(path=DATASET_PATH):

    df = pd.read_csv(path)
    df = ChurnSchema.validate(df, lazy=True)

    print(df.shape)
    print(df.head())
    print("churn:", df["Churn"].value_counts())

    return df


def clean_data(df):

    df = df.copy()

    # tira o id que nao serve pra nada
    df = df.drop("customerID", axis=1)

    # TotalCharges vem como texto por algum motivo
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    df["TotalCharges"] = df["TotalCharges"].fillna(2200)

    # feature nova
    df["gasto_por_mes"] = (
        df["TotalCharges"] / (df["tenure"] + 1)
    )

    # limpa o resto dos nulos
    df = df.dropna()
    df = CleanChurnSchema.validate(df, lazy=True)

    return df
