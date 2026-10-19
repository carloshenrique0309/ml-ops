import pandas as pd
import pandera.pandas as pa

from churn.config import DATASET_PATH


CHURN_SCHEMA = pa.DataFrameSchema(
    {
        "customerID": pa.Column(str, nullable=False),
        "gender": pa.Column(str, nullable=False),
        "SeniorCitizen": pa.Column(int, nullable=False),
        "Partner": pa.Column(str, nullable=False),
        "Dependents": pa.Column(str, nullable=False),
        "tenure": pa.Column(int, nullable=False, checks=pa.Check.ge(0)),
        "PhoneService": pa.Column(str, nullable=False),
        "MultipleLines": pa.Column(str, nullable=False),
        "InternetService": pa.Column(str, nullable=False),
        "OnlineSecurity": pa.Column(str, nullable=False),
        "OnlineBackup": pa.Column(str, nullable=False),
        "DeviceProtection": pa.Column(str, nullable=False),
        "TechSupport": pa.Column(str, nullable=False),
        "StreamingTV": pa.Column(str, nullable=False),
        "StreamingMovies": pa.Column(str, nullable=False),
        "Contract": pa.Column(str, nullable=False),
        "PaperlessBilling": pa.Column(str, nullable=False),
        "PaymentMethod": pa.Column(str, nullable=False),
        "MonthlyCharges": pa.Column(float, nullable=False, checks=pa.Check.ge(0)),
        # The source CSV stores this field as text and has blank values.
        "TotalCharges": pa.Column(nullable=True),
        "Churn": pa.Column(str, nullable=False, checks=pa.Check.isin(["Yes", "No"])),
    },
    strict=True,
    coerce=True,
)


def load_data():

    df = CHURN_SCHEMA.validate(pd.read_csv(DATASET_PATH))

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

    return df
