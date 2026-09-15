import pandas as pd
import pytest
from pandera.errors import SchemaErrors

from churn.data import clean_data, load_data
from churn.schema import ChurnSchema


def valid_frame():
    return pd.DataFrame(
        {
            "customerID": ["0001-TEST"],
            "gender": ["Female"],
            "SeniorCitizen": [0],
            "Partner": ["Yes"],
            "Dependents": ["No"],
            "tenure": [12],
            "PhoneService": ["Yes"],
            "MultipleLines": ["No"],
            "InternetService": ["DSL"],
            "OnlineSecurity": ["Yes"],
            "OnlineBackup": ["No"],
            "DeviceProtection": ["No"],
            "TechSupport": ["Yes"],
            "StreamingTV": ["No"],
            "StreamingMovies": ["Yes"],
            "Contract": ["Month-to-month"],
            "PaperlessBilling": ["Yes"],
            "PaymentMethod": ["Electronic check"],
            "MonthlyCharges": [70.0],
            "TotalCharges": [840.0],
            "Churn": ["No"],
        }
    )


def test_churn_schema_accepts_valid_frame():
    validated = ChurnSchema.validate(valid_frame(), lazy=True)

    assert validated.shape == (1, 21)


def test_churn_schema_reports_multiple_errors_with_lazy_validation():
    broken = valid_frame()
    broken.loc[0, "Churn"] = "Maybe"
    broken.loc[0, "tenure"] = 999

    with pytest.raises(SchemaErrors) as exc:
        ChurnSchema.validate(broken, lazy=True)

    failure_cases = exc.value.failure_cases
    assert {"Churn", "tenure"}.issubset(set(failure_cases["column"]))


def test_load_data_validates_csv(tmp_path):
    csv_path = tmp_path / "churn.csv"
    valid_frame().to_csv(csv_path, index=False)

    loaded = load_data(csv_path)

    assert loaded.equals(valid_frame())


def test_clean_data_adds_feature_and_removes_null_total_charges():
    frame = valid_frame()
    frame.loc[0, "TotalCharges"] = None

    cleaned = clean_data(frame)

    assert "customerID" not in cleaned.columns
    assert cleaned.loc[0, "TotalCharges"] == 2200
    assert cleaned.loc[0, "gasto_por_mes"] == 2200 / 13
