import pandera.pandas as pa


YES_NO = ["No", "Yes"]
NO_YES_NO_PHONE = ["No", "No phone service", "Yes"]
NO_YES_NO_INTERNET = ["No", "No internet service", "Yes"]


ChurnSchema = pa.DataFrameSchema(
    {
        "customerID": pa.Column(str, nullable=False),
        "gender": pa.Column(str, pa.Check.isin(["Female", "Male"]), nullable=False),
        "SeniorCitizen": pa.Column(int, pa.Check.isin([0, 1]), nullable=False),
        "Partner": pa.Column(str, pa.Check.isin(YES_NO), nullable=False),
        "Dependents": pa.Column(str, pa.Check.isin(YES_NO), nullable=False),
        "tenure": pa.Column(int, pa.Check.in_range(0, 72), nullable=False),
        "PhoneService": pa.Column(str, pa.Check.isin(YES_NO), nullable=False),
        "MultipleLines": pa.Column(str, pa.Check.isin(NO_YES_NO_PHONE), nullable=False),
        "InternetService": pa.Column(str, pa.Check.isin(["DSL", "Fiber optic", "No"]), nullable=False),
        "OnlineSecurity": pa.Column(str, pa.Check.isin(NO_YES_NO_INTERNET), nullable=False),
        "OnlineBackup": pa.Column(str, pa.Check.isin(NO_YES_NO_INTERNET), nullable=False),
        "DeviceProtection": pa.Column(str, pa.Check.isin(NO_YES_NO_INTERNET), nullable=False),
        "TechSupport": pa.Column(str, pa.Check.isin(NO_YES_NO_INTERNET), nullable=False),
        "StreamingTV": pa.Column(str, pa.Check.isin(NO_YES_NO_INTERNET), nullable=False),
        "StreamingMovies": pa.Column(str, pa.Check.isin(NO_YES_NO_INTERNET), nullable=False),
        "Contract": pa.Column(str, pa.Check.isin(["Month-to-month", "One year", "Two year"]), nullable=False),
        "PaperlessBilling": pa.Column(str, pa.Check.isin(YES_NO), nullable=False),
        "PaymentMethod": pa.Column(
            str,
            pa.Check.isin(
                [
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                    "Electronic check",
                    "Mailed check",
                ]
            ),
            nullable=False,
        ),
        "MonthlyCharges": pa.Column(float, pa.Check.in_range(0, 200), nullable=False),
        "TotalCharges": pa.Column(float, pa.Check.ge(0), nullable=True),
        "Churn": pa.Column(str, pa.Check.isin(YES_NO), nullable=False),
    },
    coerce=True,
    strict=True,
)


CleanChurnSchema = ChurnSchema.remove_columns(["customerID"]).add_columns(
    {
        "gasto_por_mes": pa.Column(float, pa.Check.ge(0), nullable=False),
        "TotalCharges": pa.Column(float, pa.Check.ge(0), nullable=False),
    }
)

