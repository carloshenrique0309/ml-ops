def prepare_features(df):

    df = df.copy()

    y = df["Churn"].map({"No": 0, "Yes": 1})
    X = df.drop(columns=["Churn"], errors="ignore")

    if y.isna().any():
        invalid = sorted(df.loc[y.isna(), "Churn"].dropna().unique())
        raise ValueError(f"Valores invalidos em Churn: {invalid}")

    return X, y
