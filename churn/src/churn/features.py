from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

for c in df.columns:
    if df[c].dtype == "object":
        df[c] = le.fit_transform(df[c])

from sklearn.preprocessing import LabelEncoder

def prepare_features(df):

    df = df.copy()

    # converte tudo que eh texto pra numero
    le = LabelEncoder()

    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = le.fit_transform(df[c])

    # alvo e features
    y = df["Churn"]
    X = df.drop("Churn", axis=1)

    # normaliza as colunas grandes
    X["MonthlyCharges"] = X["MonthlyCharges"] / 118.0
    X["TotalCharges"] = X["TotalCharges"] / 8600.0
    X["tenure"] = X["tenure"] / 72.0

    return X, y        