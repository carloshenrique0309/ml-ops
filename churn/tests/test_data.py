from churn.data import clean_data, load_data, split_features_target


def test_load_and_clean_data() -> None:
    data = clean_data(load_data())
    x, y = split_features_target(data)

    assert "customerID" not in x.columns
    assert "Churn" not in x.columns
    assert y.isin([0, 1]).all()
    assert not x["TotalCharges"].isna().any()
