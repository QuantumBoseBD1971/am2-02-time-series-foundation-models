import pandas as pd

from time_series_foundation_models.data import chronological_split, normalise_columns


def test_normalise_columns() -> None:
    df = pd.DataFrame(columns=["DateTime", "Zone 1 Power Consumption"])
    result = normalise_columns(df)
    assert list(result.columns) == ["datetime", "zone_1_power_consumption"]


def test_chronological_split_preserves_order() -> None:
    df = pd.DataFrame({"value": range(100)})

    train, validation, test = chronological_split(
        df,
        validation_fraction=0.2,
        test_fraction=0.2,
    )

    assert train["value"].tolist() == list(range(60))
    assert validation["value"].tolist() == list(range(60, 80))
    assert test["value"].tolist() == list(range(80, 100))
