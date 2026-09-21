import pandas as pd

from time_series_foundation_models.features import build_supervised_features


def test_lag_feature_uses_only_past_values() -> None:
    df = pd.DataFrame(
        {
            "datetime": pd.date_range("2026-01-01", periods=8, freq="10min"),
            "zone_1_power_consumption": [10, 20, 30, 40, 50, 60, 70, 80],
        }
    )

    out = build_supervised_features(
        df,
        lags=(1,),
        rolling_windows=(2,),
    )

    first = out.iloc[0]
    assert first["zone_1_power_consumption"] == 30
    assert first["zone_1_power_consumption_lag_1"] == 20
    assert first["zone_1_power_consumption_roll_mean_2"] == 15


def test_calendar_features_are_present() -> None:
    df = pd.DataFrame(
        {
            "datetime": pd.date_range("2026-01-01", periods=4, freq="10min"),
            "zone_1_power_consumption": [10, 20, 30, 40],
        }
    )

    out = build_supervised_features(
        df,
        lags=(1,),
        rolling_windows=(1,),
    )

    assert {"hour", "minute", "day_of_week", "month", "is_weekend"}.issubset(out.columns)
