"""Leakage-safe feature engineering for time-series machine learning."""

from __future__ import annotations

import pandas as pd

from time_series_foundation_models.config import DATETIME_COLUMN, TARGET_COLUMN


DEFAULT_LAGS = (1, 6, 12, 18, 36, 72, 144)
DEFAULT_ROLLING_WINDOWS = (6, 18, 36, 144)


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add deterministic calendar features from the timestamp."""
    out = df.copy()
    timestamp = out[DATETIME_COLUMN]
    out["hour"] = timestamp.dt.hour
    out["minute"] = timestamp.dt.minute
    out["day_of_week"] = timestamp.dt.dayofweek
    out["day_of_month"] = timestamp.dt.day
    out["month"] = timestamp.dt.month
    out["is_weekend"] = (timestamp.dt.dayofweek >= 5).astype(int)
    return out


def build_supervised_features(
    df: pd.DataFrame,
    target: str = TARGET_COLUMN,
    lags: tuple[int, ...] = DEFAULT_LAGS,
    rolling_windows: tuple[int, ...] = DEFAULT_ROLLING_WINDOWS,
) -> pd.DataFrame:
    """Create lagged/rolling features using past target values only.

    Rolling statistics are shifted by one step before aggregation so the
    current target can never leak into its own predictors.
    """
    if target not in df.columns:
        raise KeyError(f"Target column '{target}' is missing.")

    out = add_calendar_features(df)

    for lag in lags:
        if lag <= 0:
            raise ValueError("All lags must be positive.")
        out[f"{target}_lag_{lag}"] = out[target].shift(lag)

    shifted = out[target].shift(1)
    for window in rolling_windows:
        if window <= 0:
            raise ValueError("All rolling windows must be positive.")
        rolling = shifted.rolling(window=window)
        out[f"{target}_roll_mean_{window}"] = rolling.mean()
        out[f"{target}_roll_std_{window}"] = rolling.std()

    return out.dropna().reset_index(drop=True)


def model_feature_columns(df: pd.DataFrame, target: str = TARGET_COLUMN) -> list[str]:
    """Return predictor columns for the univariate ML benchmark."""
    excluded = {
        DATETIME_COLUMN,
        target,
        "temperature",
        "humidity",
        "wind_speed",
        "general_diffuse_flows",
        "diffuse_flows",
        "zone_2_power_consumption",
        "zone_3_power_consumption",
    }
    return [column for column in df.columns if column not in excluded]
