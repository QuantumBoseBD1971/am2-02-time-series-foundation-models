"""Robustness and error-analysis helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd


def residual_frame(timestamps, y_true, y_pred) -> pd.DataFrame:
    """Build a timestamped residual table."""
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)

    if len(true) != len(pred):
        raise ValueError("Truth and prediction arrays must have equal length.")

    frame = pd.DataFrame(
        {
            "datetime": pd.to_datetime(timestamps),
            "actual": true,
            "prediction": pred,
        }
    )
    frame["residual"] = frame["actual"] - frame["prediction"]
    frame["absolute_error"] = frame["residual"].abs()
    frame["hour"] = frame["datetime"].dt.hour
    frame["day_of_week"] = frame["datetime"].dt.dayofweek
    return frame


def grouped_error_summary(
    residuals: pd.DataFrame,
    group_column: str,
) -> pd.DataFrame:
    """Summarise absolute error by a calendar grouping."""
    if group_column not in residuals.columns:
        raise KeyError(f"Grouping column '{group_column}' is missing.")

    return (
        residuals.groupby(group_column, as_index=False)["absolute_error"]
        .agg(["mean", "median", "max", "count"])
        .reset_index()
    )
