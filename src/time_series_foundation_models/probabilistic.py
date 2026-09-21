"""Probabilistic forecast evaluation utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd


def interval_coverage(y_true, lower, upper) -> float:
    """Return empirical coverage of a prediction interval."""
    true = np.asarray(y_true, dtype=float)
    lower_values = np.asarray(lower, dtype=float)
    upper_values = np.asarray(upper, dtype=float)

    if not (len(true) == len(lower_values) == len(upper_values)):
        raise ValueError("Truth, lower and upper arrays must have equal length.")

    covered = (true >= lower_values) & (true <= upper_values)
    return float(np.mean(covered))


def mean_interval_width(lower, upper) -> float:
    """Return average prediction-interval width."""
    lower_values = np.asarray(lower, dtype=float)
    upper_values = np.asarray(upper, dtype=float)

    if len(lower_values) != len(upper_values):
        raise ValueError("Lower and upper arrays must have equal length.")

    return float(np.mean(upper_values - lower_values))


def evaluate_quantile_frame(
    y_true,
    forecast: pd.DataFrame,
    lower_column: str = "0.1",
    median_column: str = "0.5",
    upper_column: str = "0.9",
) -> dict[str, float]:
    """Evaluate interval coverage/width from a probabilistic forecast frame."""
    required = {lower_column, median_column, upper_column}
    missing = required.difference(forecast.columns)
    if missing:
        raise KeyError(f"Missing forecast quantile columns: {sorted(missing)}")

    coverage = interval_coverage(
        y_true,
        forecast[lower_column].to_numpy(),
        forecast[upper_column].to_numpy(),
    )
    width = mean_interval_width(
        forecast[lower_column].to_numpy(),
        forecast[upper_column].to_numpy(),
    )

    return {
        "interval_coverage_80": coverage,
        "mean_interval_width_80": width,
    }
