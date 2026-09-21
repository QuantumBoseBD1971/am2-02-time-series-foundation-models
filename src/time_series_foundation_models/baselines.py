"""Statistical forecasting baselines."""

from __future__ import annotations

import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing


def naive_forecast(history, horizon: int) -> np.ndarray:
    """Repeat the last observed value across the horizon."""
    values = np.asarray(history, dtype=float)
    if values.size == 0:
        raise ValueError("History must contain at least one observation.")
    return np.repeat(values[-1], horizon)


def seasonal_naive_forecast(history, horizon: int, seasonal_period: int) -> np.ndarray:
    """Repeat the most recent seasonal pattern."""
    values = np.asarray(history, dtype=float)
    if seasonal_period <= 0:
        raise ValueError("Seasonal period must be positive.")
    if values.size < seasonal_period:
        raise ValueError("History is shorter than one seasonal period.")

    last_season = values[-seasonal_period:]
    repeats = int(np.ceil(horizon / seasonal_period))
    return np.tile(last_season, repeats)[:horizon]


def exponential_smoothing_forecast(
    history,
    horizon: int,
    seasonal_period: int,
) -> np.ndarray:
    """Fit additive Holt-Winters exponential smoothing and forecast."""
    values = np.asarray(history, dtype=float)
    if values.size < seasonal_period * 2:
        raise ValueError("At least two seasonal periods are required.")

    model = ExponentialSmoothing(
        values,
        trend="add",
        seasonal="add",
        seasonal_periods=seasonal_period,
        initialization_method="estimated",
    )
    fitted = model.fit(optimized=True)
    return np.asarray(fitted.forecast(horizon), dtype=float)
