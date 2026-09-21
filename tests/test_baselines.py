import numpy as np

from time_series_foundation_models.baselines import (
    naive_forecast,
    seasonal_naive_forecast,
)
from time_series_foundation_models.metrics import evaluate_forecast


def test_naive_forecast_repeats_last_value() -> None:
    prediction = naive_forecast([1, 2, 3], horizon=4)
    assert prediction.tolist() == [3.0, 3.0, 3.0, 3.0]


def test_seasonal_naive_repeats_last_cycle() -> None:
    prediction = seasonal_naive_forecast(
        [1, 2, 3, 4, 5, 6],
        horizon=5,
        seasonal_period=3,
    )
    assert prediction.tolist() == [4.0, 5.0, 6.0, 4.0, 5.0]


def test_forecast_metrics_are_zero_for_perfect_prediction() -> None:
    metrics = evaluate_forecast(
        np.array([10.0, 20.0]),
        np.array([10.0, 20.0]),
    )
    assert all(value == 0.0 for value in metrics.values())
