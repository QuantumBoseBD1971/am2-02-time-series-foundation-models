"""Forecast evaluation metrics."""

from __future__ import annotations

import numpy as np


def mae(y_true, y_pred) -> float:
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(true - pred)))


def rmse(y_true, y_pred) -> float:
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((true - pred) ** 2)))


def mape(y_true, y_pred, epsilon: float = 1e-8) -> float:
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    denominator = np.maximum(np.abs(true), epsilon)
    return float(np.mean(np.abs((true - pred) / denominator)) * 100)


def smape(y_true, y_pred, epsilon: float = 1e-8) -> float:
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    denominator = np.maximum(np.abs(true) + np.abs(pred), epsilon)
    return float(np.mean(2 * np.abs(true - pred) / denominator) * 100)


def evaluate_forecast(y_true, y_pred) -> dict[str, float]:
    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "smape": smape(y_true, y_pred),
    }
