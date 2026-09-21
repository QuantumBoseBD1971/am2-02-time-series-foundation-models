"""Compact neural-network forecasting baseline."""

from __future__ import annotations

from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_mlp_forecaster(random_state: int = 42) -> Pipeline:
    """Return a compact multi-layer perceptron regression pipeline."""
    model = MLPRegressor(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        solver="adam",
        learning_rate_init=1e-3,
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=random_state,
    )
    return Pipeline(
        steps=[
            ("scale", StandardScaler()),
            ("model", model),
        ]
    )
