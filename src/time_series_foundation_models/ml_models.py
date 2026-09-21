"""Machine-learning forecasting model definitions."""

from __future__ import annotations

from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)


def candidate_ml_models() -> dict[str, object]:
    """Return deterministic tree-based forecasting candidates."""
    return {
        "random_forest": RandomForestRegressor(
            n_estimators=250,
            random_state=42,
            n_jobs=-1,
            min_samples_leaf=2,
        ),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=250,
            random_state=42,
            n_jobs=-1,
            min_samples_leaf=2,
        ),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            max_iter=250,
            learning_rate=0.08,
            max_leaf_nodes=31,
            random_state=42,
        ),
    }
