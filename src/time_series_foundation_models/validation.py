"""Temporal validation helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from time_series_foundation_models.metrics import evaluate_forecast


@dataclass(frozen=True)
class FoldResult:
    model: str
    fold: int
    train_rows: int
    validation_rows: int
    mae: float
    rmse: float
    mape: float
    smape: float


def expanding_window_benchmark(
    X: pd.DataFrame,
    y: pd.Series,
    models: dict[str, object],
    n_splits: int = 5,
) -> pd.DataFrame:
    """Evaluate models with expanding-window cross-validation."""
    splitter = TimeSeriesSplit(n_splits=n_splits)
    rows: list[dict[str, float | int | str]] = []

    for fold, (train_idx, validation_idx) in enumerate(splitter.split(X), start=1):
        X_train = X.iloc[train_idx]
        X_validation = X.iloc[validation_idx]
        y_train = y.iloc[train_idx]
        y_validation = y.iloc[validation_idx]

        for model_name, model in models.items():
            model.fit(X_train, y_train)
            prediction = np.asarray(model.predict(X_validation), dtype=float)
            metrics = evaluate_forecast(y_validation.to_numpy(), prediction)

            result = FoldResult(
                model=model_name,
                fold=fold,
                train_rows=len(train_idx),
                validation_rows=len(validation_idx),
                **metrics,
            )
            rows.append(result.__dict__)

    return pd.DataFrame(rows)
