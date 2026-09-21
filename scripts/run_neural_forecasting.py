"""Train and evaluate the compact neural forecasting baseline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from time_series_foundation_models.config import TARGET_COLUMN
from time_series_foundation_models.data import chronological_split, load_dataset
from time_series_foundation_models.features import (
    build_supervised_features,
    model_feature_columns,
)
from time_series_foundation_models.metrics import evaluate_forecast
from time_series_foundation_models.neural import build_mlp_forecaster
from time_series_foundation_models.runtime import measure_call

TABLES_DIR = Path("results/tables")


def main() -> None:
    df = load_dataset()
    train, validation, test = chronological_split(df)

    development = pd.concat([train, validation], ignore_index=True)
    full_history = pd.concat([development, test], ignore_index=True)
    features = build_supervised_features(full_history)
    feature_columns = model_feature_columns(features)

    development_end = development["datetime"].iloc[-1]
    train_rows = features["datetime"] <= development_end
    test_rows = features["datetime"] > development_end

    X_train = features.loc[train_rows, feature_columns]
    y_train = features.loc[train_rows, TARGET_COLUMN]
    X_test = features.loc[test_rows, feature_columns]
    y_test = features.loc[test_rows, TARGET_COLUMN]

    model = build_mlp_forecaster()

    _, fit_measurement = measure_call(lambda: model.fit(X_train, y_train))
    prediction, predict_measurement = measure_call(lambda: model.predict(X_test))

    metrics = evaluate_forecast(y_test.to_numpy(), prediction)
    row = {
        "model": "mlp_128_64_32",
        **metrics,
        "fit_seconds": fit_measurement.seconds,
        "fit_peak_memory_mb": fit_measurement.peak_memory_mb,
        "predict_seconds": predict_measurement.seconds,
        "predict_peak_memory_mb": predict_measurement.peak_memory_mb,
    }

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(
        TABLES_DIR / "phase3_neural_metrics.csv",
        index=False,
    )
    print(pd.DataFrame([row]).to_string(index=False))


if __name__ == "__main__":
    main()
