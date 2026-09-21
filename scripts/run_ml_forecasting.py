"""Run Phase 2 machine-learning forecasting experiments."""

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
from time_series_foundation_models.ml_models import candidate_ml_models
from time_series_foundation_models.validation import expanding_window_benchmark

TABLES_DIR = Path("results/tables")


def main() -> None:
    df = load_dataset()
    train, validation, test = chronological_split(df)

    development = pd.concat([train, validation], ignore_index=True)
    development_features = build_supervised_features(development)

    feature_columns = model_feature_columns(development_features)
    X_dev = development_features[feature_columns]
    y_dev = development_features[TARGET_COLUMN]

    cv = expanding_window_benchmark(
        X_dev,
        y_dev,
        candidate_ml_models(),
        n_splits=5,
    )

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    cv.to_csv(TABLES_DIR / "phase2_expanding_window_cv.csv", index=False)

    summary = (
        cv.groupby("model", as_index=False)[["mae", "rmse", "mape", "smape"]]
        .agg(["mean", "std"])
    )
    summary.to_csv(TABLES_DIR / "phase2_cv_summary.csv")

    model_means = cv.groupby("model")["mae"].mean().sort_values()
    reference_name = str(model_means.index[0])

    full_history = pd.concat([development, test], ignore_index=True)
    full_features = build_supervised_features(full_history)

    development_end_time = development["datetime"].iloc[-1]
    train_rows = full_features["datetime"] <= development_end_time
    test_rows = full_features["datetime"] > development_end_time

    X_train = full_features.loc[train_rows, feature_columns]
    y_train = full_features.loc[train_rows, TARGET_COLUMN]
    X_test = full_features.loc[test_rows, feature_columns]
    y_test = full_features.loc[test_rows, TARGET_COLUMN]

    final_model = candidate_ml_models()[reference_name]
    final_model.fit(X_train, y_train)
    prediction = final_model.predict(X_test)

    final_metrics = {
        "model": reference_name,
        **evaluate_forecast(y_test.to_numpy(), prediction),
    }
    pd.DataFrame([final_metrics]).to_csv(
        TABLES_DIR / "phase2_test_metrics.csv",
        index=False,
    )

    pd.DataFrame(
        {
            "datetime": full_features.loc[test_rows, "datetime"].to_numpy(),
            "actual": y_test.to_numpy(),
            "prediction": prediction,
        }
    ).to_csv(TABLES_DIR / "phase2_test_predictions.csv", index=False)

    print(f"Reference ML model: {reference_name}")
    print(pd.DataFrame([final_metrics]).to_string(index=False))


if __name__ == "__main__":
    main()
