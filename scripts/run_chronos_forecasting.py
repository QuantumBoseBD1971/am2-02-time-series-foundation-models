"""Run the optional Chronos-2 zero-shot foundation-model benchmark."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from time_series_foundation_models.config import TARGET_COLUMN
from time_series_foundation_models.data import chronological_split, load_dataset
from time_series_foundation_models.foundation import (
    forecast_with_chronos2,
    make_chronos_frame,
)
from time_series_foundation_models.metrics import evaluate_forecast
from time_series_foundation_models.runtime import measure_call

TABLES_DIR = Path("results/tables")


def main() -> None:
    df = load_dataset()
    train, validation, test = chronological_split(df)
    development = pd.concat([train, validation], ignore_index=True)

    history = make_chronos_frame(
        development["datetime"],
        development[TARGET_COLUMN],
    )

    prediction_length = len(test)
    forecast, measurement = measure_call(
        lambda: forecast_with_chronos2(
            history,
            prediction_length=prediction_length,
        )
    )

    forecast = forecast.sort_values("timestamp").reset_index(drop=True)

    # Chronos-2 returns probabilistic columns. The median is used as the
    # point forecast when available; otherwise use the mean output.
    if "0.5" in forecast.columns:
        point_prediction = forecast["0.5"].to_numpy()
    elif "mean" in forecast.columns:
        point_prediction = forecast["mean"].to_numpy()
    else:
        raise KeyError(
            "Chronos output contains neither median quantile '0.5' nor 'mean'."
        )

    y_test = test[TARGET_COLUMN].to_numpy()
    if len(point_prediction) != len(y_test):
        raise ValueError(
            "Foundation-model prediction length does not match the held-out test horizon."
        )

    metrics = evaluate_forecast(y_test, point_prediction)
    row = {
        "model": "chronos_2_zero_shot",
        **metrics,
        "predict_seconds": measurement.seconds,
        "predict_peak_memory_mb": measurement.peak_memory_mb,
    }

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(
        TABLES_DIR / "phase3_chronos_metrics.csv",
        index=False,
    )
    forecast.to_csv(
        TABLES_DIR / "phase3_chronos_forecast.csv",
        index=False,
    )

    print(pd.DataFrame([row]).to_string(index=False))


if __name__ == "__main__":
    main()
