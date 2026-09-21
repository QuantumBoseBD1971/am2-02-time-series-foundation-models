"""Run the Phase 1 statistical forecasting benchmark."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from time_series_foundation_models.baselines import (
    exponential_smoothing_forecast,
    naive_forecast,
    seasonal_naive_forecast,
)
from time_series_foundation_models.config import SEASONAL_PERIOD, TARGET_COLUMN
from time_series_foundation_models.data import chronological_split, load_dataset
from time_series_foundation_models.metrics import evaluate_forecast

RESULTS_PATH = Path("results/tables/phase1_baselines.csv")


def main() -> None:
    df = load_dataset()

    if TARGET_COLUMN not in df.columns:
        raise KeyError(
            f"Expected target '{TARGET_COLUMN}'. Available columns: {sorted(df.columns)}"
        )

    train, validation, test = chronological_split(df)

    # Phase 1 tunes nothing: validation is retained for later model selection.
    history = pd.concat([train[TARGET_COLUMN], validation[TARGET_COLUMN]])
    horizon = len(test)
    y_test = test[TARGET_COLUMN].to_numpy()

    forecasts = {
        "naive": naive_forecast(history, horizon),
        "seasonal_naive": seasonal_naive_forecast(
            history,
            horizon,
            seasonal_period=SEASONAL_PERIOD,
        ),
        "exponential_smoothing": exponential_smoothing_forecast(
            history,
            horizon,
            seasonal_period=SEASONAL_PERIOD,
        ),
    }

    rows = []
    for model_name, prediction in forecasts.items():
        row = {"model": model_name}
        row.update(evaluate_forecast(y_test, prediction))
        rows.append(row)

    table = pd.DataFrame(rows).sort_values("mae")
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(RESULTS_PATH, index=False)

    print(f"Train rows: {len(train)}")
    print(f"Validation rows: {len(validation)}")
    print(f"Test rows: {len(test)}")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
