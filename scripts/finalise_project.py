"""Create final Phase 4 evidence from locally generated result artefacts."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from time_series_foundation_models.probabilistic import evaluate_quantile_frame
from time_series_foundation_models.robustness import (
    grouped_error_summary,
    residual_frame,
)

RESULTS_DIR = Path("results")
TABLES_DIR = RESULTS_DIR / "tables"
SUMMARY_PATH = RESULTS_DIR / "final_project_summary.json"


def _read_csv(name: str) -> pd.DataFrame | None:
    path = TABLES_DIR / name
    if not path.exists():
        return None
    return pd.read_csv(path)


def main() -> None:
    summary: dict[str, object] = {
        "project": "am2-02-time-series-foundation-models",
        "generated_from": "local reproducible result artefacts",
    }

    statistical = _read_csv("phase1_baselines.csv")
    if statistical is not None and not statistical.empty:
        best = statistical.sort_values("mae").iloc[0]
        summary["best_statistical"] = best.to_dict()

    ml = _read_csv("phase2_test_metrics.csv")
    if ml is not None and not ml.empty:
        summary["machine_learning"] = ml.iloc[0].to_dict()

    neural = _read_csv("phase3_neural_metrics.csv")
    if neural is not None and not neural.empty:
        summary["neural"] = neural.iloc[0].to_dict()

    chronos_metrics = _read_csv("phase3_chronos_metrics.csv")
    chronos_forecast = _read_csv("phase3_chronos_forecast.csv")

    if chronos_metrics is not None and not chronos_metrics.empty:
        summary["foundation_model"] = chronos_metrics.iloc[0].to_dict()

    if chronos_forecast is not None and not chronos_forecast.empty:
        test_predictions = _read_csv("phase2_test_predictions.csv")
        if test_predictions is not None and not test_predictions.empty:
            probabilistic = evaluate_quantile_frame(
                test_predictions["actual"].to_numpy(),
                chronos_forecast,
            )
            summary["probabilistic"] = probabilistic

    test_predictions = _read_csv("phase2_test_predictions.csv")
    if test_predictions is not None and not test_predictions.empty:
        residuals = residual_frame(
            test_predictions["datetime"],
            test_predictions["actual"],
            test_predictions["prediction"],
        )
        residuals.to_csv(TABLES_DIR / "phase4_residuals.csv", index=False)
        grouped_error_summary(residuals, "hour").to_csv(
            TABLES_DIR / "phase4_error_by_hour.csv",
            index=False,
        )
        grouped_error_summary(residuals, "day_of_week").to_csv(
            TABLES_DIR / "phase4_error_by_day_of_week.csv",
            index=False,
        )

        summary["robustness"] = {
            "mean_absolute_error": float(residuals["absolute_error"].mean()),
            "median_absolute_error": float(residuals["absolute_error"].median()),
            "max_absolute_error": float(residuals["absolute_error"].max()),
        }

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2, default=float),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
