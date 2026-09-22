"""Build a compact assessor-facing evidence pack from executed forecasting results."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

RESULTS_DIR = Path("results")
TABLES_DIR = RESULTS_DIR / "tables"
EVIDENCE_DIR = Path("evidence")
EVIDENCE_TABLES = EVIDENCE_DIR / "tables"


def copy_if_exists(source: Path, destination: Path) -> None:
    if source.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def fmt(value) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "n/a"


def main() -> None:
    EVIDENCE_TABLES.mkdir(parents=True, exist_ok=True)

    selected_tables = [
        "phase1_baselines.csv",
        "phase2_expanding_window_cv.csv",
        "phase2_test_metrics.csv",
        "phase3_neural_metrics.csv",
        "phase3_chronos_metrics.csv",
        "phase4_error_by_hour.csv",
        "phase4_error_by_day_of_week.csv",
    ]
    for name in selected_tables:
        copy_if_exists(TABLES_DIR / name, EVIDENCE_TABLES / name)

    predictions_path = TABLES_DIR / "phase2_test_predictions.csv"
    if predictions_path.exists():
        predictions = pd.read_csv(predictions_path)
        predictions.head(250).to_csv(
            EVIDENCE_TABLES / "phase2_test_predictions_sample.csv",
            index=False,
        )

    summary_path = RESULTS_DIR / "final_project_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(
            "results/final_project_summary.json is missing. "
            "Run the experiment pipeline before building evidence."
        )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    copy_if_exists(summary_path, EVIDENCE_DIR / "final_project_summary.json")

    comparison_rows = []
    for section, family in [
        ("best_statistical", "statistical"),
        ("machine_learning", "machine_learning"),
        ("neural", "neural"),
        ("foundation_model", "foundation_model"),
    ]:
        row = summary.get(section)
        if isinstance(row, dict):
            comparison_rows.append(
                {
                    "family": family,
                    "model": row.get("model", section),
                    "mae": row.get("mae"),
                    "rmse": row.get("rmse"),
                    "mape": row.get("mape"),
                    "smape": row.get("smape"),
                    "predict_seconds": row.get("predict_seconds"),
                    "predict_peak_memory_mb": row.get("predict_peak_memory_mb"),
                }
            )

    if comparison_rows:
        pd.DataFrame(comparison_rows).to_csv(
            EVIDENCE_TABLES / "model_family_comparison.csv",
            index=False,
        )

    lines = [
        "# Real Experiment Results",
        "",
        "This evidence pack was generated from an executed forecasting workflow.",
        "Large datasets and full prediction artefacts remain outside Git; selected metrics and samples are retained here.",
        "",
    ]

    statistical = summary.get("best_statistical")
    if isinstance(statistical, dict):
        lines.extend(
            [
                "## Statistical baseline",
                "",
                f"- Best statistical model: **{statistical.get('model', 'n/a')}**",
                f"- MAE: **{fmt(statistical.get('mae'))}**",
                f"- RMSE: **{fmt(statistical.get('rmse'))}**",
                f"- MAPE: **{fmt(statistical.get('mape'))}**",
                f"- sMAPE: **{fmt(statistical.get('smape'))}**",
                "",
            ]
        )

    ml = summary.get("machine_learning")
    if isinstance(ml, dict):
        lines.extend(
            [
                "## Machine learning",
                "",
                f"- Reference ML model: **{ml.get('model', 'n/a')}**",
                f"- MAE: **{fmt(ml.get('mae'))}**",
                f"- RMSE: **{fmt(ml.get('rmse'))}**",
                f"- MAPE: **{fmt(ml.get('mape'))}**",
                f"- sMAPE: **{fmt(ml.get('smape'))}**",
                "",
            ]
        )

    neural = summary.get("neural")
    if isinstance(neural, dict):
        lines.extend(
            [
                "## Neural baseline",
                "",
                f"- Model: **{neural.get('model', 'n/a')}**",
                f"- MAE: **{fmt(neural.get('mae'))}**",
                f"- RMSE: **{fmt(neural.get('rmse'))}**",
                f"- Fit time: **{fmt(neural.get('fit_seconds'))} s**",
                f"- Prediction time: **{fmt(neural.get('predict_seconds'))} s**",
                "",
            ]
        )

    foundation = summary.get("foundation_model")
    if isinstance(foundation, dict):
        lines.extend(
            [
                "## Foundation model",
                "",
                f"- Model: **{foundation.get('model', 'n/a')}**",
                f"- MAE: **{fmt(foundation.get('mae'))}**",
                f"- RMSE: **{fmt(foundation.get('rmse'))}**",
                f"- Prediction time: **{fmt(foundation.get('predict_seconds'))} s**",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "## Foundation model",
                "",
                "- Chronos-2 was **not run in this execution**.",
                "- Re-run the workflow with **Run Chronos-2 foundation model** enabled to add foundation-model results.",
                "",
            ]
        )

    probabilistic = summary.get("probabilistic")
    if isinstance(probabilistic, dict):
        lines.extend(
            [
                "## Probabilistic evaluation",
                "",
                f"- 80% interval coverage: **{fmt(probabilistic.get('interval_coverage_80'))}**",
                f"- Mean 80% interval width: **{fmt(probabilistic.get('mean_interval_width_80'))}**",
                "",
            ]
        )

    robustness = summary.get("robustness")
    if isinstance(robustness, dict):
        lines.extend(
            [
                "## ML residual analysis",
                "",
                f"- Mean absolute error: **{fmt(robustness.get('mean_absolute_error'))}**",
                f"- Median absolute error: **{fmt(robustness.get('median_absolute_error'))}**",
                f"- Maximum absolute error: **{fmt(robustness.get('max_absolute_error'))}**",
                "",
            ]
        )

    lines.extend(
        [
            "## Evidence files",
            "",
            "- tables/model_family_comparison.csv",
            "- tables/phase1_baselines.csv",
            "- tables/phase2_expanding_window_cv.csv",
            "- tables/phase2_test_metrics.csv",
            "- tables/phase2_test_predictions_sample.csv",
            "- tables/phase3_neural_metrics.csv",
            "- tables/phase3_chronos_metrics.csv when Chronos is run",
            "- tables/phase4_error_by_hour.csv",
            "- tables/phase4_error_by_day_of_week.csv",
            "- final_project_summary.json",
            "",
            "Generated automatically by scripts/build_evidence_pack.py.",
        ]
    )

    (EVIDENCE_DIR / "RESULTS.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print(f"Evidence pack written to {EVIDENCE_DIR.resolve()}")


if __name__ == "__main__":
    main()
