from pathlib import Path

import pandas as pd

from time_series_foundation_models.probabilistic import (
    evaluate_quantile_frame,
    interval_coverage,
)
from time_series_foundation_models.robustness import residual_frame
from time_series_foundation_models.tracking import (
    append_run,
    create_run,
    read_runs,
)


def test_interval_coverage() -> None:
    coverage = interval_coverage(
        [2.0, 5.0, 9.0],
        [1.0, 4.0, 7.0],
        [3.0, 6.0, 8.0],
    )
    assert coverage == 2 / 3


def test_quantile_frame_evaluation() -> None:
    forecast = pd.DataFrame(
        {
            "0.1": [1.0, 4.0],
            "0.5": [2.0, 5.0],
            "0.9": [3.0, 6.0],
        }
    )
    result = evaluate_quantile_frame([2.0, 5.0], forecast)
    assert result["interval_coverage_80"] == 1.0
    assert result["mean_interval_width_80"] == 2.0


def test_residual_frame() -> None:
    frame = residual_frame(
        pd.date_range("2026-01-01", periods=2, freq="10min"),
        [10.0, 20.0],
        [8.0, 23.0],
    )
    assert frame["residual"].tolist() == [2.0, -3.0]
    assert frame["absolute_error"].tolist() == [2.0, 3.0]


def test_experiment_registry_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "runs.jsonl"
    run = create_run(
        experiment="unit-test",
        model="seasonal_naive",
        metrics={"mae": 1.2},
        params={"seasonal_period": 144},
    )
    append_run(run, path)
    loaded = read_runs(path)

    assert len(loaded) == 1
    assert loaded[0].run_id == run.run_id
