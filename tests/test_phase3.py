import pandas as pd

from time_series_foundation_models.foundation import make_chronos_frame
from time_series_foundation_models.neural import build_mlp_forecaster
from time_series_foundation_models.runtime import measure_call


def test_make_chronos_frame_uses_long_format() -> None:
    timestamps = pd.date_range("2026-01-01", periods=3, freq="10min")
    frame = make_chronos_frame(timestamps, [1.0, 2.0, 3.0])

    assert list(frame.columns) == ["item_id", "timestamp", "target"]
    assert frame["item_id"].nunique() == 1
    assert frame["target"].tolist() == [1.0, 2.0, 3.0]


def test_mlp_pipeline_contains_scaling_and_model() -> None:
    pipeline = build_mlp_forecaster()
    assert list(pipeline.named_steps) == ["scale", "model"]


def test_measure_call_returns_result_and_metrics() -> None:
    result, measurement = measure_call(lambda: 2 + 3)
    assert result == 5
    assert measurement.seconds >= 0
    assert measurement.peak_memory_mb >= 0
