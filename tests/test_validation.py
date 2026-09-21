import pandas as pd
from sklearn.dummy import DummyRegressor

from time_series_foundation_models.validation import expanding_window_benchmark


def test_expanding_window_benchmark_preserves_fold_order() -> None:
    X = pd.DataFrame({"x": range(30)})
    y = pd.Series(range(30), dtype=float)

    result = expanding_window_benchmark(
        X,
        y,
        {"mean": DummyRegressor(strategy="mean")},
        n_splits=3,
    )

    assert result["fold"].nunique() == 3
    fold_sizes = result.groupby("fold")["train_rows"].first().tolist()
    assert fold_sizes == sorted(fold_sizes)
