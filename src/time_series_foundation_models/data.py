"""Data acquisition, cleaning and chronological splitting."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo

from time_series_foundation_models.config import (
    DATA_DIR,
    DATETIME_COLUMN,
    RAW_DATA_PATH,
    TEST_FRACTION,
    UCI_DATASET_ID,
    VALIDATION_FRACTION,
)


def _snake_case(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9]+", "_", name.strip())
    return name.strip("_").lower()


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [_snake_case(column) for column in out.columns]
    return out


def download_dataset(destination: Path = RAW_DATA_PATH) -> Path:
    """Download the UCI Tetouan power-consumption dataset."""
    dataset = fetch_ucirepo(id=UCI_DATASET_ID)

    features = dataset.data.features.copy()
    targets = dataset.data.targets.copy()
    df = pd.concat([features, targets], axis=1)
    df = normalise_columns(df)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(destination, index=False)
    return destination


def load_dataset(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load, parse and chronologically sort the dataset."""
    if not path.exists():
        download_dataset(path)

    df = normalise_columns(pd.read_csv(path))

    if DATETIME_COLUMN not in df.columns:
        raise KeyError(
            f"Expected timestamp column '{DATETIME_COLUMN}'. "
            f"Available columns: {sorted(df.columns)}"
        )

    df[DATETIME_COLUMN] = pd.to_datetime(df[DATETIME_COLUMN], dayfirst=False)
    df = df.sort_values(DATETIME_COLUMN).reset_index(drop=True)

    if df[DATETIME_COLUMN].duplicated().any():
        raise ValueError("Duplicate timestamps detected.")

    return df


def chronological_split(
    df: pd.DataFrame,
    validation_fraction: float = VALIDATION_FRACTION,
    test_fraction: float = TEST_FRACTION,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a time series without shuffling."""
    if validation_fraction <= 0 or test_fraction <= 0:
        raise ValueError("Validation and test fractions must be positive.")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("Validation + test fractions must be less than 1.")

    n_rows = len(df)
    train_end = int(n_rows * (1 - validation_fraction - test_fraction))
    validation_end = int(n_rows * (1 - test_fraction))

    train = df.iloc[:train_end].copy()
    validation = df.iloc[train_end:validation_end].copy()
    test = df.iloc[validation_end:].copy()

    return train, validation, test
