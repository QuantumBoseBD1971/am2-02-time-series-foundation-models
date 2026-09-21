"""Optional Chronos-2 zero-shot forecasting integration."""

from __future__ import annotations

import pandas as pd

DEFAULT_CHRONOS_MODEL = "amazon/chronos-2"


def make_chronos_frame(
    timestamps,
    values,
    item_id: str = "zone_1",
) -> pd.DataFrame:
    """Convert one univariate series to Chronos long-format input."""
    return pd.DataFrame(
        {
            "item_id": item_id,
            "timestamp": pd.to_datetime(timestamps),
            "target": values,
        }
    )


def forecast_with_chronos2(
    history: pd.DataFrame,
    prediction_length: int,
    model_id: str = DEFAULT_CHRONOS_MODEL,
    device_map: str = "cpu",
) -> pd.DataFrame:
    """Run zero-shot Chronos-2 forecasting.

    Chronos is imported lazily so the core project and CI remain lightweight.
    Install the optional foundation-model dependencies with:

        pip install -e ".[foundation]"
    """
    try:
        from chronos import Chronos2Pipeline
    except ImportError as exc:
        raise ImportError(
            'Chronos support is optional. Install with: pip install -e ".[foundation]"'
        ) from exc

    if prediction_length <= 0:
        raise ValueError("prediction_length must be positive.")

    pipeline = Chronos2Pipeline.from_pretrained(
        model_id,
        device_map=device_map,
    )

    forecast = pipeline.predict_df(
        history,
        id_column="item_id",
        timestamp_column="timestamp",
        target="target",
        prediction_length=prediction_length,
        freq="10min",
    )
    return forecast
