# AM2-02 — Time-Series Foundation Model Benchmark

A comparative forecasting project using the **UCI Power Consumption of Tetouan City** dataset.

The project compares the evolution of forecasting approaches:

**statistical baselines → machine learning → deep learning → pretrained time-series foundation models**

The central engineering constraint is temporal integrity: future observations must never leak into model development.

## Research question

> How do conventional statistical and machine-learning forecasting methods compare with modern pretrained time-series foundation models when evaluated on the same future forecasting horizon?

## Current benchmark

### Statistical layer
- naive
- seasonal naive
- Holt-Winters exponential smoothing

### Machine-learning layer
- lagged demand features
- shifted rolling statistics
- calendar features
- Random Forest
- Extra Trees
- Histogram Gradient Boosting
- expanding-window validation

Later phases add deep learning and zero-shot foundation-model forecasting.

## Dataset

UCI **Power Consumption of Tetouan City**:

- 52,417 observations
- 10-minute sampling frequency
- three power-consumption zones
- weather measurements
- DOI: `10.24432/C5B034`

The initial target is Zone 1 power consumption.

## Temporal evaluation

The repository never performs a random train/test split.

Development follows chronological train/validation/test windows, and ML model comparison uses expanding-window cross-validation.

## Quick start

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

python -m pip install --upgrade pip
pip install -e ".[dev]"

python scripts/download_data.py
python scripts/run_baselines.py
python scripts/run_ml_forecasting.py
pytest
```

## Metrics

- MAE
- RMSE
- MAPE
- sMAPE

## Development status

- **Phase 1 — complete:** statistical baselines, temporal split, tests and CI.
- **Phase 2 — in progress:** lag/rolling features, tree-based forecasting and expanding-window validation.
- **Phase 3 — planned:** deep learning and pretrained time-series foundation models.
- **Phase 4 — planned:** probabilistic evaluation, robustness, experiment tracking, MLOps and final AM2 evidence.

## Responsible use

This is an educational benchmark. Historical public-data results are not sufficient evidence for real grid-operation or energy-market decisions.

## Licence

Code: MIT. Dataset: UCI source licence/citation requirements apply.
