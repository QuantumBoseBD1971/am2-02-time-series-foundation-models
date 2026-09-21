# AM2-02 — Time-Series Foundation Model Benchmark

A comparative forecasting project using the **UCI Power Consumption of Tetouan City** dataset.

The project is designed to compare the evolution of forecasting approaches:

**naive/statistical baselines → machine learning → deep learning → time-series foundation models**

The key engineering principle is that validation is strictly temporal: future observations are never allowed to leak into training.

## Research question

> How do conventional statistical and machine-learning forecasting methods compare with modern pretrained time-series foundation models when evaluated on the same future forecasting horizon?

## Dataset

Source: UCI Machine Learning Repository — **Power Consumption of Tetouan City**.

- UCI dataset id: `849`
- 52,417 observations
- 10-minute sampling frequency
- weather variables plus three power-consumption targets
- no missing values reported by the source
- DOI: `10.24432/C5B034`

This project initially forecasts **Zone 1 power consumption**. Later phases can extend the benchmark to Zones 2 and 3 and to multivariate forecasting.

The raw dataset is downloaded reproducibly and is not committed to Git.

## Project phases

### Phase 1 — statistical baseline layer
- reproducible UCI data loader
- timestamp parsing and validation
- chronological train/validation/test split
- naive forecast
- seasonal-naive forecast
- exponential smoothing
- MAE, RMSE, MAPE and sMAPE
- CI and tests

### Phase 2 — machine-learning forecasting
- lag features
- rolling-window features
- calendar features
- tree-based regressors
- walk-forward / expanding-window validation

### Phase 3 — deep learning and foundation models
- compact neural forecasting baseline
- Chronos-style pretrained foundation model
- zero-shot forecasting comparison
- runtime and resource comparison

### Phase 4 — probabilistic and production evaluation
- interval coverage
- robustness
- experiment tracking
- model card
- deployment/MLOps design
- final AM2 evidence synthesis

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
pytest
```

## Evaluation metrics

Forecasts are evaluated with:

- MAE
- RMSE
- MAPE
- sMAPE

No random train/test split is used.

## Responsible use

This repository is an educational benchmark. Results from this historical public dataset should not be interpreted as evidence that a forecasting model is ready for grid-operation or energy-market decisions.

## Licence

Code: MIT. Dataset: UCI source licence/citation requirements apply.
