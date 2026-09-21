# AM2-02 — Time-Series Foundation Model Benchmark

A comparative forecasting project using the **UCI Power Consumption of Tetouan City** dataset.

The project compares:

**statistical baselines → machine learning → neural forecasting → pretrained time-series foundation models**

The central constraint is temporal integrity: future observations must never leak into model development.

## Current model families

### Statistical
- naive
- seasonal naive
- Holt-Winters exponential smoothing

### Machine learning
- Random Forest
- Extra Trees
- Histogram Gradient Boosting
- leakage-safe lag/rolling/calendar features
- expanding-window validation

### Neural
- three-hidden-layer MLP regression baseline

### Foundation model
- Amazon Chronos-2 zero-shot forecasting via optional `chronos-forecasting` dependencies

Chronos-2 currently exposes a DataFrame-based `predict_df` API for zero-shot forecasting, including quantile outputs. The project keeps this dependency optional so normal CI remains lightweight.

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
python -m pip install --upgrade pip
pip install -e ".[dev]"

python scripts/download_data.py
python scripts/run_baselines.py
python scripts/run_ml_forecasting.py
python scripts/run_neural_forecasting.py
pytest
```

To run the foundation-model experiment:

```bash
pip install -e ".[foundation]"
python scripts/run_chronos_forecasting.py
```

## Evaluation

- MAE
- RMSE
- MAPE
- sMAPE
- runtime
- approximate peak Python memory

## Development status

- **Phase 1 — complete:** statistical baselines and temporal split.
- **Phase 2 — complete:** ML features, tree models and expanding-window validation.
- **Phase 3 — in progress:** neural and Chronos-2 zero-shot forecasting.
- **Phase 4 — planned:** probabilistic evaluation, robustness, experiment tracking, model card, MLOps and final AM2 synthesis.

## Responsible use

This is an educational benchmark. Historical public-data performance is not sufficient evidence for real grid-operation or energy-market decisions.

## Licence

Code: MIT. Dataset: UCI source licence/citation requirements apply.
