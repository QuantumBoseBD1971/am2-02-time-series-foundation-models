# AM2-02 — Time-Series Foundation Model Benchmark

A comparative forecasting project using the **UCI Power Consumption of Tetouan City** dataset.

The project compares:

**statistical baselines → machine learning → neural forecasting → pretrained time-series foundation models**

The central constraint is temporal integrity: future observations must never leak into model development.

## Model families

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
- Amazon Chronos-2 zero-shot forecasting through optional dependencies

## Evaluation

The repository covers:

- MAE
- RMSE
- MAPE
- sMAPE
- runtime
- approximate peak Python memory
- probabilistic interval coverage
- mean interval width
- residual/error analysis by hour and day
- temporal validation and leakage controls

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e ".[dev]"

python scripts/download_data.py
python scripts/run_baselines.py
python scripts/run_ml_forecasting.py
python scripts/run_neural_forecasting.py
python scripts/finalise_project.py
pytest
```

Optional foundation-model experiment:

```bash
pip install -e ".[foundation]"
python scripts/run_chronos_forecasting.py
python scripts/finalise_project.py
```

## Documentation

- `docs/dataset.md`
- `docs/methodology.md`
- `docs/phase2_ml_forecasting.md`
- `docs/phase3_deep_foundation_models.md`
- `docs/model_card.md`
- `docs/deployment_mlops.md`
- `docs/final_summary.md`
- `docs/final_reflection.md`
- `docs/am2_evidence.md`

## Development status

- **Phase 1 — complete:** statistical baselines and temporal split.
- **Phase 2 — complete:** ML features, tree models and expanding-window validation.
- **Phase 3 — complete:** neural and Chronos-2 zero-shot forecasting.
- **Phase 4 — complete:** probabilistic evaluation, robustness/error analysis, experiment tracking, MLOps and AM2 evidence synthesis.

## Responsible use

This is an educational benchmark. Historical public-data performance is not sufficient evidence for live grid-operation, trading or infrastructure decisions.

## Licence

Code: MIT. Dataset: UCI source licence/citation requirements apply.
