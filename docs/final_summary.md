# Final Project Summary

## Research question

How do conventional statistical, machine-learning, neural and pretrained time-series foundation models compare when evaluated on the same future forecasting task?

## Technical progression

The repository deliberately follows the evolution of forecasting methods:

```text
naive / seasonal naive
        ↓
Holt-Winters
        ↓
tree-based supervised ML
        ↓
compact neural forecasting
        ↓
Chronos-2 zero-shot foundation model
```

## Core engineering controls

The project emphasises:

- chronological validation
- expanding-window cross-validation
- past-only lag features
- shifted rolling statistics
- held-out final test period
- optional heavy foundation-model dependencies
- common forecast metrics
- runtime/resource comparison
- probabilistic interval evaluation
- residual/error analysis
- experiment tracking
- model/version governance

## Interpretation principle

The newest or most complex model is not automatically preferred.

A deployment decision should jointly consider:

- point-forecast accuracy
- uncertainty calibration
- robustness
- latency
- memory/compute cost
- training cost
- maintainability
- interpretability
- operational constraints

## Reproduction

Run the core experiments:

```bash
python scripts/download_data.py
python scripts/run_baselines.py
python scripts/run_ml_forecasting.py
python scripts/run_neural_forecasting.py
python scripts/finalise_project.py
pytest
```

Run the optional foundation-model experiment separately:

```bash
pip install -e ".[foundation]"
python scripts/run_chronos_forecasting.py
python scripts/finalise_project.py
```

Generated result artefacts remain under `results/` and are intentionally excluded from Git.
