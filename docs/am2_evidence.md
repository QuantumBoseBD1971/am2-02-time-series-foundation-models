# AM2 Evidence Notes

This document will evolve as the forecasting project develops.

## Current evidence

### Problem framing
Power-demand prediction is formulated as a supervised time-series forecasting problem.

### Data engineering
The project demonstrates reproducible public-data acquisition, timestamp parsing, column standardisation and chronological validation.

### Statistical forecasting
The first benchmark includes naive, seasonal-naive and Holt-Winters exponential-smoothing models.

### Evaluation
Forecasts are evaluated using MAE, RMSE, MAPE and sMAPE.

### Leakage prevention
The repository explicitly avoids random train/test splitting and preserves temporal order.

### Reproducibility
Dependencies, tests, deterministic split logic and GitHub Actions CI are included from the first phase.

## Evidence still to add

- lag and rolling feature engineering
- tree-based machine-learning models
- expanding-window validation
- deep-learning forecast
- pretrained time-series foundation model
- zero-shot comparison
- probabilistic intervals
- robustness and error analysis
- experiment tracking
- model card
- deployment/MLOps discussion
- final reflection
