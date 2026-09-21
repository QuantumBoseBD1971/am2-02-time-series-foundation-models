# AM2 Evidence Notes

This document evolves as the forecasting project develops.

## Problem framing

Power-demand prediction is formulated as a supervised time-series forecasting problem.

## Data engineering

The repository demonstrates:

- reproducible UCI data acquisition
- timestamp parsing
- column normalisation
- chronological ordering checks
- deterministic temporal splitting

## Statistical forecasting

Phase 1 benchmarks:

- naive forecasting
- seasonal-naive forecasting
- Holt-Winters exponential smoothing

## Machine-learning forecasting

Phase 2 converts historical demand into supervised features using:

- lagged observations
- shifted rolling statistics
- calendar variables

Tree-based regressors are then benchmarked under expanding-window temporal validation.

## Leakage prevention

The project includes several explicit leakage controls:

- no random train/test splitting
- lag features use past observations only
- rolling features are shifted before aggregation
- the final test period remains later than the development period
- observed future weather is excluded from the initial ML feature set

## Evaluation

Forecasts are compared using:

- MAE
- RMSE
- MAPE
- sMAPE

The ML benchmark records performance across multiple temporal folds before final test evaluation.

## Reproducibility

The repository includes:

- packaged Python modules
- declared dependencies
- deterministic model seeds
- unit tests
- GitHub Actions CI
- reproducible scripts
- generated result tables

## Evidence still to add

- compact deep-learning forecast
- pretrained time-series foundation model
- zero-shot comparison
- runtime/resource comparison
- probabilistic intervals
- robustness and error analysis
- experiment tracking
- model card
- deployment/MLOps discussion
- final reflection
