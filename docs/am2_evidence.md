# AM2 Evidence Notes

This document evolves as the forecasting project develops.

## Problem framing

Power-demand prediction is formulated as a supervised time-series forecasting problem.

## Statistical forecasting

Phase 1 provides naive, seasonal-naive and Holt-Winters baselines.

## Machine-learning forecasting

Phase 2 introduces leakage-safe lag/rolling/calendar features and tree-based regressors under expanding-window validation.

## Deep learning

Phase 3 adds a compact multi-layer perceptron using the same supervised feature set. This isolates the effect of nonlinear neural representation from changes in the information available to the model.

## Foundation models

The project provides an executable Chronos-2 integration for zero-shot forecasting.

This demonstrates the distinction between:

- project-specific model fitting
- pretrained foundation-model inference

Chronos dependencies are isolated as an optional install so CI does not require large model downloads.

## Temporal leakage prevention

Controls include:

- chronological train/validation/test splits
- no random forecasting split
- past-only target lags
- shifted rolling features
- future test data excluded from training
- observed future weather excluded from the initial supervised benchmark

## Evaluation

All model families use a common point-forecast metric set:

- MAE
- RMSE
- MAPE
- sMAPE

Phase 3 additionally compares runtime and approximate memory requirements.

## Reproducibility

The repository includes packaged Python code, tests, CI, deterministic seeds, optional dependency groups and reproducible execution scripts.

## Evidence still to add

- probabilistic interval evaluation
- interval coverage/calibration
- robustness and error analysis
- experiment tracking
- model card
- deployment/MLOps discussion
- final reflection and evidence synthesis
