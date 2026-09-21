# AM2 Evidence Notes

This repository is a self-contained AM2 portfolio example covering the evolution from statistical forecasting to pretrained time-series foundation models.

## Problem framing

Power-demand prediction is formulated as a temporal forecasting problem where validation must respect time order.

## Data engineering

Evidence includes reproducible UCI acquisition, timestamp parsing, ordering checks, deterministic splits and packaged feature-generation code.

## Statistical forecasting

Phase 1 includes naive, seasonal-naive and Holt-Winters models.

## Machine learning

Phase 2 converts historical demand into leakage-safe lag, rolling and calendar features and benchmarks tree models with expanding-window validation.

## Neural networks

Phase 3 adds a compact multi-layer perceptron using the same information set as the tree models.

## Foundation models

The repository integrates Amazon Chronos-2 for zero-shot forecasting through an optional dependency path.

This provides direct evidence of the distinction between locally fitted models and pretrained foundation-model inference.

## Evaluation

Common point metrics:

- MAE
- RMSE
- MAPE
- sMAPE

Additional evidence:

- temporal cross-validation
- runtime
- peak memory
- probabilistic interval coverage
- interval width
- residual analysis by hour/day

## Leakage prevention

Controls include chronological splitting, past-only lags, shifted rolling features, untouched final test data and exclusion of observed future weather from the initial supervised benchmark.

## Experiment tracking

A lightweight JSONL registry records immutable run ids, timestamps, models, metrics, parameters and notes.

## MLOps

The deployment design covers training-serving consistency, model versioning, foundation-model revision metadata, monitoring, retraining/update strategies and rollback.

## Responsible use

Historical public-data performance does not establish operational suitability for live energy systems.

Domain review, drift monitoring, uncertainty monitoring and governed deployment would be required in a real system.

## Reflection

The key technical lesson is that forecasting architecture cannot be assessed on predictive accuracy alone. Temporal integrity, uncertainty, computational cost and model lifecycle all materially affect model suitability.
