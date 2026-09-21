# Phase 2 — Machine-Learning Forecasting

## Objective

Phase 2 tests whether supervised machine-learning models can outperform the statistical baselines by converting the time series into a leakage-safe tabular forecasting problem.

## Feature engineering

The first ML benchmark intentionally uses **only historical Zone 1 demand plus calendar variables**.

Target-derived predictors include:

- 10-minute lag
- 1-hour lag
- 2-hour lag
- 3-hour lag
- 6-hour lag
- 12-hour lag
- 24-hour lag
- rolling means
- rolling standard deviations

All rolling features are calculated from a one-step-shifted target series.

This means the current target value cannot contribute to its own predictors.

## Why weather is initially excluded

The UCI dataset contains contemporaneous weather variables.

Using observed future weather directly would make the offline benchmark unrealistically optimistic unless an equivalent weather forecast were available at prediction time.

Therefore Phase 2 begins with historical-demand and deterministic calendar features only.

A later experiment may incorporate exogenous variables under an explicit "known in advance" or forecast-weather assumption.

## Models

The initial machine-learning comparison includes:

- Random Forest
- Extra Trees
- Histogram Gradient Boosting

These provide nonlinear tree-based alternatives without introducing external gradient-boosting dependencies.

## Temporal cross-validation

Model comparison uses scikit-learn's `TimeSeriesSplit`.

Each validation fold occurs strictly after its corresponding training window.

Conceptually:

```text
fold 1: [train] [validate]
fold 2: [------train------] [validate]
fold 3: [------------train------------] [validate]
```

The training window expands with time.

## Final test evaluation

Cross-validation is performed only on the development period.

The selected ML model family is then retrained on the full development history and evaluated on the later held-out test period.

The test set is not used to select the model.

## Metrics

The same metrics as Phase 1 are retained:

- MAE
- RMSE
- MAPE
- sMAPE

This makes statistical and ML results directly comparable.
