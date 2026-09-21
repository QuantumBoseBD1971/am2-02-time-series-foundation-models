# Phase 1 Methodology

## Objective

Establish simple but credible statistical forecasting baselines before introducing machine-learning or foundation models.

A complex model should only be considered useful if it can outperform these reference methods on future data.

## Temporal validation

Random splitting is inappropriate for forecasting because it can expose the model to information from the future.

The initial split is therefore chronological:

- 70% training
- 15% validation
- 15% test

The validation set is reserved for later model development. Phase 1 statistical baselines are finally fitted using the combined train + validation history and evaluated on the untouched final test window.

## Baselines

### Naive
Forecast every future value as the latest observed value.

### Seasonal naive
Repeat the most recent 24-hour pattern.

At a ten-minute sampling rate:

[
24 \times 6 = 144
]

observations form one daily seasonal period.

### Holt-Winters exponential smoothing
Additive trend and additive daily seasonality provide a conventional statistical benchmark that can adapt level, trend and repeating seasonal structure.

## Metrics

### MAE
Average absolute forecast error.

### RMSE
Penalises larger errors more strongly than MAE.

### MAPE
Percentage error relative to actual demand.

### sMAPE
Symmetric percentage error that reduces some of the asymmetry of conventional MAPE.

No single metric is treated as sufficient on its own.
