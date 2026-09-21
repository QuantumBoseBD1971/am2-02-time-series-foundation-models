# Model Card

## Scope

This repository compares multiple forecasting model families:

- naive/statistical
- tree-based machine learning
- compact neural network
- pretrained time-series foundation model

It does not define one permanently superior model independent of deployment context.

## Intended use

Educational forecasting benchmark and AM2 portfolio evidence using historical power-consumption data.

## Out-of-scope use

The models must not be used directly for live grid-control, pricing, market-trading, safety or infrastructure decisions.

## Data

UCI Power Consumption of Tetouan City.

The dataset is historical and location-specific. External validity to other cities, seasons, grids or operating regimes is not established.

## Evaluation

Point forecasts are assessed with:

- MAE
- RMSE
- MAPE
- sMAPE

Additional evaluation includes:

- expanding-window validation
- runtime/resource comparison
- probabilistic interval coverage
- interval width
- error analysis by hour/day
- temporal leakage controls

## Foundation-model considerations

Chronos-2 is evaluated as a zero-shot pretrained model.

Its results should be interpreted differently from models fitted directly on this dataset because training cost has occurred upstream on external corpora.

## Risks and limitations

- historical distribution may not represent future demand
- extreme events may be underrepresented
- percentage metrics can be unstable near zero
- observed weather may differ from forecast weather available in production
- interval calibration may degrade under distribution shift
- foundation-model inference may require more compute and memory
- strong benchmark accuracy does not prove operational suitability

## Human oversight

A real deployment should present forecasts, intervals, model/version metadata and monitoring signals to domain experts rather than automate critical decisions without review.
