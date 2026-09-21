# Phase 3 — Deep Learning and Time-Series Foundation Models

## Objective

Phase 3 compares a compact neural-network forecaster with a pretrained time-series foundation model.

The aim is not to assume that newer models are better. The experiment asks whether pretrained forecasting representations can provide competitive **zero-shot** performance against models trained specifically on the Tetouan series.

## Neural baseline

The neural baseline is a multi-layer perceptron with three hidden layers:

```text
input features
   ↓
128 ReLU
   ↓
64 ReLU
   ↓
32 ReLU
   ↓
continuous forecast
```

It uses the same leakage-safe lag, rolling and calendar features as the Phase 2 ML benchmark.

This provides a controlled neural comparison without changing the forecasting information set.

## Foundation model

The optional foundation-model path uses **Amazon Chronos-2** through the `chronos-forecasting` package.

Chronos-2 supports zero-shot forecasting through a pandas DataFrame API. The repository converts the historical Zone 1 series to long format:

- `item_id`
- `timestamp`
- `target`

The model is then asked to forecast the held-out horizon without fitting on this project dataset.

## Dependency isolation

Chronos and PyTorch are deliberately optional dependencies.

The normal CI path does not download foundation-model weights or require a GPU. This keeps the repository fast and reproducible while still providing a real executable foundation-model benchmark.

Install the optional stack with:

```bash
pip install -e ".[foundation]"
```

## Comparison dimensions

The deep/foundation comparison records:

- MAE
- RMSE
- MAPE
- sMAPE
- wall-clock runtime
- approximate Python peak memory

For Chronos-2, probabilistic outputs are preserved and the median forecast is used as the point forecast where available.

## Interpretation

A zero-shot foundation model has a different learning setup from the supervised models:

- ML/neural models learn directly from this project's historical training data.
- Chronos-2 arrives pretrained on external time-series corpora and performs inference without project-specific fitting.

That distinction is central to the AM2 discussion: performance must be interpreted alongside training cost, inference cost, adaptability, transparency and deployment requirements.
