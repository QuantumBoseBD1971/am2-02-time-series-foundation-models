# Dataset

## Source

**Power Consumption of Tetouan City**, UCI Machine Learning Repository.

- dataset id: 849
- DOI: `10.24432/C5B034`
- 52,417 observations
- 10-minute intervals
- no missing values reported by UCI

The dataset contains weather measurements and power consumption for three distribution zones in Tetouan, Morocco.

## Initial target

The first benchmark forecasts:

`zone_1_power_consumption`

The remaining zone targets are preserved for later multivariate and transfer-style experiments.

## Why this dataset

It is suitable for the AM2 forecasting project because it supports:

- real temporal ordering
- clear intraday seasonality
- multiple related target series
- exogenous weather variables
- classical, ML, deep-learning and foundation-model comparisons

## Leakage controls

The repository never shuffles time before splitting.

All baseline evaluation follows:

```text
past training data
        ↓
future validation window
        ↓
later untouched test window
```

Future target observations are not used to construct training features.

## Reproducibility

Run:

```bash
python scripts/download_data.py
```

The local raw CSV is written under `data/` and remains untracked.
