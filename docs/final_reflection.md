# Final Reflection

## What I learned

This project demonstrates why forecasting should not be treated as ordinary tabular regression.

The most important methodological difference is temporal causality: a model may only use information that would have been available at the forecast origin.

That requirement affected data splitting, feature engineering, validation, weather-variable use and final evaluation.

## Comparing generations of models

The project intentionally moves through several generations of forecasting technology rather than assuming a modern foundation model is automatically superior.

Statistical baselines are cheap, transparent and difficult to justify omitting.

Tree-based ML adds nonlinear relationships through engineered historical features.

A neural model tests whether additional representational flexibility helps under the same information set.

Chronos-2 introduces a fundamentally different paradigm: knowledge learned from external time-series corpora is transferred to the local series without project-specific training.

## Production lesson

Forecast quality is only one production concern.

A real forecasting service must also manage:

- forecast horizons
- data freshness
- missing timestamps
- uncertainty intervals
- model versions
- inference resources
- drift
- retraining/update strategies
- rollback

## What I would improve with more time

I would extend the benchmark with:

- multiple forecast horizons
- all three Tetouan zones
- known-ahead weather forecasts
- probabilistic scoring rules such as pinball loss
- conformal interval calibration
- stronger deep-learning sequence architectures
- controlled Chronos fine-tuning
- hardware-specific latency benchmarks

## AM2 relevance

The project gives evidence across data engineering, statistical learning, supervised ML, neural networks, foundation models, validation, uncertainty, MLOps, reproducibility and responsible deployment.
