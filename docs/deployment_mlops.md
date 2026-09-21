# Deployment and MLOps Design

## Proposed architecture

A production forecasting system could separate:

1. source ingestion
2. temporal/data-quality validation
3. feature generation
4. batch or API forecasting
5. forecast/interval persistence
6. monitoring
7. model registry and controlled promotion

## Training-serving consistency

Lag and rolling features must be generated identically in training and inference.

Feature timestamps must enforce "as-of" semantics so no observation later than the forecast origin enters the predictor set.

## Model versioning

Every trained or downloaded model should retain immutable metadata:

- code commit
- data snapshot or time range
- dependency versions
- feature configuration
- model identifier
- hyperparameters
- forecast horizon
- metrics
- runtime/resource measurements

A production alias such as `champion` should point to the active version without deleting prior artefacts.

## Foundation-model registry

For Chronos-like models, versioning should include:

- upstream model id
- model revision/commit
- inference configuration
- context length
- quantiles requested
- hardware/runtime environment

## Monitoring

Relevant monitoring includes:

- feature and target drift
- forecast residuals
- MAE/RMSE over rolling windows
- interval coverage
- interval width
- missing timestamps
- inference latency
- memory usage
- model download/cache failures

## Retraining and re-evaluation

A conventional ML/neural model may be retrained on newer local history.

A zero-shot foundation model may instead be re-evaluated, updated to a newer upstream checkpoint, or adapted/fine-tuned.

Those are distinct lifecycle strategies and should be governed separately.

## Rollback

Rollback should update the deployment alias to a previously validated model/version rather than overwrite or delete historical artefacts.
