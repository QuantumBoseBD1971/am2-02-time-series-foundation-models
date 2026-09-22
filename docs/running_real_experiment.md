# Running the real forecasting experiment

The normal CI workflow tests the package without downloading large foundation-model weights.

The manual **Run real forecasting experiment** workflow executes the actual benchmark.

## Standard run

With **Run Chronos-2 foundation model** disabled, the workflow runs:

1. dataset download
2. statistical baselines
3. machine-learning forecasting
4. expanding-window evaluation
5. neural forecasting
6. residual/error analysis
7. final evidence synthesis

This is the recommended first run because it is lightweight enough for a standard GitHub-hosted runner.

## Full foundation-model run

Enable **Run Chronos-2 foundation model** to additionally:

1. install the optional Chronos/PyTorch stack
2. download the pretrained model
3. execute zero-shot forecasting
4. retain probabilistic outputs and foundation-model metrics

This path is heavier and may take substantially longer on CPU.

## Evidence

The workflow uploads the full results directory as a GitHub Actions artifact for 90 days.

A smaller evidence directory is committed to Git when **Commit the generated evidence pack** is enabled.

The evidence pack includes model-family metrics, selected CV tables, a prediction sample, residual summaries and RESULTS.md.
