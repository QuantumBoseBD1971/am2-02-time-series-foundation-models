# Real Experiment Results

This evidence pack was generated from an executed forecasting workflow.
Large datasets and full prediction artefacts remain outside Git; selected metrics and samples are retained here.

## Statistical baseline

- Best statistical model: **seasonal_naive**
- MAE: **1971.4718**
- RMSE: **2574.2137**
- MAPE: **6.5587**
- sMAPE: **6.9029**

## Machine learning

- Reference ML model: **extra_trees**
- MAE: **223.9046**
- RMSE: **330.0266**
- MAPE: **0.7866**
- sMAPE: **0.7842**

## Neural baseline

- Model: **mlp_128_64_32**
- MAE: **272.7708**
- RMSE: **383.6210**
- Fit time: **42.3029 s**
- Prediction time: **0.0179 s**

## Foundation model

- Chronos-2 was **not run in this execution**.
- Re-run the workflow with **Run Chronos-2 foundation model** enabled to add foundation-model results.

## ML residual analysis

- Mean absolute error: **223.9046**
- Median absolute error: **157.7842**
- Maximum absolute error: **5348.5742**

## Evidence files

- tables/model_family_comparison.csv
- tables/phase1_baselines.csv
- tables/phase2_expanding_window_cv.csv
- tables/phase2_test_metrics.csv
- tables/phase2_test_predictions_sample.csv
- tables/phase3_neural_metrics.csv
- tables/phase3_chronos_metrics.csv when Chronos is run
- tables/phase4_error_by_hour.csv
- tables/phase4_error_by_day_of_week.csv
- final_project_summary.json

Generated automatically by scripts/build_evidence_pack.py.
