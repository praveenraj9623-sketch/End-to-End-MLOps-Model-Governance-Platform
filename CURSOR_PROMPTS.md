# Cursor AI Build Prompts

Use these prompts one by one inside Cursor.

## Prompt 1 — Understand the project

Read the full folder structure and README. Inspect `data/raw/hr_attrition.csv`. Explain the dataset columns, target column, and the full implementation plan before writing code.

## Prompt 2 — Data ingestion and validation

Implement `src/data/ingestion.py` and `src/data/validation.py`. The ingestion module should load `data/raw/hr_attrition.csv`. The validation module should check missing values, duplicates, target column validity, data types, allowed ranges, and save a validation summary.

## Prompt 3 — Feature engineering

Implement `src/features/engineering.py`. Convert `Attrition` into binary target, split features and target, identify numeric and categorical columns, apply preprocessing using `ColumnTransformer`, scale numeric columns, one-hot encode categorical columns, and save processed features to `data/processed/attrition_features.csv`. Also create `data/reference/reference_dataset.csv` for drift detection.

## Prompt 4 — Model training with MLflow

Implement `src/models/train.py`. Train Logistic Regression, Random Forest, XGBoost, and LightGBM. Track parameters, metrics, confusion matrix, ROC-AUC, precision, recall, F1 score, and artifacts in MLflow. Save the best model to `models/best_model.joblib`.

## Prompt 5 — Evaluation and SHAP governance report

Implement `src/models/evaluate.py`. Load the best model, calculate metrics, generate confusion matrix, classification report, ROC curve, feature importance, and SHAP plots. Save outputs in `reports/shap_reports/` and `reports/performance_logs/`.

## Prompt 6 — Model registry

Implement `src/models/register.py`. Register the best model in MLflow Model Registry and add model metadata such as dataset version, metrics, approval status, and business use case.

## Prompt 7 — Drift monitoring

Implement `src/monitoring/drift_detection.py`. Compare `data/reference/reference_dataset.csv` with new incoming data using Evidently AI. Generate HTML drift reports in `reports/drift_reports/`.

## Prompt 8 — FastAPI model serving

Implement `src/api/main.py`. Create endpoints `/health`, `/predict`, `/model-info`, `/governance`, and `/drift-report`. Load `models/best_model.joblib` and return prediction probability and risk segment.

## Prompt 9 — Streamlit governance dashboard

Implement `app.py`. Build a dashboard showing model metrics, drift status, feature importance, prediction form, recent predictions, governance checklist, and retraining status using Streamlit and Plotly.

## Prompt 10 — Tests and CI/CD

Implement tests in `tests/`. Then complete `.github/workflows/ci-cd.yml` and `.github/workflows/model-retrain.yml` so the project runs tests and scheduled retraining.
