# End-to-End MLOps & Model Governance Platform

This project is a local MLOps governance demo for IBM HR Attrition. It trains candidate attrition models, selects a champion using business-cost-aware threshold tuning, serves predictions through FastAPI, and displays governance artifacts in Streamlit.

## What The Project Does

- Validates the raw HR dataset schema, missing values, ranges, and categories.
- Splits train/test before preprocessing to avoid leakage.
- Excludes `EmployeeNumber`, `EmployeeCount`, `StandardHours`, and `Over18` from model features.
- Uses model-specific `ColumnTransformer` pipelines for imputation, scaling, and one-hot encoding.
- Trains Logistic Regression, XGBoost, and LightGBM candidates.
- Selects the champion by lowest business cost, then PR-AUC, then recall.
- Tunes the operating threshold and stores the selected decision threshold in the model artifact.
- Calibrates the champion when calibration improves Brier score.
- Generates evaluation, drift, fairness, model-card, and audit artifacts.
- Serves predictions through FastAPI and writes privacy-aware audit logs.
- Provides a Streamlit dashboard for performance, registry, drift simulation, explainability, fairness, and audit review.

## Command Prompt Setup

Run these from Windows Command Prompt inside the project folder.

```cmd
.\.venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts\setup_demo.py
```

Start FastAPI and Streamlit in two separate Command Prompt windows:

```cmd
.\.venv\Scripts\activate.bat
uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

```cmd
.\.venv\Scripts\activate.bat
streamlit run app.py --server.port 8502
```

Open:

- Streamlit dashboard: `http://localhost:8502`
- FastAPI docs: `http://localhost:8000/docs`

## Streamlit Dashboard Tabs

`Executive Summary` shows champion metrics, governance status, baseline drift status, calibration status, and production-like risk distribution.

`Model Performance` shows candidate model ranking, business-cost comparison, threshold tradeoffs, calibration curve, and confusion matrix.

`Model Registry` shows MLflow registry rows when MLflow is available, or the local champion fallback when the local MLflow database has an Alembic mismatch. It also displays the generated model card.

`Drift Monitor` lets you simulate baseline, compensation shift, workload pressure, and hiring mix scenarios. It scores the current production sample, generates Evidently reports, calculates a bounded raw-feature drift score, and flags retraining when the score crosses the threshold.

`Explainability` lets you select an employee, run the same prediction service used by the API, inspect risk level and recommended HR action, and view the strongest local model drivers.

`Fairness & Ethics` generates and displays group-level fairness diagnostics for Gender, AgeGroup, Department, JobRole, and MaritalStatus. It flags high-risk-rate and recall gaps for human review.

`Audit Trail` displays recent prediction audit rows with hashed employee identifiers and masked sensitive fields by default.

## FastAPI Endpoints

- `GET /health`: model, threshold, drift, and metric health.
- `POST /predict`: attrition risk prediction for one employee profile.
- `GET /model-registry`: MLflow registry versions or local champion fallback.
- `GET /drift-report`: latest drift score and retraining flag.
- `GET /audit-log`: recent privacy-aware prediction audit entries.
- `POST /trigger-retrain`: triggers Airflow if installed, otherwise runs the local retraining pipeline.

## Important Files

- `app.py`: Streamlit governance dashboard.
- `src/api/main.py`: FastAPI routes.
- `src/services/prediction_service.py`: shared prediction, risk, explanation, and audit logic.
- `src/services/retraining_service.py`: shared retraining orchestration.
- `src/features/engineering.py`: leakage-safe preprocessing and raw train/test split.
- `src/models/train.py`: model-specific pipelines, threshold tuning, calibration, and champion selection.
- `src/models/evaluate.py`: threshold-aware evaluation and business metrics.
- `src/monitoring/drift_detection.py`: raw-feature drift simulation and Evidently reports.
- `src/governance/fairness.py`: fairness diagnostics.
- `src/models/register.py`: local/MLflow registry governance and model-card generation.
- `scripts/setup_demo.py`: one-command local artifact builder.

## Docker

```cmd
docker compose up --build
```

Docker services:

- `mlflow`: `http://localhost:5000`
- `api`: `http://localhost:8000`
- `streamlit`: `http://localhost:8502`

## Notes

The local MLflow SQLite database can become incompatible if package versions changed between runs. The project now catches that failure and falls back to local artifacts instead of crashing the API or dashboard.

Prediction audits mask sensitive fields by default. For a local-only demo, set `ALLOW_FULL_AUDIT_PAYLOAD=true` before starting the API or dashboard if you intentionally want full request payloads.

