# End-to-End MLOps & Model Governance Platform

A production-style **MLOps and model governance platform** for employee attrition risk prediction using the IBM HR Attrition dataset.

This project demonstrates how a machine learning model can move beyond notebook experimentation into a governed ML workflow with leakage-safe preprocessing, candidate model comparison, threshold tuning, business-cost-aware champion selection, calibration, explainability, drift monitoring, fairness review, audit logging, FastAPI serving, and a Streamlit governance dashboard.

---

## Live Demo

**Streamlit Dashboard:**
https://end-to-end-mlops-model-governance-platform-et3unrnhvmpv6xscxql.streamlit.app/

**GitHub Repository:**
https://github.com/praveenraj9623-sketch/End-to-End-MLOps-Model-Governance-Platform

---

## Project Summary

Most beginner machine learning projects stop after model training and accuracy reporting. This project is designed to show a more realistic **end-to-end ML governance workflow**, where the model is evaluated not only by standard ML metrics but also by:

* Business cost
* Decision threshold policy
* Calibration quality
* Drift monitoring
* Fairness diagnostics
* Explainability
* Privacy-aware audit logging
* API readiness
* Local model registry fallback

The goal is not just to build an attrition model, but to demonstrate how a model can be monitored, explained, audited, and governed before being used as decision support.

---

## Problem Statement

Employee attrition is a major HR and business problem. Organizations want to identify employees who may be at risk of leaving so that HR teams can take timely retention actions.

However, attrition prediction is sensitive because it involves people-related decisions. A model should not be used blindly for automated employment decisions. It should be used as a **decision-support system** with proper governance, fairness review, and human oversight.

This project builds a governed attrition risk platform that predicts employee attrition risk and provides supporting governance artifacts for responsible review.

---

## Dataset

This project uses the IBM HR Analytics Employee Attrition dataset.

The dataset contains employee-level HR attributes such as:

* Age
* Department
* Job Role
* Monthly Income
* Overtime
* Years at Company
* Job Satisfaction
* Work Life Balance
* Business Travel
* Attrition label

The target variable is:

```text
Attrition: Yes / No
```

Dataset size used in the demo:

```text
1,470 rows
```

---

## Key Features

### 1. Leakage-Safe ML Pipeline

The project performs train/test splitting before preprocessing to avoid data leakage.

It excludes columns that should not be used as predictive features:

```text
EmployeeNumber
EmployeeCount
StandardHours
Over18
```

The pipeline uses model-specific preprocessing with `ColumnTransformer`, including:

* Numeric imputation
* Categorical imputation
* One-hot encoding
* Scaling where required
* Model-specific training pipelines

---

### 2. Candidate Model Training

The system trains and compares multiple candidate models:

* Logistic Regression
* XGBoost
* LightGBM

Each candidate model is evaluated using classification metrics and business-cost-aware threshold analysis.

---

### 3. Business-Cost-Aware Champion Selection

Instead of selecting the model only by accuracy, the platform selects the champion using a governance-driven policy:

```text
Lowest business cost first,
then PR-AUC,
then recall.
```

This is important because in attrition prediction, false negatives can be more expensive than false positives.

A false negative means the model misses an employee who may actually leave. A false positive means HR may spend effort reviewing an employee who may not leave.

The dashboard clearly separates:

* Active production threshold
* Best F1 threshold
* Best recall threshold
* Best business-cost threshold
* Candidate comparison threshold

---

### 4. Threshold Tuning

The project evaluates model behavior across different decision thresholds.

The dashboard shows:

* Precision
* Recall
* F1
* Business cost
* Active production confusion matrix

The active production threshold is used consistently for:

* API predictions
* Risk labels
* Audit logs
* Active business cost
* Active confusion matrix

---

### 5. Model Calibration

The champion model is calibrated when calibration improves Brier score.

Calibration helps check whether predicted probabilities are meaningful and reliable.

The dashboard includes a calibration curve to compare:

* Model predicted probability
* Observed positive rate
* Perfect calibration reference line

---

### 6. Model Registry and Model Card

The project includes a local model registry fallback when MLflow is unavailable or incompatible.

The model registry section displays:

* Model name
* Registry mode
* Data version
* Training date
* Champion model type
* Business cost
* AUC
* PR-AUC
* Decision policy

The model card documents:

* Intended use
* Not-for-use cases
* Limitations
* Leakage control
* Threshold policy
* Calibration status
* Fairness review
* Governance notes

---

### 7. Drift Monitoring

The Drift Monitor tab allows simulation of production drift scenarios such as:

* Baseline
* Compensation shift
* Workload pressure
* Hiring mix changes

The drift module calculates feature-level drift scores and displays:

* Overall drift score
* Maximum feature drift score
* Number of features above threshold
* Drift status

The system flags the model for review when feature drift exceeds the configured threshold.

---

### 8. Explainability

The Explainability tab provides employee-level explanation for predictions.

For a selected employee, the dashboard shows:

* Attrition probability
* Risk level
* Decision threshold
* Recommended HR action
* Local explanation drivers
* Global feature importance

The local explanation table uses model-margin contribution values, not direct probability-point changes. This prevents overclaiming and keeps the explanation responsible.

Example wording used in the dashboard:

```text
MonthlyRate is associated with a higher model risk score for this employee.
```

The system avoids causal claims such as:

```text
MonthlyRate causes attrition.
```

---

### 9. Fairness and Ethics Review

The Fairness & Ethics tab provides group-level fairness diagnostics across dimensions such as:

* Gender
* Age Group
* Department
* Job Role
* Marital Status

The dashboard reports:

* High-risk rate gap
* Recall gap
* Sample-size warnings
* Review status

Governance recommendation:

```text
Use this model only as HR decision support.
Do not use it for automated employment decisions.
Review dimensions marked needs_review before production rollout.
```

---

### 10. Prediction Audit Trail

The platform includes privacy-aware prediction audit logging.

Audit logs contain:

* Prediction timestamp
* Attrition probability
* Risk level
* Recommended HR action
* Masked request payload
* Response payload
* Audit ID
* Employee identifier or hash
* Model name
* Model registry version
* Decision threshold

Sensitive fields are masked by default.

The Audit Trail tab supports:

* Risk-level filtering
* Date filtering
* Row limit control
* Downloadable audit CSV
* Full row detail inspection

---

## Dashboard Tabs

### Executive Summary

Shows the current champion model, production threshold, key metrics, governance status, operational risk distribution, dataset size, and interview-friendly project positioning.

### Model Performance

Shows model ranking, candidate business cost, threshold tradeoffs, active production confusion matrix, and calibration curve.

### Model Registry

Displays model registry information, model-card metadata, decision policy, not-for-use cases, and limitations.

### Drift Monitor

Simulates drift scenarios and shows whether the model should remain monitored or require review.

### Explainability

Provides employee-level prediction explanation and global feature importance.

### Fairness & Ethics

Displays fairness diagnostics and responsible-use guidance.

### Audit Trail

Displays prediction audit records with privacy-aware payload handling.

---

## Current Demo Metrics

Representative champion model metrics from the current demo run:

| Metric               |    Value |
| -------------------- | -------: |
| AUC                  |    0.753 |
| PR-AUC               |    0.454 |
| Recall               |    0.745 |
| Precision            |    0.289 |
| F1 Score             |    0.417 |
| Active Business Cost | $309,000 |
| Active Threshold     |     0.10 |

These metrics are intentionally shown together with threshold policy and business cost because attrition prediction is an imbalanced classification problem where accuracy alone is not enough.

---

## Architecture

```text
Raw HR Dataset
      |
      v
Schema Validation
      |
      v
Train/Test Split Before Preprocessing
      |
      v
Model-Specific ColumnTransformer Pipelines
      |
      v
Candidate Model Training
      |
      v
Threshold Tuning + Business Cost Evaluation
      |
      v
Champion Selection
      |
      v
Calibration + Evaluation Artifacts
      |
      v
Model Card + Registry Fallback
      |
      v
FastAPI Prediction Service
      |
      v
Privacy-Aware Audit Logging
      |
      v
Streamlit Governance Dashboard
```

---

## Tech Stack

### Machine Learning

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* LightGBM
* ColumnTransformer pipelines
* Threshold tuning
* Calibration
* Business-cost evaluation

### MLOps and Governance

* MLflow local registry fallback
* Model card generation
* Drift monitoring
* Fairness diagnostics
* Audit logging
* Privacy-aware payload masking

### API and Dashboard

* FastAPI
* Streamlit
* Plotly
* Matplotlib
* Uvicorn

### Testing and Validation

* Pytest
* Compile checks
* Governance logic tests
* Risk logic tests
* Audit serialization tests
* Drift status tests

---

## Project Structure

```text
End-to-End-MLOps-Model-Governance-Platform/
│
├── app.py
├── requirements.txt
├── README.md
├── docker-compose.yml
├── Dockerfile
│
├── data/
│   └── raw/
│       └── hr_attrition.csv
│
├── models/
│   └── champion artifacts
│
├── reports/
│   ├── performance_logs/
│   ├── drift_reports/
│   ├── audit_logs/
│   └── fairness reports
│
├── scripts/
│   └── setup_demo.py
│
├── src/
│   ├── api/
│   │   └── main.py
│   │
│   ├── features/
│   │   └── engineering.py
│   │
│   ├── models/
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── register.py
│   │
│   ├── monitoring/
│   │   └── drift_detection.py
│   │
│   ├── governance/
│   │   └── fairness.py
│   │
│   ├── services/
│   │   ├── prediction_service.py
│   │   └── retraining_service.py
│   │
│   ├── storage/
│   └── dashboard/
│
└── tests/
    ├── test_api.py
    ├── test_data_validation.py
    ├── test_governance.py
    └── test_model.py
```

---

## Local Setup

Run these commands from Windows Command Prompt inside the project folder.

### 1. Create and activate virtual environment

```bash
python -m venv .venv
.\.venv\Scripts\activate.bat
```

### 2. Upgrade pip

```bash
python -m pip install --upgrade pip
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Build demo artifacts

```bash
python scripts\setup_demo.py
```

---

## Run Locally

Start FastAPI and Streamlit in two separate terminals.

### Terminal 1: Start FastAPI

```bash
.\.venv\Scripts\activate.bat
uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

FastAPI docs:

```text
http://localhost:8000/docs
```

### Terminal 2: Start Streamlit

```bash
.\.venv\Scripts\activate.bat
streamlit run app.py --server.port 8502
```

Streamlit dashboard:

```text
http://localhost:8502
```

---

## FastAPI Endpoints

| Method | Endpoint           | Description                                                    |
| ------ | ------------------ | -------------------------------------------------------------- |
| GET    | `/health`          | Returns model, threshold, drift, and metric health             |
| POST   | `/predict`         | Predicts attrition risk for one employee profile               |
| GET    | `/model-registry`  | Returns MLflow registry rows or local champion fallback        |
| GET    | `/drift-report`    | Returns latest drift score and retraining flag                 |
| GET    | `/audit-log`       | Returns recent privacy-aware prediction audit entries          |
| POST   | `/trigger-retrain` | Triggers Airflow if available, otherwise runs local retraining |

---

## Example Prediction Request

```json
{
  "Age": 41,
  "BusinessTravel": "Travel_Rarely",
  "DailyRate": 1102,
  "Department": "Sales",
  "DistanceFromHome": 1,
  "Education": 2,
  "EducationField": "Life Sciences",
  "EnvironmentSatisfaction": 2,
  "Gender": "Female",
  "HourlyRate": 94,
  "JobInvolvement": 3,
  "JobLevel": 2,
  "JobRole": "Sales Executive",
  "JobSatisfaction": 4,
  "MaritalStatus": "Single",
  "MonthlyIncome": 5993,
  "MonthlyRate": 19479,
  "NumCompaniesWorked": 8,
  "OverTime": "Yes",
  "PercentSalaryHike": 11,
  "PerformanceRating": 3,
  "RelationshipSatisfaction": 1,
  "StockOptionLevel": 0,
  "TotalWorkingYears": 8,
  "TrainingTimesLastYear": 0,
  "WorkLifeBalance": 1,
  "YearsAtCompany": 6,
  "YearsInCurrentRole": 4,
  "YearsSinceLastPromotion": 0,
  "YearsWithCurrManager": 5
}
```

---

## Docker Setup

The project also includes Docker support.

```bash
docker compose up --build
```

Docker services:

| Service   | URL                   |
| --------- | --------------------- |
| MLflow    | http://localhost:5000 |
| FastAPI   | http://localhost:8000 |
| Streamlit | http://localhost:8502 |

---

## Testing

Run the validation commands below:

```bash
python -m compileall src app.py tests
python -m pytest -q
```

The test suite validates core project behavior including:

* API health
* Data validation
* Model pipeline behavior
* Governance logic
* Active threshold handling
* Confusion matrix ordering
* Drift status logic
* Audit timestamp serialization
* Fairness and model-card helpers

---

## Governance Notes

This project is a portfolio-grade governance demo and should not be used directly for real HR decisions without additional validation.

Important responsible-use notes:

* The IBM HR Attrition dataset is small and public.
* Fairness results can be unstable for small groups.
* Predictions should support human HR review, not replace it.
* The model should not be used for automated termination, promotion, compensation, or disciplinary decisions.
* Production deployment would require stronger privacy, legal, fairness, monitoring, and security review.

---

## Why This Project Matters

This project demonstrates skills that are important for modern data science and machine learning roles:

* Building leakage-safe ML pipelines
* Training and comparing multiple candidate models
* Selecting a champion using business impact, not only accuracy
* Tuning thresholds for operational decision-making
* Explaining predictions responsibly
* Monitoring model drift
* Reviewing fairness metrics
* Creating model cards
* Serving predictions through an API
* Logging predictions for auditability
* Building a recruiter-friendly governance dashboard

It shows the difference between a simple ML notebook and a more complete MLOps workflow.

---

## Interview Positioning

This project can be explained in interviews as:

```text
I built an end-to-end MLOps and model governance platform for employee attrition prediction. 
The goal was not just to train a model, but to demonstrate how a model can be evaluated, governed, monitored, explained, audited, and served through an API.

I used leakage-safe preprocessing, trained multiple candidate models, selected the champion using business-cost-aware threshold tuning, calibrated the model, generated governance artifacts, created drift and fairness diagnostics, served predictions through FastAPI, and built a Streamlit dashboard for model governance review.
```

---

## Future Improvements

Possible future enhancements:

* Add CI/CD with GitHub Actions
* Add cloud deployment for FastAPI
* Add persistent database-backed audit logging
* Add authentication for dashboard access
* Add automated scheduled drift checks
* Add production-grade MLflow tracking server
* Add data versioning with DVC
* Add model monitoring alerts
* Add role-based access for HR, data science, and governance users

---

## Author

**Praveen Raj A**
Data Science / Machine Learning / MLOps Portfolio Project

GitHub: https://github.com/praveenraj9623-sketch
Live App: https://end-to-end-mlops-model-governance-platform-et3unrnhvmpv6xscxql.streamlit.app/
