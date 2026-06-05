"""Optional Airflow DAG for weekly HR attrition model retraining."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = PROJECT_ROOT / "reports/performance_logs/latest_performance_snapshot.json"
DRIFT_ALERT_PATH = PROJECT_ROOT / "reports/performance_logs/drift_alert.json"
FAIRNESS_REPORT_PATH = PROJECT_ROOT / "reports/performance_logs/fairness_report.json"
MODEL_CARD_PATH = PROJECT_ROOT / "reports/model_card.json"

try:
    from airflow import DAG
    from airflow.operators.bash import BashOperator
    from airflow.operators.python import PythonOperator
except ModuleNotFoundError:  # pragma: no cover - local demo does not install Airflow.
    DAG = None
    BashOperator = None
    PythonOperator = None


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _send_report() -> None:
    """Print final pipeline summary with metrics and governance status."""
    metrics = _read_json(METRICS_PATH)
    drift = _read_json(DRIFT_ALERT_PATH)
    fairness = _read_json(FAIRNESS_REPORT_PATH)
    model_card = _read_json(MODEL_CARD_PATH)
    summary = {
        "model_name": model_card.get("model_name"),
        "champion_model_type": model_card.get("champion_model_type"),
        "auc": metrics.get("metrics", {}).get("classification", {}).get("auc"),
        "pr_auc": metrics.get("metrics", {}).get("classification", {}).get("pr_auc"),
        "business_cost": model_card.get("performance", {}).get("business_cost"),
        "drift_score": drift.get("drift_score"),
        "retraining_required": drift.get("retraining_required"),
        "fairness_flags": fairness.get("flagged_dimensions", []),
        "promotion_result": model_card.get("promotion_result", {}),
    }
    print(json.dumps(summary, indent=2))


dag = None
if DAG is not None:
    default_args = {
        "owner": "mlops-governance",
        "depends_on_past": False,
        "retries": 1,
    }

    with DAG(
        dag_id="mlops_weekly_retraining",
        description="Weekly retraining pipeline for HR attrition governance.",
        default_args=default_args,
        schedule_interval="0 6 * * MON",
        start_date=datetime(2026, 1, 1),
        catchup=False,
        tags=["mlops", "governance", "attrition"],
    ) as dag:
        validate_data = BashOperator(
            task_id="validate_data",
            bash_command=f"cd '{PROJECT_ROOT}' && python -m src.data.validation",
        )
        engineer_features = BashOperator(
            task_id="engineer_features",
            bash_command=f"cd '{PROJECT_ROOT}' && python -m src.features.engineering",
        )
        train_model = BashOperator(
            task_id="train_model",
            bash_command=f"cd '{PROJECT_ROOT}' && python -m src.models.train",
        )
        evaluate_model = BashOperator(
            task_id="evaluate_model",
            bash_command=f"cd '{PROJECT_ROOT}' && python -m src.models.evaluate",
        )
        check_drift = BashOperator(
            task_id="check_drift",
            bash_command=f"cd '{PROJECT_ROOT}' && python -m src.monitoring.drift_detection",
        )
        fairness_report = BashOperator(
            task_id="fairness_report",
            bash_command=f"cd '{PROJECT_ROOT}' && python -m src.governance.fairness",
        )
        register_model = BashOperator(
            task_id="register_model",
            bash_command=f"cd '{PROJECT_ROOT}' && python -m src.models.register",
        )
        send_report = PythonOperator(task_id="send_report", python_callable=_send_report)

        (
            validate_data
            >> engineer_features
            >> train_model
            >> evaluate_model
            >> check_drift
            >> fairness_report
            >> register_model
            >> send_report
        )

