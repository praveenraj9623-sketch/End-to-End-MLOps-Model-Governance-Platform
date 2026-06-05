"""Build all local demo artifacts for the MLOps governance platform."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.ingestion import main as run_ingestion
from src.data.validation import main as run_validation
from src.features.engineering import TARGET_COLUMN, load_raw_data, main as run_feature_engineering
from src.governance.fairness import main as run_fairness
from src.models.evaluate import main as run_evaluation
from src.models.register import main as run_registration
from src.models.train import main as run_training
from src.monitoring.drift_detection import main as run_drift_detection, reset_current_data
from src.services.prediction_service import predict_employee


def main() -> dict:
    """Run the complete local setup workflow."""
    summary: dict[str, object] = {"project_root": str(PROJECT_ROOT), "steps": []}

    for name, function in [
        ("ingestion", run_ingestion),
        ("validation", run_validation),
        ("feature_engineering", run_feature_engineering),
        ("training", run_training),
        ("evaluation", run_evaluation),
    ]:
        print(f"Running {name}...")
        result = function()
        summary["steps"].append({"name": name, "status": "success"})
        summary[name] = result

    print("Resetting baseline current-production data...")
    reset_current_data()

    for name, function in [
        ("drift_detection", run_drift_detection),
        ("fairness", run_fairness),
        ("model_card", run_registration),
    ]:
        print(f"Running {name}...")
        result = function()
        summary["steps"].append({"name": name, "status": "success"})
        summary[name] = result

    print("Writing sample prediction audit rows...")
    raw = load_raw_data().drop(columns=[TARGET_COLUMN]).head(3)
    predictions = [predict_employee(row.to_dict(), write_audit=True) for _, row in raw.iterrows()]
    summary["sample_predictions"] = predictions

    output_path = PROJECT_ROOT / "reports/performance_logs/setup_demo_summary.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Setup summary saved to {output_path}")
    return summary


if __name__ == "__main__":
    main()

