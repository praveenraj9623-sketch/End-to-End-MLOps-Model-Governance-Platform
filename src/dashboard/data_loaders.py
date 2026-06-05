"""Data loading helpers for the Streamlit dashboard."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.features.engineering import RAW_DATA_PATH
from src.governance.fairness import FAIRNESS_REPORT_PATH
from src.models.evaluate import LATEST_SNAPSHOT_PATH, METRICS_CSV_PATH
from src.models.register import MODEL_CARD_PATH
from src.models.train import CANDIDATE_COMPARISON_PATH, THRESHOLD_METRICS_PATH
from src.monitoring.drift_detection import DRIFT_ALERT_PATH, PRODUCTION_PREDICTIONS_PATH
from src.services.prediction_service import AUDIT_LOG_PATH, load_model_artifact


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def load_raw_data() -> pd.DataFrame:
    return read_csv(RAW_DATA_PATH)


def load_dashboard_artifacts() -> dict[str, Any]:
    """Load all optional dashboard artifacts without failing the app."""
    return {
        "artifact": load_model_artifact(train_if_missing=False),
        "raw_data": load_raw_data(),
        "candidate_comparison": read_csv(CANDIDATE_COMPARISON_PATH),
        "threshold_metrics": read_csv(THRESHOLD_METRICS_PATH),
        "evaluation_snapshot": read_json(LATEST_SNAPSHOT_PATH, {}),
        "model_metrics": read_csv(METRICS_CSV_PATH),
        "drift_alert": read_json(DRIFT_ALERT_PATH, {}),
        "current_predictions": read_csv(PRODUCTION_PREDICTIONS_PATH),
        "fairness_report": read_json(FAIRNESS_REPORT_PATH, {}),
        "model_card": read_json(MODEL_CARD_PATH, {}),
        "audit_log": read_csv(AUDIT_LOG_PATH),
    }

