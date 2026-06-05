"""Model governance read helpers shared by the API and Streamlit app."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.models.register import MODEL_CARD_PATH
from src.models.train import REGISTERED_MODEL_NAME, _configure_mlflow
from src.monitoring.drift_detection import DRIFT_ALERT_PATH
from src.services.prediction_service import load_model_artifact

try:
    from mlflow.tracking import MlflowClient
except Exception:  # pragma: no cover - optional in local lightweight environments.
    MlflowClient = None


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def load_drift_status() -> dict[str, Any]:
    return load_json(
        DRIFT_ALERT_PATH,
        {"drift_score": None, "retraining_required": False, "status": "not_available"},
    )


def load_model_card() -> dict[str, Any]:
    return load_json(MODEL_CARD_PATH, {})


def registry_versions() -> list[dict[str, Any]]:
    """Return MLflow registry versions, or a local champion fallback."""
    versions: list[dict[str, Any]] = []
    if MlflowClient is not None and _configure_mlflow():
        client = MlflowClient()
        try:
            for version in client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'"):
                metrics: dict[str, Any] = {}
                training_date = None
                if version.run_id:
                    try:
                        run = client.get_run(version.run_id)
                        metrics = run.data.metrics
                        training_date = datetime.fromtimestamp(
                            run.info.start_time / 1000, tz=timezone.utc
                        ).isoformat()
                    except Exception:
                        pass
                versions.append(
                    {
                        "name": version.name,
                        "version": version.version,
                        "stage": version.current_stage,
                        "status": version.status,
                        "run_id": version.run_id,
                        "auc": metrics.get("auc"),
                        "pr_auc": metrics.get("pr_auc") or metrics.get("average_precision"),
                        "business_cost": metrics.get("business_cost"),
                        "training_date": training_date,
                    }
                )
        except Exception:
            versions = []
    if versions:
        return versions

    artifact = load_model_artifact(train_if_missing=False)
    if not artifact:
        return []
    metrics = artifact.get("metrics", {})
    return [
        {
            "name": REGISTERED_MODEL_NAME,
            "version": artifact.get("model_registry_version") or "local",
            "stage": "Local Champion",
            "status": "READY",
            "run_id": artifact.get("mlflow_run_id"),
            "auc": metrics.get("auc"),
            "pr_auc": metrics.get("pr_auc"),
            "business_cost": metrics.get("business_cost"),
            "training_date": artifact.get("trained_at_utc"),
        }
    ]

