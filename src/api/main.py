"""FastAPI governance platform for HR attrition model operations."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

try:
    from pydantic import ConfigDict
except Exception:  # pragma: no cover - pydantic v1 compatibility.
    ConfigDict = None

from src.models.train import REGISTERED_MODEL_NAME
from src.services.governance_service import load_drift_status, registry_versions
from src.services.prediction_service import (
    AUDIT_LOG_PATH,
    predict_employee,
    read_audit_entries,
    recommended_action,
    risk_level,
    top_prediction_drivers,
)
from src.services.retraining_service import trigger_retrain as trigger_retrain_service

app = FastAPI(title="HR Attrition MLOps Governance API", version="2.0.0")


class PredictionRequest(BaseModel):
    """Flexible employee profile request."""

    employee_profile: dict[str, Any] | None = Field(default=None)

    if ConfigDict is not None:
        model_config = ConfigDict(extra="allow")
    else:
        class Config:
            extra = "allow"


def _request_to_profile(payload: PredictionRequest | dict[str, Any]) -> dict[str, Any]:
    if isinstance(payload, PredictionRequest):
        raw = payload.model_dump() if hasattr(payload, "model_dump") else payload.dict()
    else:
        raw = payload
    profile = raw.get("employee_profile") or {
        key: value for key, value in raw.items() if key != "employee_profile"
    }
    if not profile:
        raise HTTPException(status_code=400, detail="Employee profile payload is empty.")
    return profile


@app.get("/health")
def health_check() -> dict[str, Any]:
    """Return platform and champion model health."""
    from src.services.prediction_service import load_model_artifact

    artifact = load_model_artifact(train_if_missing=False)
    drift = load_drift_status()
    metrics = artifact.get("metrics", {}) if artifact else {}
    return {
        "status": "ok",
        "champion_model_version": artifact.get("model_registry_version") if artifact else None,
        "champion_model_type": artifact.get("model_name") if artifact else None,
        "auc": metrics.get("auc"),
        "pr_auc": metrics.get("pr_auc"),
        "business_cost": metrics.get("business_cost"),
        "decision_threshold": artifact.get("decision_threshold") if artifact else None,
        "last_retrain_date": artifact.get("trained_at_utc") if artifact else None,
        "drift_status": "alert" if drift.get("retraining_required") else "ok",
        "drift_score": drift.get("drift_score"),
    }


@app.post("/predict")
def predict(payload: PredictionRequest) -> dict[str, Any]:
    """Predict attrition risk for one employee profile."""
    profile = _request_to_profile(payload)
    try:
        return predict_employee(profile, write_audit=True)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/model-registry")
def model_registry() -> dict[str, Any]:
    """List MLflow model registry versions and local champion fallback."""
    return {"registered_model_name": REGISTERED_MODEL_NAME, "versions": registry_versions()}


@app.get("/drift-report")
def drift_report() -> dict[str, Any]:
    """Return latest drift score and alert status."""
    return load_drift_status()


@app.get("/audit-log")
def audit_log() -> dict[str, Any]:
    """Return the last 50 privacy-aware prediction audit entries."""
    return {"entries": read_audit_entries(limit=50)}


@app.post("/trigger-retrain")
def trigger_retrain() -> dict[str, Any]:
    """Trigger Airflow when available, otherwise run the local retraining pipeline."""
    return trigger_retrain_service()


# Backward-compatible aliases for older dashboard code while the app is refactored.
_risk_level = risk_level
_recommended_action = recommended_action
_top_prediction_drivers = top_prediction_drivers

