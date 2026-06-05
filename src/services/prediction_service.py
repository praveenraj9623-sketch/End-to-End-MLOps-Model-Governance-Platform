"""Shared prediction, risk scoring, explainability, and audit logging service."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.features.engineering import ID_COLUMN, TARGET_COLUMN
from src.models.train import MODEL_PATH, _positive_class_scores, train_models

try:
    from sqlalchemy import create_engine, text
except Exception:  # pragma: no cover - optional deployment dependency.
    create_engine = None
    text = None

AUDIT_LOG_PATH = Path("reports/performance_logs/prediction_audit_log.csv")
SENSITIVE_FIELDS = {
    "Age",
    "Gender",
    "MaritalStatus",
    "MonthlyIncome",
    "MonthlyRate",
    "DailyRate",
    "HourlyRate",
}


def load_model_artifact(train_if_missing: bool = True) -> dict[str, Any] | None:
    """Load the champion model artifact, optionally training it first."""
    model_path = Path(os.getenv("MODEL_PATH", str(MODEL_PATH)))
    if not model_path.exists():
        if not train_if_missing:
            return None
        train_models()
    artifact = joblib.load(model_path)
    if not isinstance(artifact, dict):
        return {"model": artifact, "model_name": artifact.__class__.__name__}
    return artifact


def raw_feature_columns(artifact: dict[str, Any]) -> list[str]:
    """Return the raw feature columns expected by the fitted model pipeline."""
    columns = artifact.get("raw_feature_columns")
    if columns:
        return list(columns)
    schema = artifact.get("feature_schema", {})
    if schema.get("raw_feature_columns"):
        return list(schema["raw_feature_columns"])
    return [column for column in artifact.get("feature_names", []) if column != TARGET_COLUMN]


def profile_to_frame(profile: dict[str, Any], artifact: dict[str, Any]) -> pd.DataFrame:
    """Convert a flexible request profile to one raw-feature dataframe row."""
    row = {feature: profile.get(feature, np.nan) for feature in raw_feature_columns(artifact)}
    return pd.DataFrame([row], columns=raw_feature_columns(artifact))


def predict_probability(profile: dict[str, Any], artifact: dict[str, Any] | None = None) -> float:
    """Predict calibrated attrition probability for one employee profile."""
    artifact = artifact or load_model_artifact(train_if_missing=True)
    if artifact is None:
        raise FileNotFoundError("Model artifact is not available.")
    frame = profile_to_frame(profile, artifact)
    return float(_positive_class_scores(artifact["model"], frame)[0])


def risk_level(probability: float, artifact: dict[str, Any] | None = None) -> str:
    """Map probability to Low, Medium, or High using the trained risk bands."""
    bands = (artifact or {}).get("risk_bands", {}) if artifact else {}
    low_threshold = float(bands.get("low_threshold", 0.35))
    high_threshold = float(bands.get("high_threshold", 0.65))
    if probability < low_threshold:
        return "Low"
    if probability < high_threshold:
        return "Medium"
    return "High"


def recommended_action(level: str) -> str:
    """Return an HR action recommendation for the predicted risk level."""
    actions = {
        "Low": "Continue standard engagement monitoring.",
        "Medium": "Schedule a manager check-in and review workload, growth, and compensation signals.",
        "High": "Prioritize retention outreach, career-path discussion, and compensation review.",
    }
    return actions.get(level, actions["Medium"])


def _unwrap_pipeline(artifact: dict[str, Any]) -> Any:
    return (
        artifact.get("uncalibrated_pipeline")
        or artifact.get("model_pipeline")
        or artifact.get("model")
    )


def _transformed_profile(artifact: dict[str, Any], frame: pd.DataFrame) -> tuple[pd.DataFrame | None, list[str]]:
    pipeline = _unwrap_pipeline(artifact)
    if not hasattr(pipeline, "named_steps"):
        return None, list(artifact.get("feature_names", frame.columns))
    preprocessor = pipeline.named_steps.get("preprocess") or pipeline.named_steps.get("preprocessor")
    if preprocessor is None:
        return None, list(artifact.get("feature_names", frame.columns))
    try:
        feature_names = list(preprocessor.get_feature_names_out())
    except Exception:
        feature_names = list(artifact.get("feature_names", []))
    transformed = preprocessor.transform(frame)
    return pd.DataFrame(transformed, columns=feature_names), feature_names


def _estimator_importance(artifact: dict[str, Any]) -> np.ndarray | None:
    pipeline = _unwrap_pipeline(artifact)
    estimator = None
    if hasattr(pipeline, "named_steps") and "model" in pipeline.named_steps:
        estimator = pipeline.named_steps["model"]
    elif hasattr(pipeline, "feature_importances_") or hasattr(pipeline, "coef_"):
        estimator = pipeline
    if estimator is None:
        return None
    if hasattr(estimator, "feature_importances_"):
        return np.asarray(estimator.feature_importances_, dtype=float)
    if hasattr(estimator, "coef_"):
        return np.asarray(estimator.coef_, dtype=float).ravel()
    return None


def _base_feature_name(transformed_feature: str, raw_columns: list[str]) -> str:
    for column in sorted(raw_columns, key=len, reverse=True):
        if transformed_feature == column or transformed_feature.startswith(f"{column}_"):
            return column
    return transformed_feature


def _json_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


def _plain_english_driver_meaning(
    feature: str,
    profile_value: Any,
    direction: str,
    transformed_feature: str,
) -> str:
    """Return a compact human-readable explanation for a local driver."""
    direction_text = "is associated with a higher" if direction == "increases risk" else "is associated with a lower"
    labels = {
        "OverTime": "Overtime status",
        "MonthlyIncome": "Monthly income",
        "MonthlyRate": "Monthly rate",
        "DailyRate": "Daily rate",
        "JobRole": "Job role",
        "Department": "Department",
        "NumCompaniesWorked": "Number of prior companies",
        "YearsAtCompany": "Tenure at company",
        "YearsInCurrentRole": "Time in current role",
        "JobSatisfaction": "Job satisfaction score",
        "EnvironmentSatisfaction": "Environment satisfaction score",
        "WorkLifeBalance": "Work-life balance score",
        "DistanceFromHome": "Distance from home",
        "Age": "Age",
    }
    label = labels.get(feature, feature.replace("_", " "))
    if transformed_feature != feature and "_" in transformed_feature:
        encoded_value = transformed_feature.replace(f"{feature}_", "", 1)
        return f"{label} being {encoded_value} {direction_text} model risk score for this employee."
    if profile_value is None:
        return f"{label} {direction_text} model risk score for this employee."
    return f"{label} value {profile_value} {direction_text} model risk score for this employee."


def top_prediction_drivers(
    profile: dict[str, Any],
    artifact: dict[str, Any] | None = None,
    top_n: int = 3,
) -> list[dict[str, Any]]:
    """Return the strongest local model drivers for a single prediction."""
    artifact = artifact or load_model_artifact(train_if_missing=True)
    if artifact is None:
        return []
    frame = profile_to_frame(profile, artifact)
    transformed, transformed_features = _transformed_profile(artifact, frame)
    importance = _estimator_importance(artifact)
    if transformed is None or importance is None or len(importance) != len(transformed_features):
        return []

    effects = importance * np.asarray(transformed.iloc[0], dtype=float)
    raw_columns = raw_feature_columns(artifact)
    ranked = sorted(
        zip(transformed_features, effects, importance, strict=False),
        key=lambda item: abs(float(item[1])),
        reverse=True,
    )[:top_n]
    drivers: list[dict[str, Any]] = []
    for transformed_feature, effect, raw_importance in ranked:
        base_feature = _base_feature_name(transformed_feature, raw_columns)
        profile_value = _json_value(frame.iloc[0].get(base_feature))
        direction = "increases risk" if float(effect) >= 0 else "decreases risk"
        drivers.append(
            {
                "feature": base_feature,
                "transformed_feature": transformed_feature,
                "impact": float(effect),
                "shap_impact": float(effect),
                "impact_score": float(effect),
                "importance": float(raw_importance),
                "direction": direction,
                "profile_value": profile_value,
                "display_value": profile_value,
                "plain_english_meaning": _plain_english_driver_meaning(
                    base_feature,
                    profile_value,
                    direction,
                    transformed_feature,
                ),
            }
        )
    return drivers


def _hash_identifier(value: Any) -> str | None:
    if value in (None, "") or pd.isna(value):
        return None
    salt = os.getenv("AUDIT_HASH_SALT", "local-demo-salt")
    return hashlib.sha256(f"{salt}:{value}".encode("utf-8")).hexdigest()[:16]


def mask_profile(profile: dict[str, Any]) -> dict[str, Any]:
    """Mask audit payload fields unless full local demo audit is explicitly enabled."""
    if os.getenv("ALLOW_FULL_AUDIT_PAYLOAD", "false").lower() == "true":
        return dict(profile)
    masked: dict[str, Any] = {}
    for key, value in profile.items():
        if key == ID_COLUMN:
            masked[f"{ID_COLUMN}_hash"] = _hash_identifier(value)
        elif key in SENSITIVE_FIELDS:
            masked[key] = "[masked]"
        elif key in {"Department", "JobRole", "OverTime", "BusinessTravel"}:
            masked[key] = value
    return masked


def _database_url() -> str | None:
    return os.getenv("DATABASE_URL") or os.getenv("POSTGRES_URL")


def write_audit_entry(
    profile: dict[str, Any],
    response: dict[str, Any],
    artifact: dict[str, Any],
) -> None:
    """Persist a privacy-aware prediction audit entry to DB or CSV fallback."""
    row = {
        "audit_id": str(uuid.uuid4()),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "employee_id": profile.get(ID_COLUMN),
        "employee_id_hash": _hash_identifier(profile.get(ID_COLUMN)),
        "model_name": artifact.get("model_name"),
        "model_registry_version": artifact.get("model_registry_version") or "local",
        "decision_threshold": response.get("decision_threshold"),
        "attrition_probability": response["attrition_probability"],
        "risk_level": response["risk_level"],
        "top_drivers": json.dumps(response.get("top_3_shap_drivers", [])),
        "recommended_hr_action": response["recommended_hr_action"],
        "request_payload": json.dumps(mask_profile(profile)),
        "response_payload": json.dumps(response),
    }
    database_url = _database_url()
    if database_url and create_engine is not None and text is not None:
        try:
            engine = create_engine(database_url)
            with engine.begin() as connection:
                connection.execute(
                    text(
                        """
                        CREATE TABLE IF NOT EXISTS prediction_audit (
                            audit_id TEXT PRIMARY KEY,
                            timestamp_utc TEXT,
                            employee_id TEXT,
                            employee_id_hash TEXT,
                            model_name TEXT,
                            model_registry_version TEXT,
                            decision_threshold FLOAT,
                            attrition_probability FLOAT,
                            risk_level TEXT,
                            top_drivers TEXT,
                            recommended_hr_action TEXT,
                            request_payload TEXT,
                            response_payload TEXT
                        )
                        """
                    )
                )
                connection.execute(
                    text(
                        """
                        INSERT INTO prediction_audit (
                            audit_id, timestamp_utc, employee_id, employee_id_hash, model_name,
                            model_registry_version, decision_threshold, attrition_probability,
                            risk_level, top_drivers, recommended_hr_action, request_payload,
                            response_payload
                        )
                        VALUES (
                            :audit_id, :timestamp_utc, :employee_id, :employee_id_hash, :model_name,
                            :model_registry_version, :decision_threshold, :attrition_probability,
                            :risk_level, :top_drivers, :recommended_hr_action, :request_payload,
                            :response_payload
                        )
                        """
                    ),
                    row,
                )
            return
        except Exception:
            pass

    AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    new_row = pd.DataFrame([row])
    if AUDIT_LOG_PATH.exists():
        existing = pd.read_csv(AUDIT_LOG_PATH)
        new_row = pd.concat([existing, new_row], ignore_index=True)
    new_row.to_csv(AUDIT_LOG_PATH, index=False)


def read_audit_entries(limit: int = 50) -> list[dict[str, Any]]:
    """Return recent prediction audit rows."""
    database_url = _database_url()
    if database_url and create_engine is not None and text is not None:
        try:
            engine = create_engine(database_url)
            with engine.begin() as connection:
                result = connection.execute(
                    text(
                        """
                        SELECT audit_id, timestamp_utc, employee_id_hash, model_name,
                               employee_id,
                               model_registry_version, decision_threshold, attrition_probability,
                               risk_level, top_drivers, recommended_hr_action, request_payload,
                               response_payload
                        FROM prediction_audit
                        ORDER BY timestamp_utc DESC
                        LIMIT :limit
                        """
                    ),
                    {"limit": limit},
                )
                return [dict(row._mapping) for row in result]
        except Exception:
            pass

    if not AUDIT_LOG_PATH.exists():
        return []
    return pd.read_csv(AUDIT_LOG_PATH).tail(limit).iloc[::-1].to_dict(orient="records")


def predict_employee(
    profile: dict[str, Any],
    *,
    write_audit: bool = True,
    artifact: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the full governance prediction workflow for one employee."""
    artifact = artifact or load_model_artifact(train_if_missing=True)
    if artifact is None:
        raise FileNotFoundError("Model artifact is not available.")
    probability = predict_probability(profile, artifact)
    level = risk_level(probability, artifact)
    response = {
        "attrition_probability": probability,
        "risk_level": level,
        "top_3_shap_drivers": top_prediction_drivers(profile, artifact, top_n=3),
        "recommended_hr_action": recommended_action(level),
        "decision_threshold": artifact.get("decision_threshold") or artifact.get("threshold"),
        "model_name": artifact.get("model_name"),
        "model_registry_version": artifact.get("model_registry_version") or "local",
    }
    if write_audit:
        write_audit_entry(profile, response, artifact)
    return response
