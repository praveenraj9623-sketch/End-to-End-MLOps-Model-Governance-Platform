"""Small dashboard governance helpers with testable behavior."""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

CONFUSION_KEYS = ("true_negatives", "false_positives", "false_negatives", "true_positives")


def standard_confusion_matrix(counts: dict[str, Any]) -> list[list[int]]:
    """Return [[TN, FP], [FN, TP]] for Actual No/Yes rows and Predicted No/Yes columns."""
    return [
        [int(counts.get("true_negatives", 0) or 0), int(counts.get("false_positives", 0) or 0)],
        [int(counts.get("false_negatives", 0) or 0), int(counts.get("true_positives", 0) or 0)],
    ]


def confusion_counts_match(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """Return True when two confusion-count dictionaries represent the same matrix."""
    return all(int(left.get(key, 0) or 0) == int(right.get(key, 0) or 0) for key in CONFUSION_KEYS)


def should_show_training_confusion_artifact(
    active_counts: dict[str, Any],
    training_counts: dict[str, Any] | None,
) -> bool:
    """Show the static image only when its counts differ from the active matrix."""
    if not training_counts:
        return False
    return not confusion_counts_match(active_counts, training_counts)


def active_threshold_metrics_for_display(artifact: dict[str, Any]) -> dict[str, Any]:
    """Return the artifact's active-threshold metrics before generic summary metrics."""
    metrics = artifact.get("metrics", {})
    active = metrics.get("active_threshold_metrics")
    if active:
        return dict(active)
    counts = metrics.get("active_confusion_matrix") or artifact.get("thresholds", {}).get("selected_confusion_matrix", {})
    return {
        **counts,
        "threshold": artifact.get("decision_threshold") or artifact.get("threshold"),
        "business_cost": metrics.get("business_cost"),
    }


def normalize_candidate_comparison_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Rename old ambiguous candidate columns to explicit diagnostic names."""
    if data.empty:
        return data
    frame = data.copy()
    if "candidate_selected_threshold" not in frame and "selected_threshold" in frame:
        frame = frame.rename(columns={"selected_threshold": "candidate_selected_threshold"})
    if "candidate_business_cost" not in frame and "business_cost" in frame:
        frame = frame.rename(columns={"business_cost": "candidate_business_cost"})
    return frame


def drift_status_summary(
    feature_scores: dict[str, Any] | None,
    threshold: float,
    *,
    near_zero: float = 1e-6,
) -> dict[str, Any]:
    """Summarize feature-level drift using the configured feature threshold."""
    feature_scores = feature_scores or {}
    numeric_scores = {
        feature: float(score)
        for feature, score in feature_scores.items()
        if score is not None and pd.notna(score)
    }
    max_feature = max(numeric_scores.values(), default=0.0)
    features_above = [feature for feature, score in numeric_scores.items() if score > float(threshold)]
    if features_above:
        status = "Review Required"
    elif max_feature > near_zero:
        status = "Monitor"
    else:
        status = "OK"
    return {
        "max_feature_drift_score": float(max_feature),
        "features_above_threshold": sorted(features_above),
        "features_above_threshold_count": len(features_above),
        "status": status,
    }


def _to_iso(value: Any) -> Any:
    if value is None or value == "":
        return ""
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return value


def serialize_audit_dataframe(data: pd.DataFrame) -> pd.DataFrame:
    """Convert timestamp-like audit values to ISO strings for display/export."""
    if data.empty:
        return data.copy()
    frame = data.copy()
    for column in frame.columns:
        if column == "timestamp_utc" or pd.api.types.is_datetime64_any_dtype(frame[column]):
            frame[column] = frame[column].apply(_to_iso)
    return frame


def shorten_payload_columns(data: pd.DataFrame, limit: int = 140) -> pd.DataFrame:
    """Shorten request/response payloads for table display only."""
    frame = data.copy()
    for column in ["request_payload", "response_payload"]:
        if column in frame.columns:
            frame[column] = frame[column].fillna("").astype(str).apply(
                lambda value: value if len(value) <= limit else f"{value[:limit]}..."
            )
    return frame


def _missing_text(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    text = str(value).strip()
    return text == "" or text.lower() in {"n/a", "na", "none", "null", "nan"}


def registry_mode_label(value: Any) -> str:
    """Return a portfolio-friendly registry label when MLflow registry data is absent."""
    if _missing_text(value):
        return "Local registry fallback"
    return str(value)


def _format_datetime_for_display(value: Any) -> str:
    if _missing_text(value):
        return "Not recorded"
    timestamp = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(timestamp):
        return str(value)
    return timestamp.strftime("%Y-%m-%d %H:%M UTC")


def model_card_metadata_rows(model_card: dict[str, Any] | None) -> tuple[list[dict[str, str]], dict[str, str]]:
    """Build non-truncated model-card metadata rows and advanced details."""
    card = model_card or {}
    data_version = card.get("data_version") or card.get("dataset_path")
    registry_value = card.get("model_registry_version") or card.get("registry_version") or card.get("mlflow_run_id")
    model_name = card.get("model_name") or card.get("registered_model_name") or "Attrition risk model"
    champion_type = card.get("champion_model_type") or card.get("model_type") or card.get("algorithm")

    rows = [
        {"metadata": "Model name", "value": str(model_name)},
        {"metadata": "Registry mode", "value": registry_mode_label(registry_value)},
        {
            "metadata": "Data version",
            "value": "Not recorded" if _missing_text(data_version) else Path(str(data_version)).name,
        },
        {
            "metadata": "Generated",
            "value": _format_datetime_for_display(card.get("generated_at_utc") or card.get("generated_at")),
        },
        {
            "metadata": "Training date",
            "value": _format_datetime_for_display(
                card.get("training_date_utc") or card.get("trained_at_utc") or card.get("last_trained_at")
            ),
        },
        {
            "metadata": "Champion model type",
            "value": "Not recorded" if _missing_text(champion_type) else str(champion_type),
        },
    ]
    details = {
        "full_data_version": "" if _missing_text(data_version) else str(data_version),
        "raw_registry_value": "" if _missing_text(registry_value) else str(registry_value),
    }
    return rows, details


def parse_json_payload(value: Any) -> Any:
    """Parse stored JSON payload strings for expandable audit row details."""
    if _missing_text(value):
        return {}
    if isinstance(value, (dict, list)):
        return value
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return value


def audit_row_for_detail(row: dict[str, Any] | pd.Series) -> dict[str, Any]:
    """Return one audit row with nested request/response payloads restored."""
    clean = row.to_dict() if isinstance(row, pd.Series) else dict(row)
    if "timestamp_utc" in clean:
        clean["timestamp_utc"] = _to_iso(clean.get("timestamp_utc"))
    for column in ["request_payload", "response_payload"]:
        if column in clean:
            clean[column] = parse_json_payload(clean.get(column))
    return clean


def confusion_counts_from_matrix(matrix: list[list[int]] | None) -> dict[str, int]:
    """Convert [[TN, FP], [FN, TP]] into named counts."""
    if not matrix or len(matrix) < 2 or len(matrix[0]) < 2 or len(matrix[1]) < 2:
        return {}
    return {
        "true_negatives": int(matrix[0][0]),
        "false_positives": int(matrix[0][1]),
        "false_negatives": int(matrix[1][0]),
        "true_positives": int(matrix[1][1]),
    }
