"""Tests for fairness reporting and audit governance schemas."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.features.engineering import ID_COLUMN
from src.governance.fairness import FAIRNESS_FEATURES, generate_fairness_report
from src.dashboard.governance_logic import (
    active_threshold_metrics_for_display,
    audit_row_for_detail,
    model_card_metadata_rows,
    normalize_candidate_comparison_columns,
    serialize_audit_dataframe,
    should_show_training_confusion_artifact,
    standard_confusion_matrix,
)
from src.monitoring.drift_detection import drift_status_summary
from src.services import prediction_service


def test_fairness_report_output_schema() -> None:
    report = generate_fairness_report()
    assert "dimensions" in report
    assert "flagged_dimensions_table" in report
    assert set(FAIRNESS_FEATURES).issubset(set(report["fairness_features"]))

    dimensions = {dimension["feature"]: dimension for dimension in report["dimensions"]}
    for feature in FAIRNESS_FEATURES:
        assert feature in dimensions
        dimension = dimensions[feature]
        assert "max_high_risk_rate_gap" in dimension
        assert "max_recall_gap" in dimension
        assert "review_status" in dimension
        assert "sample_warning" in dimension
        assert isinstance(dimension["groups"], dict)

    summary = pd.DataFrame(report["flagged_dimensions_table"])
    assert {"feature", "max_high_risk_rate_gap", "max_recall_gap", "review_status", "sample_warning"}.issubset(summary.columns)


def test_audit_entry_schema_includes_request_and_response_payload(monkeypatch) -> None:
    audit_path = Path("reports/performance_logs/test_prediction_audit_log.csv")
    if audit_path.exists():
        audit_path.unlink()
    monkeypatch.setattr(prediction_service, "AUDIT_LOG_PATH", audit_path)

    profile = {ID_COLUMN: 123, "Department": "Sales", "Gender": "Female"}
    response = {
        "attrition_probability": 0.42,
        "risk_level": "Medium",
        "top_3_shap_drivers": [],
        "recommended_hr_action": "Schedule a manager check-in.",
        "decision_threshold": 0.1,
    }
    artifact = {"model_name": "unit-test-model", "model_registry_version": "local"}
    prediction_service.write_audit_entry(profile, response, artifact)

    rows = pd.read_csv(audit_path)
    expected_columns = {
        "audit_id",
        "timestamp_utc",
        "employee_id",
        "employee_id_hash",
        "attrition_probability",
        "risk_level",
        "recommended_hr_action",
        "request_payload",
        "response_payload",
    }
    assert expected_columns.issubset(rows.columns)
    assert int(rows.loc[0, "employee_id"]) == 123
    assert json.loads(rows.loc[0, "response_payload"])["risk_level"] == "Medium"
    masked_request = json.loads(rows.loc[0, "request_payload"])
    assert masked_request["Gender"] == "[masked]"
    if audit_path.exists():
        audit_path.unlink()


def test_active_confusion_matrix_standard_ordering() -> None:
    counts = {
        "true_negatives": 11,
        "false_positives": 2,
        "false_negatives": 3,
        "true_positives": 7,
    }
    assert standard_confusion_matrix(counts) == [[11, 2], [3, 7]]


def test_active_threshold_metrics_are_preferred_for_dashboard_display() -> None:
    artifact = {
        "decision_threshold": 0.1,
        "metrics": {
            "business_cost": 999,
            "active_threshold_metrics": {
                "threshold": 0.1,
                "false_positives": 2,
                "false_negatives": 1,
                "business_cost": 18000,
            },
        },
    }
    display_metrics = active_threshold_metrics_for_display(artifact)
    assert display_metrics["threshold"] == 0.1
    assert display_metrics["business_cost"] == 18000


def test_candidate_threshold_columns_are_labeled_as_candidate_selected() -> None:
    frame = pd.DataFrame(
        {
            "model_name": ["xgboost"],
            "selected_threshold": [0.2],
            "business_cost": [18000],
        }
    )
    normalized = normalize_candidate_comparison_columns(frame)
    assert "candidate_selected_threshold" in normalized.columns
    assert "candidate_business_cost" in normalized.columns
    assert "selected_threshold" not in normalized.columns
    assert "business_cost" not in normalized.columns


def test_duplicate_training_confusion_image_is_hidden_when_counts_match() -> None:
    active_counts = {
        "true_negatives": 10,
        "false_positives": 1,
        "false_negatives": 2,
        "true_positives": 5,
    }
    assert should_show_training_confusion_artifact(active_counts, dict(active_counts)) is False
    different_counts = {**active_counts, "false_positives": 3}
    assert should_show_training_confusion_artifact(active_counts, different_counts) is True


def test_drift_status_review_required_when_max_feature_exceeds_threshold() -> None:
    summary = drift_status_summary({"Age": 0.05, "JobRole": 0.35}, threshold=0.3)
    assert summary["status"] == "Review Required"
    assert summary["features_above_threshold"] == ["JobRole"]
    assert summary["features_above_threshold_count"] == 1


def test_drift_status_monitor_when_feature_drift_is_nonzero_below_threshold() -> None:
    summary = drift_status_summary({"Age": 0.05, "JobRole": 0.12}, threshold=0.3)
    assert summary["status"] == "Monitor"
    assert summary["max_feature_drift_score"] == 0.12
    assert summary["features_above_threshold_count"] == 0


def test_model_card_metadata_uses_readable_registry_fallback_and_dataset_name() -> None:
    rows, details = model_card_metadata_rows(
        {
            "model_name": "AttritionPredictor",
            "registered_model_name": None,
            "data_version": "data/raw/hr_attrition.csv",
            "generated_at_utc": "2026-06-05T10:01:25.316078+00:00",
            "champion_model_type": "xgboost",
        }
    )
    metadata = {row["metadata"]: row["value"] for row in rows}
    assert metadata["Model name"] == "AttritionPredictor"
    assert metadata["Registry mode"] == "Local registry fallback"
    assert metadata["Data version"] == "hr_attrition.csv"
    assert metadata["Generated"] == "2026-06-05 10:01 UTC"
    assert metadata["Champion model type"] == "xgboost"
    assert details["full_data_version"] == "data/raw/hr_attrition.csv"


def test_audit_timestamps_serialize_to_iso_strings() -> None:
    frame = pd.DataFrame(
        {
            "timestamp_utc": [pd.Timestamp("2026-06-05T13:00:00Z")],
            "audit_id": ["a1"],
        }
    )
    serialized = serialize_audit_dataframe(frame)
    assert serialized.loc[0, "timestamp_utc"] == "2026-06-05T13:00:00+00:00"
    assert "Timestamp(" not in serialized.to_csv(index=False)


def test_audit_row_detail_parses_nested_payload_json() -> None:
    row = {
        "audit_id": "a1",
        "timestamp_utc": pd.Timestamp("2026-06-05T13:00:00Z"),
        "request_payload": '{"Age": 41, "Department": "Sales"}',
        "response_payload": '{"risk_level": "High", "recommended_hr_action": "Schedule retention review."}',
    }
    detail = audit_row_for_detail(row)
    assert detail["timestamp_utc"] == "2026-06-05T13:00:00+00:00"
    assert detail["request_payload"]["Department"] == "Sales"
    assert detail["response_payload"]["risk_level"] == "High"
