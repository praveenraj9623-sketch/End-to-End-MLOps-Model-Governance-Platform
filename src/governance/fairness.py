"""Fairness and ethics reporting for HR attrition predictions."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score

from src.features.engineering import TARGET_COLUMN, encode_target, load_raw_data, split_raw_data
from src.services.prediction_service import load_model_artifact, raw_feature_columns
from src.models.train import _positive_class_scores

FAIRNESS_REPORT_PATH = Path("reports/performance_logs/fairness_report.json")
MIN_GROUP_SIZE = 20
FAIRNESS_FEATURES = ["Gender", "AgeGroup", "Department", "JobRole", "MaritalStatus"]
HIGH_RISK_GAP_THRESHOLD = 0.15
RECALL_GAP_THRESHOLD = 0.20


def _age_group(age: Any) -> str:
    try:
        value = int(age)
    except Exception:
        return "Unknown"
    if value < 30:
        return "Under 30"
    if value < 40:
        return "30-39"
    if value < 50:
        return "40-49"
    return "50+"


def _group_metrics(group: pd.DataFrame, threshold: float) -> dict[str, Any]:
    y_true = group["actual"].astype(int)
    y_pred = (group["probability"] >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "sample_size": int(len(group)),
        "actual_attrition_rate": float(y_true.mean()) if len(group) else 0.0,
        "high_risk_rate": float(y_pred.mean()) if len(group) else 0.0,
        "average_probability": float(group["probability"].mean()) if len(group) else 0.0,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "false_positive_rate": float(fp / max(fp + tn, 1)),
        "false_negative_rate": float(fn / max(fn + tp, 1)),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
        "sample_warning": len(group) < MIN_GROUP_SIZE,
    }


def _analyze_column(scored: pd.DataFrame, column: str, threshold: float) -> dict[str, Any]:
    groups: dict[str, Any] = {}
    for value, group in scored.groupby(column, dropna=False):
        label = "Unknown" if pd.isna(value) else str(value)
        groups[label] = _group_metrics(group, threshold)

    high_risk_rates = [item["high_risk_rate"] for item in groups.values() if not item["sample_warning"]]
    recall_rates = [item["recall"] for item in groups.values() if not item["sample_warning"]]
    max_high_risk_rate_gap = float(max(high_risk_rates) - min(high_risk_rates)) if len(high_risk_rates) >= 2 else None
    max_recall_gap = float(max(recall_rates) - min(recall_rates)) if len(recall_rates) >= 2 else None
    sample_warning = any(item["sample_warning"] for item in groups.values())
    needs_review = (
        (max_high_risk_rate_gap is not None and max_high_risk_rate_gap >= HIGH_RISK_GAP_THRESHOLD)
        or (max_recall_gap is not None and max_recall_gap >= RECALL_GAP_THRESHOLD)
    )
    if len(high_risk_rates) < 2:
        review_status = "insufficient_sample"
    elif needs_review:
        review_status = "needs_review"
    elif sample_warning:
        review_status = "monitor_sample_size"
    else:
        review_status = "monitor"
    return {
        "feature": column,
        "groups": groups,
        "max_high_risk_rate_gap": max_high_risk_rate_gap,
        "max_recall_gap": max_recall_gap,
        "review_status": review_status,
        "sample_warning": sample_warning,
    }


def generate_fairness_report() -> dict[str, Any]:
    """Score the holdout split and save group fairness diagnostics."""
    raw_df = load_raw_data()
    artifact = load_model_artifact(train_if_missing=True)
    if artifact is None:
        raise FileNotFoundError("Model artifact is not available.")

    _, X_test, _, y_test = split_raw_data(raw_df)
    X_test = X_test.copy()
    X_test[TARGET_COLUMN] = y_test.astype(int).values
    X_test["AgeGroup"] = X_test["Age"].apply(_age_group) if "Age" in X_test else "Unknown"

    feature_columns = raw_feature_columns(artifact)
    probabilities = _positive_class_scores(artifact["model"], X_test[feature_columns])
    threshold = float(artifact.get("decision_threshold") or artifact.get("threshold") or 0.5)
    scored = X_test.copy()
    scored["probability"] = probabilities
    scored["actual"] = encode_target(scored[TARGET_COLUMN]) if scored[TARGET_COLUMN].dtype == object else scored[TARGET_COLUMN]

    fairness_features = [column for column in FAIRNESS_FEATURES if column in scored.columns]
    dimensions = [_analyze_column(scored, column, threshold) for column in fairness_features]
    flagged = [
        {
            "feature": item["feature"],
            "max_high_risk_rate_gap": item["max_high_risk_rate_gap"],
            "max_recall_gap": item["max_recall_gap"],
            "review_status": item["review_status"],
            "sample_warning": item["sample_warning"],
        }
        for item in dimensions
        if item["review_status"] in {"needs_review", "insufficient_sample", "monitor_sample_size"}
    ]
    flagged_summary = [
        {
            "feature": item["feature"],
            "max_high_risk_rate_gap": item["max_high_risk_rate_gap"],
            "max_recall_gap": item["max_recall_gap"],
            "review_status": item["review_status"],
            "sample_warning": item["sample_warning"],
        }
        for item in dimensions
    ]

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": artifact.get("model_name"),
        "model_registry_version": artifact.get("model_registry_version") or "local",
        "decision_threshold": threshold,
        "minimum_group_size": MIN_GROUP_SIZE,
        "high_risk_gap_threshold": HIGH_RISK_GAP_THRESHOLD,
        "recall_gap_threshold": RECALL_GAP_THRESHOLD,
        "fairness_features": FAIRNESS_FEATURES,
        "dimensions": dimensions,
        "flagged_dimensions": flagged,
        "flagged_dimensions_table": flagged_summary,
        "ethics_note": (
            "This is a public demo dataset. Fairness diagnostics are for governance review only "
            "and must not be used for automated employment decisions."
        ),
    }
    FAIRNESS_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    FAIRNESS_REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> dict[str, Any]:
    report = generate_fairness_report()
    print(f"Fairness report saved to {FAIRNESS_REPORT_PATH}")
    print(json.dumps({"flagged_dimensions": report["flagged_dimensions"]}, indent=2))
    return report


if __name__ == "__main__":
    main()
