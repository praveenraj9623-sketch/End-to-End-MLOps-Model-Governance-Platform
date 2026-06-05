"""Raw data drift detection and production-drift simulation."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.features.engineering import (
    EXCLUDED_FEATURE_COLUMNS,
    ID_COLUMN,
    TARGET_COLUMN,
    encode_target,
    load_raw_data,
    split_raw_data,
)
from src.models.train import MODEL_PATH, _positive_class_scores, train_models
from src.services.prediction_service import load_model_artifact, raw_feature_columns

DRIFT_REPORT_DIR = Path("reports/drift_reports")
PERFORMANCE_LOG_DIR = Path("reports/performance_logs")
PRODUCTION_PREDICTIONS_PATH = PERFORMANCE_LOG_DIR / "current_production_predictions.csv"
DRIFT_ALERT_PATH = PERFORMANCE_LOG_DIR / "drift_alert.json"
DRIFT_SCORE_THRESHOLD = 0.3
PREDICTION_COLUMNS = {"prediction_probability", "prediction", "scenario"}
EXCLUDED_DRIFT_COLUMNS = set(EXCLUDED_FEATURE_COLUMNS)


def drift_status_summary(
    feature_scores: dict[str, Any] | None,
    threshold: float,
    *,
    near_zero: float = 1e-6,
) -> dict[str, Any]:
    """Summarize drift status from feature-level scores and a feature threshold."""
    feature_scores = feature_scores or {}
    numeric_scores = {
        feature: float(score)
        for feature, score in feature_scores.items()
        if score is not None and pd.notna(score)
    }
    max_feature_score = max(numeric_scores.values(), default=0.0)
    features_above_threshold = [
        feature for feature, score in numeric_scores.items() if score > float(threshold)
    ]
    if features_above_threshold:
        status = "Review Required"
    elif max_feature_score > near_zero:
        status = "Monitor"
    else:
        status = "OK"
    return {
        "max_feature_drift_score": float(max_feature_score),
        "features_above_threshold": sorted(features_above_threshold),
        "features_above_threshold_count": len(features_above_threshold),
        "status": status,
    }


def _raw_reference_and_holdout() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    raw_df = load_raw_data()
    X_train, X_test, y_train, y_test = split_raw_data(raw_df)
    reference = X_train.copy()
    reference[TARGET_COLUMN] = y_train.astype(int).values
    current = X_test.copy()
    current[TARGET_COLUMN] = y_test.astype(int).values
    return reference, current, y_train, y_test


def _load_reference_data() -> pd.DataFrame:
    reference, _, _, _ = _raw_reference_and_holdout()
    return reference


def _score_current(current: pd.DataFrame, scenario: str = "baseline") -> pd.DataFrame:
    artifact = load_model_artifact(train_if_missing=True)
    if artifact is None:
        if not MODEL_PATH.exists():
            train_models()
        artifact = joblib.load(MODEL_PATH)
    feature_columns = raw_feature_columns(artifact)
    probabilities = _positive_class_scores(artifact["model"], current[feature_columns])
    threshold = float(artifact.get("decision_threshold") or artifact.get("threshold") or 0.5)
    scored = current.copy()
    if TARGET_COLUMN in scored.columns and scored[TARGET_COLUMN].dtype == object:
        scored[TARGET_COLUMN] = encode_target(scored[TARGET_COLUMN])
    scored["prediction_probability"] = probabilities
    scored["prediction"] = (probabilities >= threshold).astype(int)
    scored["scenario"] = scenario
    PERFORMANCE_LOG_DIR.mkdir(parents=True, exist_ok=True)
    scored.to_csv(PRODUCTION_PREDICTIONS_PATH, index=False)
    return scored


def _load_current_data() -> pd.DataFrame:
    artifact = load_model_artifact(train_if_missing=True)
    expected_features = set(raw_feature_columns(artifact or {}))
    categorical_features = set((artifact or {}).get("categorical_columns", []))
    if PRODUCTION_PREDICTIONS_PATH.exists():
        current = pd.read_csv(PRODUCTION_PREDICTIONS_PATH)
        has_expected_features = expected_features.issubset(set(current.columns))
        categorical_looks_raw = all(
            column not in current.columns or not pd.api.types.is_numeric_dtype(current[column])
            for column in categorical_features
        )
        if has_expected_features and categorical_looks_raw:
            return current
    _, holdout, _, _ = _raw_reference_and_holdout()
    return _score_current(holdout, scenario="baseline")


def simulate_drift(scenario: str = "compensation_shift", intensity: float = 0.25) -> pd.DataFrame:
    """Create a scored current-production sample with a named synthetic drift scenario."""
    _, current, _, _ = _raw_reference_and_holdout()
    simulated = current.copy()
    scenario = scenario or "baseline"
    intensity = float(np.clip(intensity, 0.0, 0.8))

    if scenario == "baseline":
        return _score_current(simulated, scenario=scenario)

    if scenario == "compensation_shift":
        if "MonthlyIncome" in simulated:
            simulated["MonthlyIncome"] = (simulated["MonthlyIncome"] * (1 - intensity)).clip(lower=1000).round()
        if "PercentSalaryHike" in simulated:
            simulated["PercentSalaryHike"] = (simulated["PercentSalaryHike"] * (1 - intensity / 2)).clip(lower=0).round()
        if "OverTime" in simulated:
            simulated.loc[simulated.sample(frac=min(0.6, intensity + 0.2), random_state=42).index, "OverTime"] = "Yes"
    elif scenario == "workload_pressure":
        if "OverTime" in simulated:
            simulated.loc[simulated.sample(frac=min(0.75, intensity + 0.25), random_state=43).index, "OverTime"] = "Yes"
        if "WorkLifeBalance" in simulated:
            simulated["WorkLifeBalance"] = (simulated["WorkLifeBalance"] - 1).clip(lower=1)
        if "DistanceFromHome" in simulated:
            simulated["DistanceFromHome"] = (simulated["DistanceFromHome"] * (1 + intensity)).round()
    elif scenario == "hiring_mix":
        if "Department" in simulated:
            simulated.loc[simulated.sample(frac=min(0.5, intensity + 0.15), random_state=44).index, "Department"] = "Sales"
        if "JobLevel" in simulated:
            simulated["JobLevel"] = (simulated["JobLevel"] - 1).clip(lower=1)
        if "TotalWorkingYears" in simulated:
            simulated["TotalWorkingYears"] = (simulated["TotalWorkingYears"] * (1 - intensity)).clip(lower=0).round()
    else:
        raise ValueError(f"Unknown drift scenario: {scenario}")

    return _score_current(simulated, scenario=scenario)


def reset_current_data() -> pd.DataFrame:
    """Reset the current production sample back to the unchanged holdout split."""
    if PRODUCTION_PREDICTIONS_PATH.exists():
        PRODUCTION_PREDICTIONS_PATH.unlink()
    _, holdout, _, _ = _raw_reference_and_holdout()
    return _score_current(holdout, scenario="baseline")


def _write_fallback_html(path: Path, title: str, payload: dict[str, Any]) -> None:
    path.write_text(
        "<html><head><title>{title}</title></head><body>"
        "<h1>{title}</h1><pre>{payload}</pre></body></html>".format(
            title=title, payload=json.dumps(payload, indent=2)
        ),
        encoding="utf-8",
    )


def _report_columns(reference: pd.DataFrame, current: pd.DataFrame) -> list[str]:
    common = [column for column in reference.columns if column in current.columns]
    return [
        column
        for column in common
        if column != ID_COLUMN
        and column not in PREDICTION_COLUMNS
        and column not in EXCLUDED_DRIFT_COLUMNS
    ]


def _run_evidently_reports(
    reference: pd.DataFrame, current: pd.DataFrame, timestamp: str
) -> dict[str, str | None]:
    DRIFT_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    data_report_path = DRIFT_REPORT_DIR / f"data_drift_report_{timestamp}.html"
    target_report_path = DRIFT_REPORT_DIR / f"target_drift_report_{timestamp}.html"

    common_columns = _report_columns(reference, current)
    reference_common = reference[common_columns]
    current_common = current[common_columns]

    try:
        from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
        from evidently.report import Report

        data_columns = [column for column in common_columns if column != TARGET_COLUMN]
        data_report = Report(metrics=[DataDriftPreset()])
        data_report.run(reference_data=reference_common[data_columns], current_data=current_common[data_columns])
        data_report.save_html(str(data_report_path))

        if TARGET_COLUMN in common_columns:
            target_report = Report(metrics=[TargetDriftPreset()])
            target_report.run(reference_data=reference_common, current_data=current_common)
            target_report.save_html(str(target_report_path))
        else:
            target_report_path = None
    except Exception as exc:  # pragma: no cover - Evidently API changes by version.
        payload = {"error": str(exc), "columns": common_columns}
        _write_fallback_html(data_report_path, "Data Drift Report Fallback", payload)
        if TARGET_COLUMN in common_columns:
            _write_fallback_html(target_report_path, "Target Drift Report Fallback", payload)
        else:
            target_report_path = None

    return {
        "data_drift_report": str(data_report_path),
        "target_drift_report": str(target_report_path) if target_report_path else None,
    }


def _numeric_drift(ref_series: pd.Series, cur_series: pd.Series) -> float | None:
    ref_num = pd.to_numeric(ref_series, errors="coerce")
    cur_num = pd.to_numeric(cur_series, errors="coerce")
    if ref_num.notna().sum() == 0 or cur_num.notna().sum() == 0:
        return None
    ref_std = float(ref_num.std(ddof=0)) or 1.0
    shift = abs(float(cur_num.mean()) - float(ref_num.mean())) / ref_std
    return float(min(shift / 3, 1.0))


def _categorical_drift(ref_series: pd.Series, cur_series: pd.Series) -> float:
    ref_dist = ref_series.astype(str).fillna("Unknown").value_counts(normalize=True)
    cur_dist = cur_series.astype(str).fillna("Unknown").value_counts(normalize=True)
    labels = sorted(set(ref_dist.index).union(cur_dist.index))
    total_variation = sum(abs(float(ref_dist.get(label, 0.0)) - float(cur_dist.get(label, 0.0))) for label in labels) / 2
    return float(min(total_variation, 1.0))


def calculate_drift_score(reference: pd.DataFrame, current: pd.DataFrame) -> dict[str, Any]:
    """Calculate bounded drift scores for numeric and categorical raw features."""
    common_columns = [
        column
        for column in _report_columns(reference, current)
        if column != TARGET_COLUMN
    ]
    feature_scores: dict[str, float] = {}
    for column in common_columns:
        numeric_score = _numeric_drift(reference[column], current[column])
        if numeric_score is None:
            feature_scores[column] = _categorical_drift(reference[column], current[column])
        else:
            feature_scores[column] = numeric_score

    drift_score = float(sum(feature_scores.values()) / max(len(feature_scores), 1))
    top_drifted_features = sorted(
        feature_scores.items(), key=lambda item: item[1], reverse=True
    )[:10]
    return {
        "drift_score": drift_score,
        "feature_drift_scores": feature_scores,
        "top_drifted_features": top_drifted_features,
    }


def write_drift_alert(
    drift_result: dict[str, Any], report_paths: dict[str, str | None], scenario: str
) -> dict[str, Any]:
    """Write retraining alert metadata when drift exceeds the configured threshold."""
    status_summary = drift_status_summary(
        drift_result["feature_drift_scores"],
        DRIFT_SCORE_THRESHOLD,
    )
    alert = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scenario": scenario,
        "drift_score": drift_result["drift_score"],
        "threshold": DRIFT_SCORE_THRESHOLD,
        "max_feature_drift_score": status_summary["max_feature_drift_score"],
        "features_above_threshold": status_summary["features_above_threshold"],
        "features_above_threshold_count": status_summary["features_above_threshold_count"],
        "status": status_summary["status"],
        "retraining_required": status_summary["status"] == "Review Required",
        "feature_drift_scores": drift_result["feature_drift_scores"],
        "top_drifted_features": drift_result["top_drifted_features"],
        "reports": report_paths,
    }
    PERFORMANCE_LOG_DIR.mkdir(parents=True, exist_ok=True)
    DRIFT_ALERT_PATH.write_text(json.dumps(alert, indent=2), encoding="utf-8")
    return alert


def main() -> dict[str, Any]:
    """Run drift detection and persist reports plus alert metadata."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    reference = _load_reference_data()
    current = _load_current_data()
    scenario = str(current["scenario"].iloc[0]) if "scenario" in current.columns and len(current) else "baseline"
    report_paths = _run_evidently_reports(reference, current, timestamp)
    drift_result = calculate_drift_score(reference, current)
    alert = write_drift_alert(drift_result, report_paths, scenario)
    print(f"Data drift report saved to {report_paths['data_drift_report']}")
    print(json.dumps({"drift_score": alert["drift_score"], "retraining_required": alert["retraining_required"]}, indent=2))
    return alert


if __name__ == "__main__":
    main()
