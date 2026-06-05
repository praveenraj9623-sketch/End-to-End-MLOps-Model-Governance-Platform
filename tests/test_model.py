"""Tests for trained HR attrition model artifacts."""

from __future__ import annotations

from pathlib import Path

import joblib
import pytest
from sklearn.metrics import roc_auc_score

from src.features.engineering import (
    DROP_CONSTANT_COLUMNS,
    ID_COLUMN,
    PROCESSED_PATH,
    TARGET_COLUMN,
    build_preprocessor,
    feature_schema,
    load_raw_data,
    main as engineer_main,
)
from src.models.train import (
    FN_COST,
    FP_COST,
    MODEL_PATH,
    _positive_class_scores,
    business_cost_from_counts,
    load_training_data,
    metrics_at_threshold,
    train_models,
)
from src.services.prediction_service import mask_profile, predict_probability, raw_feature_columns


@pytest.fixture(scope="session")
def model_artifact() -> dict:
    if not PROCESSED_PATH.exists():
        engineer_main()
    if not MODEL_PATH.exists():
        train_models()
    artifact = joblib.load(MODEL_PATH)
    if ID_COLUMN in artifact.get("raw_feature_columns", []) or ID_COLUMN in artifact.get("feature_names", []):
        train_models()
        artifact = joblib.load(MODEL_PATH)
    return artifact


def test_model_loads(model_artifact: dict) -> None:
    assert model_artifact["model"] is not None
    assert Path(MODEL_PATH).exists()


def test_employee_number_is_excluded_from_features(model_artifact: dict) -> None:
    assert ID_COLUMN not in model_artifact["raw_feature_columns"]
    assert ID_COLUMN not in model_artifact["feature_names"]
    assert ID_COLUMN in model_artifact["excluded_features"]
    for constant_column in DROP_CONSTANT_COLUMNS:
        assert constant_column not in model_artifact["raw_feature_columns"]
        assert constant_column not in model_artifact["feature_names"]
        assert constant_column in model_artifact["excluded_features"]


def test_prediction_service_returns_probability(model_artifact: dict) -> None:
    sample = load_raw_data().drop(columns=[TARGET_COLUMN]).iloc[0].to_dict()
    probability = predict_probability(sample, model_artifact)
    assert 0.0 <= probability <= 1.0


def test_unseen_categories_are_handled(model_artifact: dict) -> None:
    sample = load_raw_data().drop(columns=[TARGET_COLUMN]).iloc[0].to_dict()
    sample["BusinessTravel"] = "Travel_By_Submarine"
    sample["Department"] = "Future Department"
    probability = predict_probability(sample, model_artifact)
    assert 0.0 <= probability <= 1.0


def test_threshold_artifacts_present(model_artifact: dict) -> None:
    assert 0.0 < model_artifact["decision_threshold"] < 1.0
    assert model_artifact["metrics"]["business_cost"] >= 0
    assert model_artifact["thresholds"]["selected_threshold"] == model_artifact["decision_threshold"]
    assert model_artifact.get("threshold_policy", {}).get("active_production_threshold") == model_artifact["decision_threshold"]


def test_raw_feature_contract_matches_pipeline(model_artifact: dict) -> None:
    X_train, X_test, _, _, feature_columns = load_training_data()
    assert raw_feature_columns(model_artifact) == feature_columns
    probabilities = _positive_class_scores(model_artifact["model"], X_test.head(5))
    assert len(probabilities) == 5
    assert list(X_train.columns) == feature_columns


def test_model_above_baseline_auc(model_artifact: dict) -> None:
    _, X_test, _, y_test, _ = load_training_data()
    probabilities = _positive_class_scores(model_artifact["model"], X_test)
    auc = roc_auc_score(y_test, probabilities)
    assert auc > 0.70


def test_audit_masking_hides_sensitive_fields() -> None:
    masked = mask_profile(
        {
            ID_COLUMN: 123,
            "Gender": "Female",
            "MonthlyIncome": 10000,
            "Department": "Sales",
        }
    )
    assert masked[f"{ID_COLUMN}_hash"] != "123"
    assert masked["Gender"] == "[masked]"
    assert masked["MonthlyIncome"] == "[masked]"
    assert masked["Department"] == "Sales"


def test_business_cost_formula_multiple_thresholds() -> None:
    y_true = [0, 0, 1, 1]
    probabilities = [0.1, 0.8, 0.4, 0.9]

    low_threshold = metrics_at_threshold(y_true, probabilities, 0.3)
    assert low_threshold["false_positives"] == 1
    assert low_threshold["false_negatives"] == 0
    assert low_threshold["business_cost"] == FP_COST

    high_threshold = metrics_at_threshold(y_true, probabilities, 0.7)
    assert high_threshold["false_positives"] == 1
    assert high_threshold["false_negatives"] == 1
    assert high_threshold["business_cost"] == FP_COST + FN_COST
    assert business_cost_from_counts(2, 3) == 2 * FP_COST + 3 * FN_COST


def test_one_hot_policy_keeps_binary_levels_but_excludes_leakage_columns() -> None:
    raw = load_raw_data()
    schema = feature_schema(raw)
    preprocessor = build_preprocessor(schema["numeric_columns"], schema["categorical_columns"])
    preprocessor.fit(raw[schema["raw_feature_columns"]])
    names = set(preprocessor.get_feature_names_out())
    assert "OverTime_No" in names
    assert "OverTime_Yes" in names
    assert ID_COLUMN not in names
    for constant_column in DROP_CONSTANT_COLUMNS:
        assert constant_column not in names
