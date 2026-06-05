"""Train HR attrition models and track governance metadata."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.features.engineering import (
    ID_COLUMN,
    PREPROCESSING_PATH,
    PROCESSED_PATH,
    RAW_DATA_PATH,
    TARGET_COLUMN,
    build_preprocessor,
    encode_target,
    engineer_features,
    feature_schema,
    load_raw_data,
    save_feature_outputs,
    split_raw_data,
)

try:
    import mlflow
    import mlflow.sklearn
    from mlflow.tracking import MlflowClient
except Exception:  # pragma: no cover - optional dependency in lean environments.
    mlflow = None
    MlflowClient = None

TRAIN_PATH = Path("data/processed/train_features.csv")
TEST_PATH = Path("data/processed/test_features.csv")
MODEL_PATH = Path("models/attrition_model.joblib")
LEGACY_MODEL_PATH = Path("models/best_model.joblib")
SHAP_REPORT_DIR = Path("reports/shap_reports")
PERFORMANCE_DIR = Path("reports/performance_logs")
METRICS_PATH = PERFORMANCE_DIR / "training_metrics.json"
CANDIDATE_COMPARISON_PATH = PERFORMANCE_DIR / "candidate_model_comparison.csv"
THRESHOLD_METRICS_PATH = PERFORMANCE_DIR / "threshold_metrics.csv"
CALIBRATION_CURVE_PATH = PERFORMANCE_DIR / "calibration_curve.png"
REGISTERED_MODEL_NAME = "AttritionPredictor"
EXPERIMENT_NAME = "HR Attrition Governance"
RANDOM_STATE = 42
FN_COST = int(os.getenv("ATTRITION_FALSE_NEGATIVE_COST", "15000"))
FP_COST = int(os.getenv("ATTRITION_FALSE_POSITIVE_COST", "1500"))


def business_cost_from_counts(
    false_positives: int,
    false_negatives: int,
    *,
    false_positive_cost: int = FP_COST,
    false_negative_cost: int = FN_COST,
) -> int:
    """Return the configured business cost for confusion-matrix errors."""
    return int(false_positive_cost * false_positives + false_negative_cost * false_negatives)


def confusion_counts_at_threshold(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, int]:
    """Return confusion-matrix counts for a concrete operating threshold."""
    y_array = np.asarray(y_true).astype(int)
    probability_array = np.asarray(probabilities, dtype=float)
    predictions = (probability_array >= float(threshold)).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_array, predictions, labels=[0, 1]).ravel()
    return {
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def metrics_at_threshold(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    *,
    model_name: str = "model",
) -> dict[str, Any]:
    """Calculate threshold-specific classification and business-cost metrics."""
    y_array = np.asarray(y_true).astype(int)
    probability_array = np.asarray(probabilities, dtype=float)
    predictions = (probability_array >= float(threshold)).astype(int)
    counts = confusion_counts_at_threshold(y_array, probability_array, threshold)
    return {
        "model_name": model_name,
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_array, predictions)),
        "precision": float(precision_score(y_array, predictions, zero_division=0)),
        "recall": float(recall_score(y_array, predictions, zero_division=0)),
        "f1": float(f1_score(y_array, predictions, zero_division=0)),
        **counts,
        "false_positive_cost": FP_COST,
        "false_negative_cost": FN_COST,
        "business_cost": business_cost_from_counts(
            counts["false_positives"],
            counts["false_negatives"],
        ),
    }


def _ensure_processed_data() -> None:
    if not PROCESSED_PATH.exists() or not TRAIN_PATH.exists() or not TEST_PATH.exists():
        raw_df = load_raw_data()
        outputs = engineer_features(raw_df)
        save_feature_outputs(*outputs)


def load_training_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, list[str]]:
    """Load raw train/test data for model pipelines."""
    raw_df = load_raw_data()
    schema = feature_schema(raw_df)
    X_train, X_test, y_train, y_test = split_raw_data(raw_df)
    feature_columns = schema["raw_feature_columns"]
    return X_train[feature_columns], X_test[feature_columns], y_train, y_test, feature_columns


def _positive_class_scores(model: Any, X: pd.DataFrame) -> np.ndarray:
    """Return positive-class probabilities from classifiers or calibrated wrappers."""
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X))[:, 1]
    if hasattr(model, "decision_function"):
        logits = model.decision_function(X)
        return 1 / (1 + np.exp(-logits))
    return np.asarray(model.predict(X), dtype=float)


def _threshold_metrics(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    *,
    model_name: str,
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    """Calculate precision, recall, F1, confusion matrix, and business cost by threshold."""
    thresholds = thresholds if thresholds is not None else np.round(np.linspace(0.05, 0.95, 19), 2)
    rows: list[dict[str, Any]] = []
    y_array = np.asarray(y_true).astype(int)
    for threshold in thresholds:
        rows.append(metrics_at_threshold(y_array, probabilities, float(threshold), model_name=model_name))
    return pd.DataFrame(rows)


def _best_thresholds(threshold_df: pd.DataFrame) -> dict[str, Any]:
    best_f1 = threshold_df.sort_values(["f1", "recall"], ascending=[False, False]).iloc[0]
    best_recall = threshold_df.sort_values(["recall", "precision"], ascending=[False, False]).iloc[0]
    best_cost = threshold_df.sort_values(["business_cost", "recall"], ascending=[True, False]).iloc[0]
    return {
        "best_f1_threshold": float(best_f1["threshold"]),
        "best_recall_threshold": float(best_recall["threshold"]),
        "best_business_cost_threshold": float(best_cost["threshold"]),
        "selected_threshold": float(best_cost["threshold"]),
        "selected_business_cost": int(best_cost["business_cost"]),
        "selected_precision": float(best_cost["precision"]),
        "selected_recall": float(best_cost["recall"]),
        "selected_f1": float(best_cost["f1"]),
        "selected_confusion_matrix": {
            "true_negatives": int(best_cost["true_negatives"]),
            "false_positives": int(best_cost["false_positives"]),
            "false_negatives": int(best_cost["false_negatives"]),
            "true_positives": int(best_cost["true_positives"]),
        },
    }


def calculate_metrics(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    model_name: str = "model",
) -> dict[str, Any]:
    """Calculate model metrics with tuned threshold details."""
    probabilities = _positive_class_scores(model, X_test)
    threshold_df = _threshold_metrics(y_test, probabilities, model_name=model_name)
    thresholds = _best_thresholds(threshold_df)
    predictions = (probabilities >= thresholds["selected_threshold"]).astype(int)
    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "auc": float(roc_auc_score(y_test, probabilities)),
        "pr_auc": float(average_precision_score(y_test, probabilities)),
        "average_precision": float(average_precision_score(y_test, probabilities)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "brier_score": float(brier_score_loss(y_test, probabilities)),
        "business_cost": int(thresholds["selected_business_cost"]),
        "thresholds": thresholds,
        "threshold_table": threshold_df.to_dict(orient="records"),
    }


def _candidate_models(schema: dict[str, Any], y_train: pd.Series) -> dict[str, Pipeline]:
    """Build model-specific pipelines."""
    negative_count = int((y_train == 0).sum())
    positive_count = int((y_train == 1).sum())
    scale_pos_weight = negative_count / max(positive_count, 1)

    candidates: dict[str, Pipeline] = {
        "logistic_regression": Pipeline(
            steps=[
                (
                    "preprocess",
                    build_preprocessor(schema["numeric_columns"], schema["categorical_columns"], scale_numeric=True),
                ),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2500,
                        class_weight="balanced",
                        solver="liblinear",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )
    }

    try:
        from xgboost import XGBClassifier

        candidates["xgboost"] = Pipeline(
            steps=[
                (
                    "preprocess",
                    build_preprocessor(schema["numeric_columns"], schema["categorical_columns"], scale_numeric=False),
                ),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=180,
                        max_depth=3,
                        learning_rate=0.05,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        objective="binary:logistic",
                        eval_metric="logloss",
                        scale_pos_weight=scale_pos_weight,
                        random_state=RANDOM_STATE,
                        n_jobs=1,
                    ),
                ),
            ]
        )
    except Exception as exc:  # pragma: no cover - optional package behavior.
        print(f"Skipping XGBoost: {exc}")

    try:
        from lightgbm import LGBMClassifier

        candidates["lightgbm"] = Pipeline(
            steps=[
                (
                    "preprocess",
                    build_preprocessor(schema["numeric_columns"], schema["categorical_columns"], scale_numeric=False),
                ),
                (
                    "model",
                    LGBMClassifier(
                        n_estimators=180,
                        max_depth=4,
                        learning_rate=0.05,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        verbose=-1,
                    ),
                ),
            ]
        )
    except Exception as exc:  # pragma: no cover - optional package behavior.
        print(f"Skipping LightGBM: {exc}")

    return candidates


def _configure_mlflow() -> bool:
    if mlflow is None:
        print("MLflow is not installed; training will continue without run tracking.")
        return False
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or "sqlite:///mlflow.db"
    if tracking_uri.startswith("file:"):
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    try:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_registry_uri(tracking_uri)
        mlflow.set_experiment(EXPERIMENT_NAME)
        return True
    except Exception as exc:
        print(f"MLflow tracking unavailable; using local model artifacts only: {exc}")
        return False


def _log_params_safely(params: dict[str, Any]) -> None:
    if mlflow is None:
        return
    mlflow.log_params({key: str(value)[:250] for key, value in params.items()})


def _feature_names_from_pipeline(pipeline: Pipeline) -> list[str]:
    return [str(name) for name in pipeline.named_steps["preprocess"].get_feature_names_out()]


def _estimator_from_pipeline(pipeline: Pipeline) -> Any:
    return pipeline.named_steps["model"]


def _plot_feature_importance(pipeline: Pipeline, output_path: Path) -> None:
    """Save a compact global importance plot for the transformed model features."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    estimator = _estimator_from_pipeline(pipeline)
    feature_names = _feature_names_from_pipeline(pipeline)
    if hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_)
    elif hasattr(estimator, "coef_"):
        values = np.abs(np.asarray(estimator.coef_).ravel())
    else:
        values = np.zeros(len(feature_names))

    importance = (
        pd.DataFrame({"feature": feature_names, "importance": values})
        .sort_values("importance", ascending=False)
        .head(20)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 7))
    plt.barh(importance["feature"][::-1], importance["importance"][::-1], color="#2563eb")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_shap_summary_plot(
    model: Any,
    X_test: pd.DataFrame,
    feature_names: list[str] | None = None,
    model_name: str = "champion",
) -> Path:
    """Save a SHAP-like summary plot, falling back to model importance for pipeline models."""
    output_path = SHAP_REPORT_DIR / f"{model_name}_shap_summary.png"
    if isinstance(model, Pipeline):
        _plot_feature_importance(model, output_path)
        return output_path

    # Compatibility fallback for non-pipeline estimators.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = feature_names or list(X_test.columns)
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_)
    elif hasattr(model, "coef_"):
        values = np.abs(np.asarray(model.coef_).ravel())
    else:
        values = np.zeros(len(names))
    importance = pd.DataFrame({"feature": names, "importance": values}).sort_values(
        "importance", ascending=False
    ).head(20)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 7))
    plt.barh(importance["feature"][::-1], importance["importance"][::-1], color="#2563eb")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def _calibration_plot(y_test: pd.Series, probabilities: np.ndarray, brier_score: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    prob_true, prob_pred = calibration_curve(y_test, probabilities, n_bins=8, strategy="uniform")
    CALIBRATION_CURVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 5))
    plt.plot(prob_pred, prob_true, marker="o", label="Model")
    plt.plot([0, 1], [0, 1], linestyle="--", color="#667085", label="Perfect calibration")
    plt.title(f"Calibration Curve (Brier={brier_score:.3f})")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.legend()
    plt.tight_layout()
    plt.savefig(CALIBRATION_CURVE_PATH, dpi=150)
    plt.close()


def _try_calibrate(
    pipeline: Pipeline,
    X_calib: pd.DataFrame,
    y_calib: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[Any, dict[str, Any]]:
    """Calibrate a fitted pipeline when it improves Brier score."""
    original_probabilities = _positive_class_scores(pipeline, X_test)
    original_brier = float(brier_score_loss(y_test, original_probabilities))
    calibration_result = {
        "method": "none",
        "calibrated": False,
        "brier_score_before": original_brier,
        "brier_score_after": original_brier,
        "status": "acceptable" if original_brier <= 0.2 else "needs_review",
        "plot_path": str(CALIBRATION_CURVE_PATH),
    }

    try:
        try:
            calibrated_model = CalibratedClassifierCV(estimator=pipeline, method="sigmoid", cv="prefit")
        except TypeError:
            calibrated_model = CalibratedClassifierCV(base_estimator=pipeline, method="sigmoid", cv="prefit")
        calibrated_model.fit(X_calib, y_calib)
        calibrated_probabilities = _positive_class_scores(calibrated_model, X_test)
        calibrated_brier = float(brier_score_loss(y_test, calibrated_probabilities))
        if calibrated_brier < original_brier:
            calibration_result.update(
                {
                    "method": "sigmoid",
                    "calibrated": True,
                    "brier_score_after": calibrated_brier,
                    "status": "acceptable" if calibrated_brier <= 0.2 else "needs_review",
                }
            )
            _calibration_plot(y_test, calibrated_probabilities, calibrated_brier)
            return calibrated_model, calibration_result
    except Exception as exc:
        calibration_result["error"] = str(exc)

    _calibration_plot(y_test, original_probabilities, original_brier)
    return pipeline, calibration_result


def _register_champion(run_id: str | None) -> str | None:
    if not run_id or mlflow is None or MlflowClient is None:
        return None
    try:
        model_uri = f"runs:/{run_id}/model"
        version = mlflow.register_model(model_uri, REGISTERED_MODEL_NAME)
        client = MlflowClient()
        client.transition_model_version_stage(
            name=REGISTERED_MODEL_NAME,
            version=version.version,
            stage="Staging",
            archive_existing_versions=False,
        )
        return str(version.version)
    except Exception as exc:  # pragma: no cover - registry support depends on backend.
        print(f"MLflow registry registration skipped: {exc}")
        return None


def _risk_bands(selected_threshold: float) -> dict[str, float]:
    low = min(0.35, max(0.05, selected_threshold * 0.75))
    high = max(0.65, min(0.95, selected_threshold * 1.35))
    return {
        "low_threshold": float(round(low, 3)),
        "medium_threshold": float(round(selected_threshold, 3)),
        "high_threshold": float(round(high, 3)),
    }


def train_models() -> dict[str, Any]:
    """Train candidate models, select champion by business cost, and save artifacts."""
    _ensure_processed_data()
    raw_df = load_raw_data()
    schema = feature_schema(raw_df)
    X_train, X_test, y_train, y_test = split_raw_data(raw_df)
    feature_columns = schema["raw_feature_columns"]
    X_train = X_train[feature_columns]
    X_test = X_test[feature_columns]
    X_fit, X_calib, y_fit, y_calib = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_train,
    )

    tracking_enabled = _configure_mlflow()
    candidates = _candidate_models(schema, y_fit)
    if not candidates:
        raise RuntimeError("No candidate models are available for training.")

    run_records: list[dict[str, Any]] = []
    threshold_tables: list[pd.DataFrame] = []

    for model_name, pipeline in candidates.items():
        run_context = mlflow.start_run(run_name=model_name) if tracking_enabled else None
        if run_context is not None:
            run_context.__enter__()
        try:
            fitted = clone(pipeline)
            fitted.fit(X_fit, y_fit)
            metrics = calculate_metrics(fitted, X_test, y_test, model_name=model_name)
            threshold_tables.append(pd.DataFrame(metrics["threshold_table"]))
            shap_plot_path = save_shap_summary_plot(fitted, X_test, model_name=model_name)

            run_id = mlflow.active_run().info.run_id if tracking_enabled else None
            if tracking_enabled:
                mlflow.set_tag("model_type", model_name)
                _log_params_safely(_estimator_from_pipeline(fitted).get_params())
                mlflow.log_metrics(
                    {
                        "auc": metrics["auc"],
                        "pr_auc": metrics["pr_auc"],
                        "f1": metrics["f1"],
                        "recall": metrics["recall"],
                        "precision": metrics["precision"],
                        "business_cost": metrics["business_cost"],
                        "brier_score": metrics["brier_score"],
                    }
                )
                mlflow.log_artifact(str(shap_plot_path))
                mlflow.sklearn.log_model(fitted, artifact_path="model")

            run_records.append(
                {
                    "model_name": model_name,
                    "pipeline": fitted,
                    "metrics": metrics,
                    "run_id": run_id,
                    "shap_plot_path": str(shap_plot_path),
                }
            )
        finally:
            if run_context is not None:
                run_context.__exit__(None, None, None)

    comparison_rows = []
    for record in run_records:
        metrics = record["metrics"]
        comparison_rows.append(
            {
                "model_name": record["model_name"],
                "auc": metrics["auc"],
                "pr_auc": metrics["pr_auc"],
                "average_precision": metrics["average_precision"],
                "f1": metrics["f1"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "candidate_business_cost": metrics["business_cost"],
                "brier_score": metrics["brier_score"],
                "candidate_selected_threshold": metrics["thresholds"]["selected_threshold"],
                "candidate_best_f1_threshold": metrics["thresholds"]["best_f1_threshold"],
                "candidate_best_recall_threshold": metrics["thresholds"]["best_recall_threshold"],
                "candidate_best_business_cost_threshold": metrics["thresholds"]["best_business_cost_threshold"],
                "run_id": record["run_id"],
            }
        )

    comparison_df = pd.DataFrame(comparison_rows).sort_values(
        ["candidate_business_cost", "pr_auc", "recall"],
        ascending=[True, False, False],
    )
    best_name = str(comparison_df.iloc[0]["model_name"])
    best_record = next(record for record in run_records if record["model_name"] == best_name)
    champion_pipeline = best_record["pipeline"]
    champion_model, calibration = _try_calibrate(champion_pipeline, X_calib, y_calib, X_test, y_test)
    champion_probabilities = _positive_class_scores(champion_model, X_test)
    champion_threshold_df = _threshold_metrics(y_test, champion_probabilities, model_name=best_name)
    champion_thresholds = _best_thresholds(champion_threshold_df)
    champion_metrics = calculate_metrics(champion_model, X_test, y_test, model_name=best_name)
    champion_metrics["thresholds"] = champion_thresholds
    champion_metrics["threshold_table"] = champion_threshold_df.to_dict(orient="records")
    champion_metrics["business_cost"] = int(champion_thresholds["selected_business_cost"])
    champion_metrics["brier_score"] = float(brier_score_loss(y_test, champion_probabilities))
    active_threshold_metrics = metrics_at_threshold(
        y_test,
        champion_probabilities,
        champion_thresholds["selected_threshold"],
        model_name=best_name,
    )
    champion_metrics["active_threshold_metrics"] = active_threshold_metrics
    champion_metrics["active_confusion_matrix"] = {
        key: active_threshold_metrics[key]
        for key in ["true_negatives", "false_positives", "false_negatives", "true_positives"]
    }

    if tracking_enabled and best_record["run_id"] and MlflowClient is not None:
        try:
            MlflowClient().set_tag(best_record["run_id"], "model_role", "champion")
        except Exception as exc:
            print(f"MLflow champion tag skipped: {exc}")

    registry_version = _register_champion(best_record["run_id"]) if tracking_enabled else None
    transformed_feature_names = _feature_names_from_pipeline(champion_pipeline)
    risk_bands = _risk_bands(champion_thresholds["selected_threshold"])
    thresholds = {
        **champion_thresholds,
        **risk_bands,
        "false_negative_cost": FN_COST,
        "false_positive_cost": FP_COST,
    }
    threshold_policy = {
        "active_production_threshold": thresholds["selected_threshold"],
        "best_f1_threshold": champion_thresholds["best_f1_threshold"],
        "best_recall_threshold": champion_thresholds["best_recall_threshold"],
        "best_business_cost_threshold": champion_thresholds["best_business_cost_threshold"],
        "candidate_model_comparison_threshold": (
            "candidate_selected_threshold is calculated per candidate during model comparison. "
            "The active production threshold is recalculated for the final champion artifact after calibration."
        ),
        "business_cost_formula": "business_cost = false_positive_cost * FP + false_negative_cost * FN",
        "false_positive_cost": FP_COST,
        "false_negative_cost": FN_COST,
    }

    model_card_metadata = {
        "intended_use": "Portfolio MLOps governance demo and HR retention risk prioritization support.",
        "prohibited_use": "Not for automated employment decisions, termination, compensation, or promotion decisions.",
        "human_review_required": True,
        "sensitive_feature_warning": "HR data may contain sensitive or proxy-sensitive features. Use fairness review before any real deployment.",
        "training_data_summary": {
            "dataset": str(RAW_DATA_PATH),
            "rows": int(raw_df.shape[0]),
            "columns": int(raw_df.shape[1]),
            "target": TARGET_COLUMN,
        },
    }

    artifact = {
        "model": champion_model,
        "model_pipeline": champion_model,
        "uncalibrated_pipeline": champion_pipeline,
        "model_name": best_name,
        "metrics": champion_metrics,
        "feature_schema": schema,
        "feature_names": transformed_feature_names,
        "raw_feature_columns": schema["raw_feature_columns"],
        "categorical_columns": schema["categorical_columns"],
        "numeric_columns": schema["numeric_columns"],
        "excluded_features": schema["excluded_features"],
        "id_column": ID_COLUMN,
        "target_column": TARGET_COLUMN,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "mlflow_run_id": best_record["run_id"],
        "registered_model_name": REGISTERED_MODEL_NAME,
        "model_registry_version": registry_version,
        "preprocessing_path": str(PREPROCESSING_PATH),
        "decision_threshold": thresholds["selected_threshold"],
        "threshold": thresholds["selected_threshold"],
        "thresholds": thresholds,
        "threshold_policy": threshold_policy,
        "risk_bands": risk_bands,
        "candidate_model_comparison": comparison_df.to_dict(orient="records"),
        "calibration": calibration,
        "model_card_metadata": model_card_metadata,
        "shap_plot_path": best_record["shap_plot_path"],
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    PERFORMANCE_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, MODEL_PATH)
    joblib.dump(artifact, LEGACY_MODEL_PATH)

    preprocessing_payload = {
        **schema,
        "preprocessor": champion_pipeline.named_steps["preprocess"],
        "feature_columns": transformed_feature_names,
        "risk_bands": risk_bands,
        "thresholds": thresholds,
        "threshold_policy": threshold_policy,
        "encoding_policy": {
            "one_hot_drop": None,
            "handle_unknown": "ignore",
            "binary_indicators": "Both binary levels are retained for readability in this demo; no identifiers or target-derived columns are encoded.",
        },
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    joblib.dump(preprocessing_payload, PREPROCESSING_PATH)

    comparison_df.to_csv(CANDIDATE_COMPARISON_PATH, index=False)
    pd.concat([*threshold_tables, champion_threshold_df], ignore_index=True).drop_duplicates().to_csv(
        THRESHOLD_METRICS_PATH, index=False
    )

    metrics_payload = {
        "champion": {
            "model_name": artifact["model_name"],
            "metrics": artifact["metrics"],
            "run_id": artifact["mlflow_run_id"],
            "model_registry_version": artifact["model_registry_version"],
            "selection_rule": "lowest business_cost, then higher pr_auc, then higher recall",
        },
        "candidates": comparison_df.to_dict(orient="records"),
        "threshold_metrics_path": str(THRESHOLD_METRICS_PATH),
        "candidate_comparison_path": str(CANDIDATE_COMPARISON_PATH),
        "calibration": calibration,
    }
    METRICS_PATH.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    print(f"Champion model saved to {MODEL_PATH}")
    print(json.dumps(metrics_payload["champion"], indent=2, default=str))
    return metrics_payload


def main() -> dict[str, Any]:
    """CLI entrypoint."""
    return train_models()


if __name__ == "__main__":
    main()
