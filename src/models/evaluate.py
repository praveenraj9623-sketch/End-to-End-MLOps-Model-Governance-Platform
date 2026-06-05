"""Model evaluation, explainability, and business impact reporting."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.models.train import (
    FN_COST,
    FP_COST,
    MODEL_PATH,
    TARGET_COLUMN,
    _positive_class_scores,
    load_training_data,
    save_shap_summary_plot,
    train_models,
)

PERFORMANCE_DIR = Path("reports/performance_logs")
SHAP_REPORT_DIR = Path("reports/shap_reports")
CONFUSION_MATRIX_PATH = Path("reports/performance_logs/confusion_matrix.png")
METRICS_CSV_PATH = Path("reports/performance_logs/model_metrics.csv")
LATEST_SNAPSHOT_PATH = Path("reports/performance_logs/latest_performance_snapshot.json")


class ModelEvaluator:
    """Evaluate the champion attrition model and produce governance artifacts."""

    def __init__(
        self,
        model_artifact: dict[str, Any] | None = None,
        X_test: pd.DataFrame | None = None,
        y_test: pd.Series | None = None,
        feature_names: list[str] | None = None,
    ) -> None:
        if model_artifact is None:
            if not MODEL_PATH.exists():
                train_models()
            model_artifact = joblib.load(MODEL_PATH)
        if X_test is None or y_test is None or feature_names is None:
            _, X_test, _, y_test, feature_names = load_training_data()

        self.model_artifact = model_artifact
        self.model = model_artifact["model"] if isinstance(model_artifact, dict) else model_artifact
        self.X_test = X_test
        self.y_test = y_test.astype(int)
        self.feature_names = feature_names
        self.threshold = float(
            model_artifact.get("decision_threshold")
            or model_artifact.get("threshold")
            or 0.5
        ) if isinstance(model_artifact, dict) else 0.5
        self.probabilities = _positive_class_scores(self.model, self.X_test)
        self.predictions = (self.probabilities >= self.threshold).astype(int)

    def generate_classification_report(
        self, model: Any | None = None, X_test: pd.DataFrame | None = None, y_test: pd.Series | None = None
    ) -> dict[str, Any]:
        """Return standard classification metrics and sklearn's detailed report."""
        model = model or self.model
        X_test = X_test if X_test is not None else self.X_test
        y_test = y_test.astype(int) if y_test is not None else self.y_test
        probabilities = _positive_class_scores(model, X_test)
        threshold = self.threshold if model is self.model else 0.5
        predictions = (probabilities >= threshold).astype(int)
        return {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "auc": float(roc_auc_score(y_test, probabilities)),
            "pr_auc": float(average_precision_score(y_test, probabilities)),
            "average_precision": float(average_precision_score(y_test, probabilities)),
            "brier_score": float(brier_score_loss(y_test, probabilities)),
            "f1": float(f1_score(y_test, predictions, zero_division=0)),
            "precision": float(precision_score(y_test, predictions, zero_division=0)),
            "recall": float(recall_score(y_test, predictions, zero_division=0)),
            "decision_threshold": float(threshold),
            "classification_report": classification_report(
                y_test, predictions, output_dict=True, zero_division=0
            ),
        }

    def generate_shap_report(
        self,
        model: Any | None = None,
        X_test: pd.DataFrame | None = None,
        feature_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """Generate and save a SHAP summary plot for the evaluated model."""
        model = model or self.model_artifact.get("uncalibrated_pipeline") or self.model
        X_test = X_test if X_test is not None else self.X_test
        feature_names = feature_names or self.model_artifact.get("feature_names") or self.feature_names
        model_name = self.model_artifact.get("model_name", "champion")
        path = save_shap_summary_plot(model, X_test, feature_names, f"{model_name}_evaluation")
        return {"shap_summary_plot": str(path)}

    def generate_confusion_matrix_plot(self) -> dict[str, Any]:
        """Generate and save the confusion matrix plot."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        matrix = confusion_matrix(self.y_test, self.predictions)
        CONFUSION_MATRIX_PATH.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(5, 4))
        image = ax.imshow(matrix, cmap="Blues")
        ax.set_xticks([0, 1], labels=["No", "Yes"])
        ax.set_yticks([0, 1], labels=["No", "Yes"])
        ax.set_xlabel("Predicted Attrition")
        ax.set_ylabel("Actual Attrition")
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                ax.text(col, row, matrix[row, col], ha="center", va="center", color="#111827")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(CONFUSION_MATRIX_PATH, dpi=150)
        plt.close(fig)
        return {
            "confusion_matrix_plot": str(CONFUSION_MATRIX_PATH),
            "confusion_matrix": matrix.tolist(),
        }

    def calculate_business_metrics(self) -> dict[str, Any]:
        """Estimate HR cost from false negatives and false positives."""
        tn, fp, fn, tp = confusion_matrix(self.y_test, self.predictions).ravel()
        missed_attrition_cost = int(fn * FN_COST)
        unnecessary_intervention_cost = int(fp * FP_COST)
        total_cost = missed_attrition_cost + unnecessary_intervention_cost
        return {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
            "decision_threshold": self.threshold,
            "false_negative_cost_usd": FN_COST,
            "false_positive_cost_usd": FP_COST,
            "missed_attrition_cost_usd": missed_attrition_cost,
            "unnecessary_intervention_cost_usd": unnecessary_intervention_cost,
            "estimated_total_error_cost_usd": total_cost,
            "cost_context": "False negatives represent likely attrition cases HR missed; false positives represent unnecessary retention outreach.",
        }

    def save_performance_snapshot(self, metrics_dict: dict[str, Any]) -> Path:
        """Save timestamped JSON plus stable CSV/JSON latest metrics."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        PERFORMANCE_DIR.mkdir(parents=True, exist_ok=True)
        snapshot_path = PERFORMANCE_DIR / f"performance_snapshot_{timestamp}.json"
        payload = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "model_name": self.model_artifact.get("model_name", "unknown"),
            "model_registry_version": self.model_artifact.get("model_registry_version"),
            "mlflow_run_id": self.model_artifact.get("mlflow_run_id"),
            "metrics": metrics_dict,
        }
        snapshot_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        LATEST_SNAPSHOT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        flat_metrics = {
            "model_name": payload["model_name"],
            "model_registry_version": payload["model_registry_version"],
            "mlflow_run_id": payload["mlflow_run_id"],
            "timestamp_utc": payload["timestamp_utc"],
            "accuracy": metrics_dict["classification"]["accuracy"],
            "auc": metrics_dict["classification"]["auc"],
            "pr_auc": metrics_dict["classification"]["pr_auc"],
            "f1": metrics_dict["classification"]["f1"],
            "precision": metrics_dict["classification"]["precision"],
            "recall": metrics_dict["classification"]["recall"],
            "decision_threshold": metrics_dict["classification"]["decision_threshold"],
            "estimated_total_error_cost_usd": metrics_dict["business"][
                "estimated_total_error_cost_usd"
            ],
        }
        pd.DataFrame([flat_metrics]).to_csv(METRICS_CSV_PATH, index=False)
        return snapshot_path


def main() -> dict[str, Any]:
    """Run evaluation and save all reports."""
    evaluator = ModelEvaluator()
    classification = evaluator.generate_classification_report()
    shap_report = evaluator.generate_shap_report()
    confusion = evaluator.generate_confusion_matrix_plot()
    business = evaluator.calculate_business_metrics()
    metrics = {
        "classification": classification,
        "shap": shap_report,
        "confusion_matrix": confusion,
        "business": business,
    }
    snapshot_path = evaluator.save_performance_snapshot(metrics)
    print(f"Evaluation snapshot saved to {snapshot_path}")
    print(json.dumps({"auc": classification["auc"], "f1": classification["f1"]}, indent=2))
    return metrics


if __name__ == "__main__":
    main()
