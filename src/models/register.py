"""MLflow model registry governance utilities."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib

from src.models.train import MODEL_PATH, REGISTERED_MODEL_NAME, _configure_mlflow, train_models

try:
    from mlflow.tracking import MlflowClient
except Exception:  # pragma: no cover - depends on optional MLflow install.
    MlflowClient = None

MODEL_CARD_PATH = Path("reports/model_card.json")
FAIRNESS_REPORT_PATH = Path("reports/performance_logs/fairness_report.json")
PROMOTION_REPORT_PATH = Path("reports/performance_logs/model_promotion_report.json")
PROMOTION_THRESHOLD = 0.01


def _load_local_champion() -> dict[str, Any]:
    if not MODEL_PATH.exists():
        train_models()
    return joblib.load(MODEL_PATH)


def _version_metrics(client: Any, version: Any) -> dict[str, float]:
    if not getattr(version, "run_id", None):
        return {}
    run = client.get_run(version.run_id)
    return dict(run.data.metrics)


def _latest_stage_version(client: Any, stage: str) -> Any | None:
    versions = client.get_latest_versions(REGISTERED_MODEL_NAME, stages=[stage])
    return versions[0] if versions else None


def compare_and_promote() -> dict[str, Any]:
    """Promote Staging when business cost improves, then PR-AUC, then recall."""
    local_champion = _load_local_champion()
    local_auc = float(local_champion.get("metrics", {}).get("auc", 0.0))
    promotion_result: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "registered_model_name": REGISTERED_MODEL_NAME,
        "local_champion_auc": local_auc,
        "local_champion_pr_auc": local_champion.get("metrics", {}).get("pr_auc"),
        "local_champion_business_cost": local_champion.get("metrics", {}).get("business_cost"),
        "promoted": False,
        "reason": "",
    }

    registry_available = _configure_mlflow() and MlflowClient is not None
    if not registry_available:
        promotion_result["reason"] = "MLflow registry is not available; wrote local model card only."
        return promotion_result

    client = MlflowClient()
    try:
        staging = _latest_stage_version(client, "Staging")
        production = _latest_stage_version(client, "Production")
    except Exception as exc:
        promotion_result["reason"] = f"Could not read MLflow registry: {exc}"
        return promotion_result

    if staging is None:
        promotion_result["reason"] = "No Staging model version found."
        return promotion_result

    staging_metrics = _version_metrics(client, staging)
    production_metrics = _version_metrics(client, production) if production is not None else {}
    staging_auc = staging_metrics.get("auc")
    production_auc = production_metrics.get("auc")
    promotion_result.update(
        {
            "staging_version": staging.version,
            "staging_auc": staging_auc,
            "staging_pr_auc": staging_metrics.get("pr_auc") or staging_metrics.get("average_precision"),
            "staging_business_cost": staging_metrics.get("business_cost"),
            "staging_recall": staging_metrics.get("recall"),
            "production_version": production.version if production is not None else None,
            "production_auc": production_auc,
            "production_pr_auc": production_metrics.get("pr_auc") or production_metrics.get("average_precision"),
            "production_business_cost": production_metrics.get("business_cost"),
            "production_recall": production_metrics.get("recall"),
        }
    )

    if production_auc is None:
        should_promote = True
        reason = "No Production model exists."
    else:
        staging_cost = staging_metrics.get("business_cost")
        production_cost = production_metrics.get("business_cost")
        staging_pr_auc = staging_metrics.get("pr_auc") or staging_metrics.get("average_precision") or 0.0
        production_pr_auc = production_metrics.get("pr_auc") or production_metrics.get("average_precision") or 0.0
        staging_recall = staging_metrics.get("recall") or 0.0
        production_recall = production_metrics.get("recall") or 0.0
        if staging_cost is not None and production_cost is not None:
            should_promote = (
                staging_cost < production_cost
                or (staging_cost == production_cost and staging_pr_auc > production_pr_auc)
                or (
                    staging_cost == production_cost
                    and staging_pr_auc == production_pr_auc
                    and staging_recall > production_recall
                )
            )
            reason = "Staging improves business-cost governance ranking." if should_promote else "Staging does not improve business-cost governance ranking."
        else:
            required_auc = production_auc * (1 + PROMOTION_THRESHOLD)
            should_promote = staging_auc is not None and staging_auc >= required_auc
            reason = (
                f"Staging AUC {staging_auc:.4f} >= required AUC {required_auc:.4f}."
                if should_promote
                else f"Staging AUC did not beat Production by {PROMOTION_THRESHOLD:.0%}."
            )

    if should_promote:
        if production is not None:
            client.transition_model_version_stage(
                name=REGISTERED_MODEL_NAME,
                version=production.version,
                stage="Archived",
                archive_existing_versions=False,
            )
        client.transition_model_version_stage(
            name=REGISTERED_MODEL_NAME,
            version=staging.version,
            stage="Production",
            archive_existing_versions=True,
        )
        promotion_result["promoted"] = True
    promotion_result["reason"] = reason
    return promotion_result


def generate_model_card(promotion_result: dict[str, Any]) -> dict[str, Any]:
    """Generate a model card JSON artifact for governance review."""
    champion = _load_local_champion()
    fairness_summary: dict[str, Any] = {}
    if FAIRNESS_REPORT_PATH.exists():
        try:
            fairness_summary = json.loads(FAIRNESS_REPORT_PATH.read_text(encoding="utf-8"))
        except Exception:
            fairness_summary = {}
    card = {
        "model_name": REGISTERED_MODEL_NAME,
        "description": "Binary classifier estimating employee attrition risk for HR retention planning.",
        "intended_use": "Portfolio MLOps governance demo, HR risk prioritization, and model monitoring workflows.",
        "not_for_use": [
            "Automated employment decisions.",
            "Termination, promotion, compensation, or discipline decisions without human review.",
            "Production use on real HR records without legal, privacy, and fairness review.",
        ],
        "limitations": [
            "IBM HR Attrition is a small public dataset and may not reflect a real workforce.",
            "Predictions should support human HR review, not automated employment decisions.",
            "Fairness metrics can be unstable for small groups; review sample warnings before interpreting gaps.",
        ],
        "preprocessing": {
            "approach": "Train/test split before preprocessing, ColumnTransformer pipelines per model, median numeric imputation, most-frequent categorical imputation, one-hot encoding for categoricals.",
            "excluded_features": champion.get("excluded_features"),
            "leakage_control": "EmployeeNumber and constant columns are excluded from training features.",
            "one_hot_encoding": {
                "drop": None,
                "binary_indicators": (
                    "Both levels of binary categorical features, such as OverTime_No and OverTime_Yes, "
                    "are retained for readability in this demo. The model still excludes identifiers, "
                    "constant columns, and target-derived fields."
                ),
                "handle_unknown": "ignore",
            },
        },
        "decision_policy": {
            "champion_selection": "Lowest business cost first, then PR-AUC, then recall.",
            "active_production_threshold": champion.get("decision_threshold"),
            "best_f1_threshold": champion.get("thresholds", {}).get("best_f1_threshold"),
            "best_recall_threshold": champion.get("thresholds", {}).get("best_recall_threshold"),
            "best_business_cost_threshold": champion.get("thresholds", {}).get("best_business_cost_threshold"),
            "candidate_model_comparison_threshold": (
                "candidate_selected_threshold is a per-candidate diagnostic from comparison. "
                "active_production_threshold is the final champion threshold used for API and dashboard decisions."
            ),
            "business_cost_formula": "business_cost = false_positive_cost * FP + false_negative_cost * FN",
            "false_positive_cost": champion.get("thresholds", {}).get("false_positive_cost"),
            "false_negative_cost": champion.get("thresholds", {}).get("false_negative_cost"),
            "thresholds": champion.get("thresholds"),
            "threshold_policy": champion.get("threshold_policy"),
            "calibration": champion.get("calibration"),
        },
        "performance": champion.get("metrics", {}),
        "champion_model_type": champion.get("model_name"),
        "training_date_utc": champion.get("trained_at_utc"),
        "data_version": "data/raw/hr_attrition.csv",
        "mlflow_run_id": champion.get("mlflow_run_id"),
        "model_registry_version": champion.get("model_registry_version"),
        "fairness_summary": {
            "report_path": str(FAIRNESS_REPORT_PATH),
            "generated_at_utc": fairness_summary.get("generated_at_utc"),
            "flagged_dimensions": fairness_summary.get("flagged_dimensions", []),
        },
        "promotion_result": promotion_result,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    MODEL_CARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODEL_CARD_PATH.write_text(json.dumps(card, indent=2), encoding="utf-8")
    PROMOTION_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROMOTION_REPORT_PATH.write_text(json.dumps(promotion_result, indent=2), encoding="utf-8")
    return card


def main() -> dict[str, Any]:
    """Compare registry stages, promote if warranted, and write the model card."""
    promotion_result = compare_and_promote()
    model_card = generate_model_card(promotion_result)
    print(f"Model card saved to {MODEL_CARD_PATH}")
    print(json.dumps(promotion_result, indent=2))
    return model_card


if __name__ == "__main__":
    main()
