"""Streamlit dashboard for the HR attrition MLOps governance platform.

App-only UI/UX polish version. It does not change training, prediction,
registry, drift, or fairness logic. Replace the current app.py with this file
only after taking a backup of your existing app.py.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from src.dashboard.data_loaders import load_dashboard_artifacts, read_csv
from src.dashboard.figures import (
    candidate_comparison_chart,
    confusion_matrix_chart,
    drift_feature_chart,
    fairness_group_chart,
    risk_distribution_chart,
    threshold_chart,
)
from src.dashboard.governance_logic import (
    active_threshold_metrics_for_display,
    audit_row_for_detail,
    confusion_counts_from_matrix,
    model_card_metadata_rows,
    normalize_candidate_comparison_columns,
    serialize_audit_dataframe,
    shorten_payload_columns,
    should_show_training_confusion_artifact,
)
from src.governance.fairness import FAIRNESS_REPORT_PATH, generate_fairness_report
from src.monitoring.drift_detection import (
    DRIFT_ALERT_PATH,
    drift_status_summary,
    main as run_drift_detection,
    reset_current_data,
    simulate_drift,
)
from src.services.governance_service import registry_versions
from src.services.prediction_service import predict_employee, read_audit_entries
from src.services.retraining_service import run_local_retrain_pipeline
from src.models.train import _positive_class_scores, load_training_data, metrics_at_threshold


st.set_page_config(
    page_title="MLOps Governance Platform",
    page_icon=":shield:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --bg: #f8fafc;
        --panel: #ffffff;
        --ink: #0f172a;
        --muted: #64748b;
        --line: #e2e8f0;
        --blue: #2563eb;
        --green: #16a34a;
        --amber: #d97706;
        --red: #dc2626;
        --soft-blue: #eff6ff;
        --soft-green: #ecfdf5;
        --soft-amber: #fffbeb;
        --soft-red: #fef2f2;
    }
    .block-container {
        padding-top: 1.15rem;
        padding-bottom: 2.5rem;
        max-width: 1320px;
    }
    [data-testid="stSidebar"] {
        background: #f1f5f9;
        border-right: 1px solid var(--line);
    }
    [data-testid="stMetric"] {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 1rem 1rem 0.8rem 1rem;
        box-shadow: 0 8px 22px rgba(15, 23, 42, 0.04);
    }
    [data-testid="stMetricLabel"] p {
        color: var(--muted);
        font-weight: 700;
        letter-spacing: .02em;
    }
    [data-testid="stMetricValue"] div {
        color: var(--ink);
        font-weight: 800;
    }
    div[data-testid="stDataFrame"] {
        border: 1px solid var(--line);
        border-radius: 12px;
        overflow: hidden;
    }
    .metadata-table {
        width: 100%;
        border-collapse: collapse;
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 12px;
        overflow: hidden;
        margin-bottom: 10px;
    }
    .metadata-table th,
    .metadata-table td {
        text-align: left;
        padding: 10px 12px;
        border-bottom: 1px solid var(--line);
        vertical-align: top;
    }
    .metadata-table th {
        color: var(--muted);
        font-weight: 800;
        background: #f8fafc;
    }
    .metadata-table td:first-child {
        width: 230px;
        font-weight: 800;
        color: var(--ink);
    }
    .metadata-table td:last-child {
        word-break: break-word;
    }
    .hero {
        background: linear-gradient(135deg, #0f172a 0%, #1d4ed8 100%);
        color: white;
        padding: 28px 32px;
        border-radius: 20px;
        margin: 4px 0 18px 0;
        box-shadow: 0 22px 45px rgba(15, 23, 42, 0.18);
    }
    .hero h1 {
        margin: 0 0 8px 0;
        font-size: 2.05rem;
        line-height: 1.15;
        color: #ffffff !important;
        text-shadow: 0 2px 12px rgba(15, 23, 42, 0.35);
    }
    .hero p { margin: 0; color: #dbeafe; font-size: 0.98rem; }
    .section-card {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 16px;
        padding: 18px 20px;
        margin: 12px 0 18px 0;
        box-shadow: 0 10px 25px rgba(15, 23, 42, 0.035);
    }
    .callout {
        border-left: 5px solid var(--blue);
        background: var(--soft-blue);
        padding: 12px 14px;
        border-radius: 10px;
        color: #1e3a8a;
        margin: 10px 0 14px 0;
    }
    .warn { border-left-color: var(--amber); background: var(--soft-amber); color: #92400e; }
    .danger { border-left-color: var(--red); background: var(--soft-red); color: #991b1b; }
    .ok { border-left-color: var(--green); background: var(--soft-green); color: #166534; }
    .small-note { color: var(--muted); font-size: 0.88rem; }
    .badge {
        display: inline-block;
        border-radius: 999px;
        padding: 5px 10px;
        font-size: .78rem;
        font-weight: 800;
        border: 1px solid var(--line);
        background: #fff;
        margin: 0 7px 7px 0;
    }
    .badge-green { color: var(--green); background: var(--soft-green); border-color: #bbf7d0; }
    .badge-blue { color: var(--blue); background: var(--soft-blue); border-color: #bfdbfe; }
    .badge-amber { color: var(--amber); background: var(--soft-amber); border-color: #fde68a; }
    .badge-red { color: var(--red); background: var(--soft-red); border-color: #fecaca; }
    .caption-tight { color: var(--muted); font-size: .85rem; margin-top: -8px; }
    h2, h3 { color: var(--ink); }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# Formatting and display helpers
# -----------------------------

def _fmt(value: Any, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "N/A"
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return str(value)


def _fmt_pct(value: Any, digits: int = 1) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "N/A"
    try:
        value = float(value)
        if value <= 1:
            value *= 100
        return f"{value:.{digits}f}%"
    except Exception:
        return str(value)


def _fmt_money(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "N/A"
    try:
        return f"${float(value):,.0f}"
    except Exception:
        return str(value)


def _sample_warning_label(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if isinstance(value, str) and value.strip().lower() in {"", "false", "0", "no", "none", "nan"}:
        return ""
    return "Small sample size: interpret carefully."


def _metric_row(metrics: dict[str, Any]) -> None:
    active = metrics.get("active_threshold_metrics", {})
    cols = st.columns(6)
    cols[0].metric("AUC", _fmt(metrics.get("auc")))
    cols[1].metric("PR-AUC", _fmt(metrics.get("pr_auc") or metrics.get("average_precision")))
    cols[2].metric("Recall", _fmt(active.get("recall", metrics.get("recall"))))
    cols[3].metric("Precision", _fmt(active.get("precision", metrics.get("precision"))))
    cols[4].metric("F1", _fmt(active.get("f1", metrics.get("f1"))))
    cols[5].metric("Business Cost", _fmt_money(active.get("business_cost", metrics.get("business_cost"))))


def _show_artifact_bootstrap(artifact: dict[str, Any] | None) -> bool:
    if artifact:
        return True
    st.warning("Champion model artifacts are missing. Build the demo artifacts before using the dashboard.")
    if st.button("Build Demo Artifacts", type="primary"):
        with st.spinner("Running validation, feature engineering, training, evaluation, drift, fairness, and model-card steps..."):
            report = run_local_retrain_pipeline()
        st.code(json.dumps(report, indent=2), language="json")
        st.rerun()
    return False


def _flatten_fairness(report: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for dimension in report.get("dimensions", []):
        for group, metrics in dimension.get("groups", {}).items():
            rows.append({"feature": dimension.get("feature"), "group": group, **metrics})
    return pd.DataFrame(rows)


def _selected_employee_profile(raw_data: pd.DataFrame) -> dict[str, Any]:
    if "EmployeeNumber" in raw_data.columns:
        employee_numbers = raw_data["EmployeeNumber"].astype(str).tolist()
        selected = st.selectbox("Employee", employee_numbers)
        row = raw_data[raw_data["EmployeeNumber"].astype(str) == selected].iloc[0]
    else:
        index = st.number_input("Row", min_value=0, max_value=max(len(raw_data) - 1, 0), value=0)
        row = raw_data.iloc[int(index)]
    return row.drop(labels=["Attrition"], errors="ignore").to_dict()


def _image_if_exists(path: str | None, caption: str, width: int = 850) -> None:
    if not path:
        return
    image_path = Path(path)
    if image_path.exists():
        st.image(str(image_path), caption=caption, width=width)


def _plotly(fig: Any, *, height: int | None = None) -> None:
    try:
        fig.update_layout(
            height=height,
            margin=dict(l=20, r=20, t=42, b=30),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(size=12),
        )
    except Exception:
        pass
    debug_mode = os.getenv("STREAMLIT_DEBUG_MODE", "0") == "1"
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": debug_mode, "responsive": True})


def _info_callout(text: str, kind: str = "callout") -> None:
    st.markdown(f'<div class="callout {kind}">{text}</div>', unsafe_allow_html=True)


def _model_card_summary(model_card: dict[str, Any]) -> None:
    if not model_card:
        st.info("Run the local retraining pipeline to generate the model card.")
        return

    metadata_rows, metadata_details = model_card_metadata_rows(model_card)
    st.markdown("#### Model card metadata")
    metadata_df = pd.DataFrame(metadata_rows).rename(columns={"metadata": "Metadata", "value": "Value"})
    st.markdown(metadata_df.to_html(index=False, escape=True, classes="metadata-table"), unsafe_allow_html=True)
    with st.expander("Advanced: full metadata paths/details", expanded=False):
        st.json(metadata_details)

    st.markdown("#### Governance narrative")
    preprocessing = model_card.get("preprocessing", {})
    narrative_rows = []
    if model_card.get("description"):
        narrative_rows.append({"Area": "Description", "Details": model_card.get("description")})
    if model_card.get("intended_use"):
        narrative_rows.append({"Area": "Intended use", "Details": model_card.get("intended_use")})
    if preprocessing.get("leakage_control"):
        narrative_rows.append({"Area": "Leakage control", "Details": preprocessing.get("leakage_control")})
    if preprocessing.get("one_hot_encoding"):
        narrative_rows.append(
            {
                "Area": "One-hot encoding",
                "Details": preprocessing["one_hot_encoding"].get("binary_indicators"),
            }
        )
    if narrative_rows:
        st.dataframe(pd.DataFrame(narrative_rows), use_container_width=True, hide_index=True)

    col_a, col_b = st.columns(2)
    with col_a:
        not_for_use = model_card.get("not_for_use", [])
        st.markdown("#### Not for use")
        if not_for_use:
            for item in not_for_use:
                st.markdown(f"- {item}")
        else:
            st.caption("No not-for-use section found.")
    with col_b:
        limitations = model_card.get("limitations", [])
        st.markdown("#### Limitations")
        if limitations:
            for item in limitations:
                st.markdown(f"- {item}")
        else:
            st.caption("No limitation section found.")

    decision_policy = model_card.get("decision_policy", {})
    if decision_policy:
        st.markdown("#### Decision policy")
        policy_rows = [
            {"Policy item": "Active production threshold", "Value": decision_policy.get("active_production_threshold")},
            {"Policy item": "Best F1 threshold", "Value": decision_policy.get("best_f1_threshold")},
            {"Policy item": "Best recall threshold", "Value": decision_policy.get("best_recall_threshold")},
            {"Policy item": "Best business-cost threshold", "Value": decision_policy.get("best_business_cost_threshold")},
            {"Policy item": "Business-cost formula", "Value": decision_policy.get("business_cost_formula")},
            {"Policy item": "False positive cost", "Value": decision_policy.get("false_positive_cost")},
            {"Policy item": "False negative cost", "Value": decision_policy.get("false_negative_cost")},
        ]
        st.dataframe(pd.DataFrame(policy_rows), use_container_width=True, hide_index=True)
        with st.expander("Advanced: raw decision-policy JSON", expanded=False):
            st.json(decision_policy)

    with st.expander("Advanced: raw model card JSON", expanded=False):
        st.json(model_card)


def _threshold_consistency_note(artifact: dict[str, Any], candidate_data: pd.DataFrame) -> None:
    active = artifact.get("decision_threshold")
    if active is None or candidate_data.empty or "model_name" not in candidate_data.columns:
        return
    candidate_data = _normalized_candidate_data(candidate_data)
    model_name = artifact.get("model_name")
    rows = candidate_data[candidate_data["model_name"].astype(str).str.lower() == str(model_name).lower()]
    if rows.empty or "candidate_selected_threshold" not in rows.columns:
        return
    candidate_threshold = rows.iloc[0].get("candidate_selected_threshold")
    try:
        mismatch = abs(float(active) - float(candidate_threshold)) > 1e-9
    except Exception:
        mismatch = str(active) != str(candidate_threshold)
    kind = "warn" if mismatch else "callout"
    _info_callout(
        f"<strong>Threshold policy:</strong> active production threshold is <code>{_fmt(active, 3)}</code>. "
        f"The candidate-comparison threshold for <code>{model_name}</code> is "
        f"<code>{_fmt(candidate_threshold, 3)}</code> and is labeled as <code>candidate_selected_threshold</code>. "
        "Best F1, best recall, and best business-cost thresholds are diagnostics; the API and dashboard decisions use the active production threshold.",
        kind,
    )


def _mask_long_payload_columns(df: pd.DataFrame) -> pd.DataFrame:
    return shorten_payload_columns(df)


def _normalized_candidate_data(data: pd.DataFrame) -> pd.DataFrame:
    return normalize_candidate_comparison_columns(data)


def _active_threshold_metrics(artifact: dict[str, Any]) -> dict[str, Any]:
    try:
        _, X_test, _, y_test, _ = load_training_data()
        probabilities = _positive_class_scores(artifact["model"], X_test)
        return metrics_at_threshold(
            y_test,
            probabilities,
            float(artifact.get("decision_threshold") or artifact.get("threshold") or 0.5),
            model_name=artifact.get("model_name", "champion"),
        )
    except Exception:
        return active_threshold_metrics_for_display(artifact)


# -----------------------------
# Data loading
# -----------------------------

artifacts = load_dashboard_artifacts()
artifact = artifacts["artifact"]

with st.sidebar:
    st.title("Governance Console")
    st.caption("HR attrition MLOps demo")
    if artifact:
        st.markdown(f'<span class="badge badge-green">Champion: {artifact.get("model_name", "unknown")}</span>', unsafe_allow_html=True)
        st.markdown(f'<span class="badge badge-blue">Threshold: {_fmt(artifact.get("decision_threshold"), 2)}</span>', unsafe_allow_html=True)
        st.caption(f"Last trained: {str(artifact.get('trained_at_utc', 'N/A'))[:19]}")
    st.divider()
    if st.button("Run Full Local Retrain", type="primary"):
        with st.spinner("Running local retraining pipeline..."):
            retrain_report = run_local_retrain_pipeline()
        st.code(json.dumps(retrain_report, indent=2), language="json")
        st.rerun()
    st.caption("Use retraining only after code/data changes. It may overwrite local artifacts.")

st.markdown(
    """
    <div class="hero">
      <h1>End-to-End MLOps & Model Governance Platform</h1>
      <p>Attrition risk modeling with leakage controls, tuned thresholds, drift monitoring, explainability, fairness review, and audit logging.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not _show_artifact_bootstrap(artifact):
    st.stop()

metrics = artifact.get("metrics", {})
raw_data = artifacts["raw_data"]
candidate_data = _normalized_candidate_data(artifacts["candidate_comparison"])
threshold_data = artifacts["threshold_metrics"]
current_predictions = artifacts["current_predictions"]
drift_alert = artifacts["drift_alert"]
fairness_report = artifacts["fairness_report"]
active_metrics = _active_threshold_metrics(artifact)

# Read CSV imported above is intentionally kept available because some project variants use it in app.py.
_ = read_csv

tabs = st.tabs(
    [
        "Executive Summary",
        "Model Performance",
        "Model Registry",
        "Drift Monitor",
        "Explainability",
        "Fairness & Ethics",
        "Audit Trail",
    ]
)

with tabs[0]:
    st.subheader("Champion Overview")
    _metric_row({**metrics, "active_threshold_metrics": active_metrics})
    _threshold_consistency_note(artifact, candidate_data)
    _info_callout(
        "<strong>Business-cost formula:</strong> false positives cost unnecessary intervention time, "
        "but false negatives are more expensive because missed attrition risk can mean losing employees without intervention. "
        "The dashboard uses <code>business_cost = false_positive_cost * FP + false_negative_cost * FN</code> at the active production threshold.",
        "callout",
    )

    left, right = st.columns([1.08, 0.92])
    with left:
        st.markdown("#### Operational risk distribution")
        _plotly(risk_distribution_chart(current_predictions), height=430)
        st.caption("Distribution of predicted attrition probabilities for the current employee scoring batch.")
    with right:
        st.markdown("#### Governance status")
        governance_df = pd.DataFrame(
            [
                {"Check": "Leakage control", "Status": "EmployeeNumber excluded from features"},
                {"Check": "One-hot encoding", "Status": "Both binary levels retained for readability; no identifiers or target-derived fields encoded"},
                {"Check": "Calibration", "Status": artifact.get("calibration", {}).get("status", "not_available")},
                {"Check": "Drift", "Status": "alert" if drift_alert.get("retraining_required") else "ok"},
                {"Check": "Fairness", "Status": "available" if fairness_report else "not_available"},
                {"Check": "Last trained", "Status": artifact.get("trained_at_utc")},
            ]
        )
        st.dataframe(governance_df, use_container_width=True, hide_index=True)
        row_count = len(raw_data) if not raw_data.empty else 0
        st.metric("Dataset rows", f"{row_count:,}" if row_count else "N/A")
        _info_callout(
            "<strong>Governance capability summary:</strong> This dashboard demonstrates model governance controls across threshold policy, leakage control, drift monitoring, explainability, fairness diagnostics, model registry metadata, and prediction audit logs.",
            "ok",
        )

with tabs[1]:
    st.subheader("Model Performance & Threshold Policy")
    c1, c2, c3 = st.columns([1, 1, 1])
    c1.metric("Champion", artifact.get("model_name", "N/A"))
    c2.metric("Active threshold", _fmt(artifact.get("decision_threshold"), 2))
    c3.metric("Active business cost", _fmt_money(active_metrics.get("business_cost", metrics.get("business_cost"))))

    _threshold_consistency_note(artifact, candidate_data)
    threshold_policy = artifact.get("threshold_policy", {})
    threshold_rows = [
        {"Threshold": "Active production threshold", "Value": _fmt(artifact.get("decision_threshold"), 3), "Meaning": "Used by FastAPI, audit logs, dashboard risk decisions, active business cost, and the dynamic confusion matrix."},
        {"Threshold": "Best F1 threshold", "Value": _fmt(threshold_policy.get("best_f1_threshold") or artifact.get("thresholds", {}).get("best_f1_threshold"), 3), "Meaning": "Diagnostic threshold that maximizes F1 on the holdout split."},
        {"Threshold": "Best recall threshold", "Value": _fmt(threshold_policy.get("best_recall_threshold") or artifact.get("thresholds", {}).get("best_recall_threshold"), 3), "Meaning": "Diagnostic threshold that catches the most attrition cases."},
        {"Threshold": "Best business-cost threshold", "Value": _fmt(threshold_policy.get("best_business_cost_threshold") or artifact.get("thresholds", {}).get("best_business_cost_threshold"), 3), "Meaning": "Diagnostic threshold minimizing FP/FN cost on the holdout split."},
        {"Threshold": "Candidate selected threshold", "Value": "per candidate", "Meaning": "Stored as candidate_selected_threshold in the comparison table; not automatically the active production threshold."},
    ]
    st.dataframe(pd.DataFrame(threshold_rows), use_container_width=True, hide_index=True)

    st.markdown("#### Candidate model ranking")
    _info_callout("Candidate rows use candidate_business_cost and candidate_selected_threshold. Lower candidate business cost is better. PR-AUC matters because attrition-positive cases are usually the minority class.")
    _plotly(candidate_comparison_chart(candidate_data), height=420)
    if not candidate_data.empty:
        st.dataframe(candidate_data, use_container_width=True, hide_index=True)

    st.markdown("#### Threshold tradeoffs")
    selected_model = artifact.get("model_name")
    _plotly(threshold_chart(threshold_data, selected_model), height=440)
    _info_callout(
        "API predictions, audit logs, risk labels, and active business cost use the active production threshold. "
        "False-negative cost is intentionally higher because missed attrition risk can mean losing employees without intervention. "
        "False-positive cost represents unnecessary retention outreach.",
        "callout",
    )

    calibration_path = artifact.get("calibration", {}).get("plot_path")
    evaluation_snapshot = artifacts.get("evaluation_snapshot", {})
    training_counts = confusion_counts_from_matrix(
        evaluation_snapshot.get("metrics", {})
        .get("confusion_matrix", {})
        .get("confusion_matrix")
    )
    show_training_matrix = should_show_training_confusion_artifact(active_metrics, training_counts)

    if show_training_matrix:
        matrix_cols = st.columns(2)
    else:
        matrix_cols = st.columns([0.2, 0.6, 0.2])

    with matrix_cols[0 if show_training_matrix else 1]:
        st.markdown("#### Active production confusion matrix")
        _plotly(
            confusion_matrix_chart(active_metrics, title=f"Active Threshold Confusion Matrix ({_fmt(artifact.get('decision_threshold'), 2)})"),
            height=360,
        )
        st.caption("Rows: Actual No, Actual Yes. Columns: Predicted No, Predicted Yes. Counts are recomputed for the active production threshold.")

    if show_training_matrix:
        with matrix_cols[1]:
            st.markdown("#### Training artifact confusion matrix")
            with st.expander("Static training artifact confusion matrix", expanded=True):
                st.caption("Shown because the static training artifact differs from the active-threshold matrix.")
                _image_if_exists("reports/performance_logs/confusion_matrix.png", "Static training artifact confusion matrix", width=520)
    else:
        _info_callout(
            "Training artifact matches the active-threshold matrix, so the duplicate static confusion image is hidden.",
            "ok",
        )

    st.markdown("#### Calibration curve")
    cal_left, cal_mid, cal_right = st.columns([0.18, 0.64, 0.18])
    with cal_mid:
        with st.container():
            st.caption("Shows how close predicted probabilities are to observed outcomes. Lower Brier score is better.")
            _image_if_exists(calibration_path, "Calibration curve", width=560)

with tabs[2]:
    st.subheader("Model Registry")
    versions = pd.DataFrame(registry_versions())
    if versions.empty:
        st.info("No registry rows available yet.")
    else:
        st.dataframe(versions, use_container_width=True, hide_index=True)

    st.subheader("Model Card")
    _model_card_summary(artifacts["model_card"])

with tabs[3]:
    st.subheader("Drift Simulation")
    cols = st.columns([1.15, 1.15, 0.75, 0.75])
    scenario_options = ["baseline", "compensation_shift", "workload_pressure", "hiring_mix"]
    current_alert_scenario = str(drift_alert.get("scenario", "baseline"))
    default_scenario_index = scenario_options.index(current_alert_scenario) if current_alert_scenario in scenario_options else 0
    scenario = cols[0].selectbox("Scenario", scenario_options, index=default_scenario_index)
    intensity = cols[1].slider("Intensity", min_value=0.05, max_value=0.8, value=0.25, step=0.05)
    if scenario == "baseline":
        cols[1].caption("Baseline simulates normal sampling variation; non-baseline scenarios apply synthetic shifts.")
    if cols[2].button("Simulate", type="primary"):
        simulate_drift(scenario, intensity)
        run_drift_detection()
        st.rerun()
    if cols[3].button("Reset"):
        reset_current_data()
        run_drift_detection()
        st.rerun()

    drift_alert = json.loads(DRIFT_ALERT_PATH.read_text(encoding="utf-8")) if DRIFT_ALERT_PATH.exists() else {}
    drift_threshold = float(drift_alert.get("threshold", 0.30))
    drift_summary = drift_status_summary(
        drift_alert.get("feature_drift_scores", {}),
        drift_threshold,
    )
    status = drift_alert.get("status") or drift_summary["status"]
    metric_cols = st.columns(4)
    metric_cols[0].metric("Overall Drift Score", _fmt(drift_alert.get("drift_score")))
    metric_cols[1].metric("Max Feature Drift", _fmt(drift_alert.get("max_feature_drift_score", drift_summary["max_feature_drift_score"])))
    metric_cols[2].metric("Features Above Threshold", drift_alert.get("features_above_threshold_count", drift_summary["features_above_threshold_count"]))
    metric_cols[3].metric("Status", status)
    _info_callout(
        f"Current scored-batch scenario: <code>{drift_alert.get('scenario', scenario)}</code>. "
        "Overall drift score is an average across monitored features. Max feature drift is the largest individual feature score. "
        "Status becomes Review Required when any feature exceeds the configured feature threshold.",
        "callout" if status != "Review Required" else "warn",
    )
    _plotly(drift_feature_chart(drift_alert), height=430)
    report_paths = drift_alert.get("reports", {})
    if report_paths:
        with st.expander("Generated drift report paths", expanded=False):
            st.json(report_paths)

with tabs[4]:
    st.subheader("Employee-Level Explanation")
    if raw_data.empty:
        st.info("Raw dataset not available.")
    else:
        profile = _selected_employee_profile(raw_data)
        prediction = predict_employee(profile, write_audit=False, artifact=artifact)
        cols = st.columns(4)
        cols[0].metric("Attrition Probability", _fmt_pct(prediction["attrition_probability"]))
        cols[1].metric("Risk Level", prediction["risk_level"])
        cols[2].metric("Decision Threshold", _fmt(prediction["decision_threshold"], 2))
        cols[3].metric("Model", prediction.get("model_name") or "unknown")

        _info_callout(f"<strong>Recommended HR action:</strong> {prediction['recommended_hr_action']}", "callout")

        drivers = pd.DataFrame(prediction.get("top_3_shap_drivers", []))
        if not drivers.empty:
            st.markdown("#### Local explanation drivers")
            display_columns = [
                column
                for column in [
                    "feature",
                    "display_value",
                    "profile_value",
                    "shap_impact",
                    "impact",
                    "direction",
                    "plain_english_meaning",
                ]
                if column in drivers.columns
            ]
            driver_view = drivers[display_columns].copy()
            if "profile_value" in driver_view.columns and "display_value" not in driver_view.columns:
                driver_view = driver_view.rename(columns={"profile_value": "display_value"})
            if "impact" in driver_view.columns and "shap_impact" not in driver_view.columns:
                driver_view = driver_view.rename(columns={"impact": "shap_impact"})
            if "shap_impact" in driver_view.columns:
                driver_view = driver_view.rename(columns={"shap_impact": "model_margin_impact"})
            st.dataframe(driver_view, use_container_width=True, hide_index=True)
            st.caption("The driver table uses model_margin_impact: model-margin/local contribution scores, not direct probability-point changes. Positive values push the model score upward; negative values push it downward.")

        with st.expander("Global feature importance image", expanded=False):
            st.caption("Global importance explains model behavior overall. Local drivers above explain the selected employee.")
            _image_if_exists(artifact.get("shap_plot_path"), "Global feature importance", width=820)
        with st.expander("Raw employee profile used for scoring", expanded=False):
            st.json(profile)

with tabs[5]:
    st.subheader("Fairness & Ethics")
    if st.button("Generate Fairness Report", type="primary"):
        with st.spinner("Scoring holdout groups..."):
            fairness_report = generate_fairness_report()
        st.rerun()

    fairness_report = json.loads(FAIRNESS_REPORT_PATH.read_text(encoding="utf-8")) if FAIRNESS_REPORT_PATH.exists() else {}
    if not fairness_report:
        st.info("Fairness report has not been generated yet.")
    else:
        ethics_note = fairness_report.get("ethics_note") or (
            "This dataset is a small public HR sample. Fairness diagnostics are for governance review and should not justify automated employment decisions."
        )
        _info_callout(f"<strong>Ethics note:</strong> {ethics_note}", "warn")
        _info_callout(
            "<strong>Governance recommendation:</strong> Use this model only as HR decision support. "
            "Do not use it for automated employment decisions. Review dimensions marked <code>needs_review</code> before production rollout.",
            "callout",
        )

        flagged_dimensions = fairness_report.get("flagged_dimensions", [])
        flagged_table = pd.DataFrame(
            fairness_report.get("flagged_dimensions_table") or flagged_dimensions
        )
        flat_fairness = _flatten_fairness(fairness_report)
        overview_cols = st.columns(4)
        overview_cols[0].metric("Dimensions reviewed", len(fairness_report.get("dimensions", [])))
        overview_cols[1].metric("Needs review", int((flagged_table.get("review_status", pd.Series(dtype=str)) == "needs_review").sum()) if not flagged_table.empty else len(flagged_dimensions))
        overview_cols[2].metric("Groups scored", 0 if flat_fairness.empty else len(flat_fairness))
        overview_cols[3].metric("Sample warnings", int(flagged_table.get("sample_warning", pd.Series(dtype=bool)).fillna(False).sum()) if not flagged_table.empty else 0)

        if not flagged_table.empty:
            st.markdown("#### Fairness review summary")
            flagged_display = flagged_table.copy()
            if "sample_warning" in flagged_display.columns:
                flagged_display["sample_warning_text"] = flagged_display["sample_warning"].apply(_sample_warning_label)
            display_cols = [
                column
                for column in [
                    "feature",
                    "max_high_risk_rate_gap",
                    "max_recall_gap",
                    "review_status",
                    "sample_warning_text",
                ]
                if column in flagged_display.columns
            ]
            st.dataframe(flagged_display[display_cols], use_container_width=True, hide_index=True)

        dimension_names = [item["feature"] for item in fairness_report.get("dimensions", [])]
        if dimension_names:
            selected_dimension = st.selectbox("Dimension", dimension_names)
            _plotly(fairness_group_chart(fairness_report, selected_dimension), height=430)
        if not flat_fairness.empty:
            st.markdown("#### Group-level fairness table")
            if "sample_warning" in flat_fairness.columns and flat_fairness["sample_warning"].fillna(False).any():
                _info_callout("Small sample size: interpret this group carefully.", "warn")
            flat_fairness_display = flat_fairness.copy()
            if "sample_warning" in flat_fairness_display.columns:
                flat_fairness_display["sample_warning_text"] = flat_fairness_display["sample_warning"].apply(_sample_warning_label)
                flat_fairness_display = flat_fairness_display.drop(columns=["sample_warning"])
            st.dataframe(flat_fairness_display, use_container_width=True, hide_index=True)
        with st.expander("Advanced: raw fairness report JSON", expanded=False):
            st.json(fairness_report)

with tabs[6]:
    st.subheader("Prediction Audit Trail")
    filter_cols = st.columns([1, 1, 1])
    row_limit = filter_cols[0].slider("Row limit", min_value=25, max_value=500, value=250, step=25)
    audit_entries = pd.DataFrame(read_audit_entries(limit=row_limit))
    if audit_entries.empty:
        st.info("No prediction audit entries yet. Use the API `/predict` endpoint to create audited predictions.")
    else:
        if "timestamp_utc" in audit_entries:
            audit_entries["timestamp_utc"] = pd.to_datetime(audit_entries["timestamp_utc"], errors="coerce", utc=True)
            valid_dates = audit_entries["timestamp_utc"].dropna()
            if not valid_dates.empty:
                start_default = valid_dates.min().date()
                end_default = valid_dates.max().date()
                date_range = filter_cols[1].date_input("Date range", value=(start_default, end_default))
                if isinstance(date_range, tuple) and len(date_range) == 2:
                    start_date, end_date = date_range
                    audit_entries = audit_entries[
                        audit_entries["timestamp_utc"].dt.date.between(start_date, end_date)
                    ]
        if "risk_level" in audit_entries:
            risks = sorted(audit_entries["risk_level"].dropna().unique().tolist())
            selected_risks = filter_cols[2].multiselect("Risk level", risks, default=risks)
            if selected_risks:
                audit_entries = audit_entries[audit_entries["risk_level"].isin(selected_risks)]
        for expected_col in [
            "audit_id",
            "timestamp_utc",
            "employee_id",
            "employee_id_hash",
            "attrition_probability",
            "risk_level",
            "recommended_hr_action",
            "request_payload",
            "response_payload",
        ]:
            if expected_col not in audit_entries.columns:
                audit_entries[expected_col] = None
        audit_export = serialize_audit_dataframe(audit_entries)
        st.caption("Payloads are shortened in the table. The CSV export and row detail keep the full filtered payloads.")
        st.dataframe(_mask_long_payload_columns(audit_export), use_container_width=True, hide_index=True)
        st.download_button(
            "Download Audit CSV",
            data=audit_export.to_csv(index=False),
            file_name="prediction_audit_log.csv",
            mime="text/csv",
        )
        with st.expander("Audit row detail", expanded=False):
            if len(audit_export):
                fallback_labels = pd.Series(audit_export.index.astype(str), index=audit_export.index)
                row_labels = audit_export["audit_id"].fillna(fallback_labels).astype(str).tolist()
                selected_audit = st.selectbox("Audit row", row_labels)
                selected_index = row_labels.index(selected_audit)
                st.json(audit_row_for_detail(audit_export.iloc[selected_index].fillna("")))
        with st.expander("Advanced: full filtered audit rows", expanded=False):
            st.json([audit_row_for_detail(row) for row in audit_export.fillna("").to_dict(orient="records")])
