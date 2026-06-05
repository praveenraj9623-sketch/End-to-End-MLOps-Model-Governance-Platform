"""Plotly figures for the Streamlit dashboard."""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.dashboard.governance_logic import standard_confusion_matrix


def candidate_comparison_chart(data: pd.DataFrame) -> go.Figure:
    if data.empty:
        return go.Figure()
    frame = data.copy()
    if "candidate_business_cost" not in frame and "business_cost" in frame:
        frame["candidate_business_cost"] = frame["business_cost"]
    if "candidate_selected_threshold" not in frame and "selected_threshold" in frame:
        frame["candidate_selected_threshold"] = frame["selected_threshold"]
    fig = px.bar(
        frame.sort_values("candidate_business_cost"),
        x="model_name",
        y="candidate_business_cost",
        color="pr_auc" if "pr_auc" in frame else None,
        text="candidate_selected_threshold" if "candidate_selected_threshold" in frame else None,
        labels={
            "candidate_business_cost": "Candidate Business Cost",
            "candidate_selected_threshold": "Candidate Selected Threshold",
            "model_name": "Model",
            "pr_auc": "PR-AUC",
        },
    )
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10))
    return fig


def threshold_chart(data: pd.DataFrame, model_name: str | None = None) -> go.Figure:
    if data.empty:
        return go.Figure()
    frame = data.copy()
    if model_name and "model_name" in frame:
        frame = frame[frame["model_name"] == model_name]
    fig = go.Figure()
    for metric in ["precision", "recall", "f1"]:
        if metric in frame:
            fig.add_trace(go.Scatter(x=frame["threshold"], y=frame[metric], mode="lines+markers", name=metric.title()))
    if "business_cost" in frame:
        fig.add_trace(
            go.Scatter(
                x=frame["threshold"],
                y=frame["business_cost"],
                mode="lines+markers",
                name="Business Cost",
                yaxis="y2",
            )
        )
    fig.update_layout(
        height=380,
        margin=dict(l=10, r=10, t=30, b=10),
        yaxis=dict(title="Metric"),
        yaxis2=dict(title="Cost", overlaying="y", side="right"),
        xaxis=dict(title="Threshold"),
    )
    return fig


def drift_feature_chart(drift_alert: dict[str, Any]) -> go.Figure:
    items = drift_alert.get("top_drifted_features", [])
    if not items:
        return go.Figure()
    frame = pd.DataFrame(items, columns=["feature", "drift_score"])
    fig = px.bar(frame.sort_values("drift_score"), x="drift_score", y="feature", orientation="h")
    threshold = drift_alert.get("threshold")
    if threshold is not None:
        fig.add_vline(
            x=float(threshold),
            line_dash="dash",
            line_color="#dc2626",
            annotation_text="Feature threshold",
            annotation_position="top right",
        )
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10))
    return fig


def risk_distribution_chart(scored: pd.DataFrame) -> go.Figure:
    if scored.empty or "prediction_probability" not in scored:
        return go.Figure()
    fig = px.histogram(scored, x="prediction_probability", nbins=24, color="scenario" if "scenario" in scored else None)
    fig.update_layout(height=330, margin=dict(l=10, r=10, t=30, b=10), xaxis_title="Attrition Probability")
    return fig


def fairness_group_chart(report: dict[str, Any], feature: str) -> go.Figure:
    dimensions = {item["feature"]: item for item in report.get("dimensions", [])}
    dimension = dimensions.get(feature)
    if not dimension:
        return go.Figure()
    rows = []
    for group, metrics in dimension.get("groups", {}).items():
        rows.append(
            {
                "group": group,
                "high_risk_rate": metrics.get("high_risk_rate"),
                "recall": metrics.get("recall"),
                "false_negative_rate": metrics.get("false_negative_rate"),
                "sample_size": metrics.get("sample_size"),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return go.Figure()
    fig = px.bar(
        frame,
        x="group",
        y=["high_risk_rate", "recall", "false_negative_rate"],
        barmode="group",
        hover_data=["sample_size"],
    )
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10), yaxis_title="Rate")
    return fig


def calibration_curve_chart(points: pd.DataFrame, brier_score: float | None = None) -> go.Figure:
    """Build a calibration curve from holdout probability bins."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(color="#64748b", dash="dash"),
            name="Perfect calibration",
            hovertemplate="Predicted=%{x:.2f}<br>Observed=%{y:.2f}<extra></extra>",
        )
    )
    if not points.empty:
        fig.add_trace(
            go.Scatter(
                x=points["mean_predicted_probability"],
                y=points["observed_attrition_rate"],
                mode="lines+markers",
                line=dict(color="#2563eb", width=3),
                marker=dict(size=8),
                name="Model calibration",
                hovertemplate="Mean predicted=%{x:.3f}<br>Observed attrition=%{y:.3f}<extra></extra>",
            )
        )
    title = "Dynamic Calibration Curve"
    if brier_score is not None:
        title = f"{title} (Brier={brier_score:.3f})"
    fig.update_layout(
        title=title,
        height=360,
        margin=dict(l=10, r=10, t=45, b=10),
        xaxis=dict(title="Mean Predicted Probability", range=[0, 1]),
        yaxis=dict(title="Observed Attrition Rate", range=[0, 1]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def confusion_matrix_chart(counts: dict[str, int], title: str = "Active Threshold Confusion Matrix") -> go.Figure:
    """Build a compact active-threshold confusion-matrix heatmap."""
    matrix = standard_confusion_matrix(counts)
    labels = [["TN", "FP"], ["FN", "TP"]]
    text = [
        [f"{labels[row][col]}<br>{matrix[row][col]}" for col in range(2)]
        for row in range(2)
    ]
    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=["Predicted No", "Predicted Yes"],
            y=["Actual No", "Actual Yes"],
            text=text,
            texttemplate="%{text}",
            colorscale="Blues",
            showscale=False,
        )
    )
    fig.update_layout(
        title=title,
        height=360,
        margin=dict(l=10, r=10, t=45, b=10),
        xaxis=dict(title="Prediction"),
        yaxis=dict(title="Actual", autorange="reversed"),
    )
    return fig
