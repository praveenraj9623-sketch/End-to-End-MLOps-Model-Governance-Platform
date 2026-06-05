"""Calibration data helpers for dashboard rendering."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

from src.features.engineering import TARGET_COLUMN, split_raw_data
from src.models.train import _positive_class_scores
from src.services.prediction_service import raw_feature_columns


def calibration_points_from_artifact(
    raw_data: pd.DataFrame,
    artifact: dict[str, Any],
    *,
    n_bins: int = 8,
) -> tuple[pd.DataFrame, float | None]:
    """Return holdout calibration points from the active dashboard artifact."""
    if raw_data.empty or TARGET_COLUMN not in raw_data.columns or not artifact.get("model"):
        return pd.DataFrame(), None

    try:
        _, X_test, _, y_test = split_raw_data(raw_data)
        feature_columns = raw_feature_columns(artifact)
        probabilities = _positive_class_scores(artifact["model"], X_test[feature_columns])
        observed_rate, mean_probability = calibration_curve(
            y_test,
            probabilities,
            n_bins=n_bins,
            strategy="uniform",
        )
        frame = pd.DataFrame(
            {
                "mean_predicted_probability": mean_probability,
                "observed_attrition_rate": observed_rate,
            }
        )
        return frame, float(brier_score_loss(y_test, probabilities))
    except Exception:
        return pd.DataFrame(), None
