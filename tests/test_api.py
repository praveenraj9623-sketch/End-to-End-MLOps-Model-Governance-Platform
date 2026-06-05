"""Tests for FastAPI app."""

from fastapi.testclient import TestClient

from src.api.main import app
from src.features.engineering import TARGET_COLUMN, load_raw_data

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_endpoint_returns_governance_payload():
    profile = load_raw_data().drop(columns=[TARGET_COLUMN]).iloc[0].to_dict()
    response = client.post("/predict", json={"employee_profile": profile})
    payload = response.json()
    assert response.status_code == 200
    assert 0.0 <= payload["attrition_probability"] <= 1.0
    assert payload["risk_level"] in {"Low", "Medium", "High"}
    assert "recommended_hr_action" in payload
    assert "top_3_shap_drivers" in payload
