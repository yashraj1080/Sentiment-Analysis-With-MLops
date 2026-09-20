"""
Unit and integration tests for FastAPI REST API endpoints.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.app import app


@pytest.fixture(scope="module")
def client():
    """Create a TestClient that triggers FastAPI lifespan event handlers."""
    with TestClient(app) as test_client:
        yield test_client


def test_root_endpoint(client: TestClient):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "Sentiment Analysis MLOps Pipeline"
    assert data["status"] == "operational"
    assert "/docs" in data["docs_url"]


def test_health_endpoint(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert "uptime_seconds" in data
    assert data["model_version"] != ""


def test_predict_endpoint_positive(client: TestClient):
    payload = {
        "text": "An absolute masterpiece of cinema! Outstanding performances and thrilling directing."
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["sentiment"] == "positive"
    assert data["confidence"] >= 0.5
    assert "latency_ms" in data
    assert data["latency_ms"] >= 0.0
    assert "probabilities" in data
    assert "positive" in data["probabilities"]
    assert "negative" in data["probabilities"]


def test_predict_endpoint_negative(client: TestClient):
    payload = {
        "text": "Terrible and utterly boring. Horrible acting and a complete waste of my time."
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["sentiment"] == "negative"
    assert data["confidence"] >= 0.5
    assert data["probabilities"]["negative"] > data["probabilities"]["positive"]


def test_predict_batch_endpoint(client: TestClient):
    payload = {
        "texts": [
            "Great product, works like a charm! Five stars.",
            "Horrible experience, completely broken on arrival.",
            "Loved the whole movie, brilliantly written script!",
        ]
    }
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_processed"] == 3
    assert len(data["results"]) == 3
    assert data["results"][0]["sentiment"] == "positive"
    assert data["results"][1]["sentiment"] == "negative"
    assert data["results"][2]["sentiment"] == "positive"
    assert data["total_latency_ms"] > 0.0


def test_predict_validation_error_empty_string(client: TestClient):
    payload = {"text": ""}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Pydantic validation error


def test_predict_validation_error_missing_field(client: TestClient):
    payload = {"invalid_key": "Some review"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_metrics_endpoint(client: TestClient):
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert data["total_requests"] > 0
    assert data["total_predictions"] > 0
    assert "average_latency_ms" in data
    assert "uptime_seconds" in data
