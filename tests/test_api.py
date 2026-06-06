"""Tests for the FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient
from src.app.api import app


client = TestClient(app)


def test_root_returns_200():
    """Root endpoint should return service info."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "service" in data
    assert "endpoints" in data


def test_health_returns_200():
    """Health endpoint should return model metadata."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "model_name" in data
    assert "city" in data
    assert data["city"] == "Islamabad"
