"""
Tests for the inference module — unit tests with mocking (no Hopsworks required).
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from src.alerts import check_alerts
from src.utils import pm25_to_aqi, classify_aqi


# ── Alert System ─────────────────────────────────────────────────────────────

def test_alert_active_for_unhealthy_aqi():
    """Alert should fire when AQI > 150 (Unhealthy)."""
    alert = check_alerts(predicted_aqi=160)
    assert alert["is_active"] is True
    assert alert["level"] == "HIGH"


def test_alert_inactive_for_good_aqi():
    """Alert should NOT fire when AQI is Good."""
    alert = check_alerts(predicted_aqi=25)
    assert alert["is_active"] is False


def test_alert_active_for_hazardous_aqi():
    """Alert should fire with highest severity for Hazardous AQI."""
    alert = check_alerts(predicted_aqi=400)
    assert alert["is_active"] is True
    assert alert["level"] == "EMERGENCY"
    assert "recommended_action" in alert


# ── AQI Conversion Pipeline ─────────────────────────────────────────────────

def test_pm25_to_aqi_roundtrip():
    """Verify that pm25_to_aqi + classify_aqi produces consistent categories."""
    pm25_val = 45.0  # Should be in the "Unhealthy for Sensitive Groups" range
    aqi = pm25_to_aqi(pm25_val)
    category = classify_aqi(aqi)
    assert category == "Unhealthy for Sensitive Groups"


def test_pm25_zero():
    """PM2.5 of 0.0 should produce AQI 0."""
    assert pm25_to_aqi(0.0) == 0
