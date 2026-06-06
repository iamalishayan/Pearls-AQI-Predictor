"""Tests for the feature pipeline utilities."""

from src.utils import classify_aqi, pm25_to_aqi, add_time_features
import pandas as pd


# ── classify_aqi ──────────────────────────────────────────────────────────────

def test_classify_aqi_good():
    assert classify_aqi(25) == "Good"


def test_classify_aqi_moderate():
    assert classify_aqi(75) == "Moderate"


def test_classify_aqi_unhealthy():
    assert classify_aqi(175) == "Unhealthy"


def test_classify_aqi_hazardous():
    assert classify_aqi(350) == "Hazardous"


def test_classify_aqi_above_500():
    assert classify_aqi(600) == "Hazardous"


# ── pm25_to_aqi ──────────────────────────────────────────────────────────────

def test_pm25_to_aqi_good():
    """PM2.5 of 5.0 should map to AQI in the 'Good' range (0-50)."""
    aqi = pm25_to_aqi(5.0)
    assert 0 <= aqi <= 50


def test_pm25_to_aqi_moderate():
    """PM2.5 of 20.0 should map to AQI in the 'Moderate' range (51-100)."""
    aqi = pm25_to_aqi(20.0)
    assert 51 <= aqi <= 100


def test_pm25_to_aqi_unhealthy():
    """PM2.5 of 100.0 should map to AQI in the 'Unhealthy' range (151-200)."""
    aqi = pm25_to_aqi(100.0)
    assert 151 <= aqi <= 200


def test_pm25_to_aqi_max():
    """PM2.5 above 500.4 should cap at AQI 500."""
    aqi = pm25_to_aqi(999.0)
    assert aqi == 500


# ── add_time_features ────────────────────────────────────────────────────────

def test_add_time_features():
    """Verify time features are correctly extracted."""
    df = pd.DataFrame({
        "datetime": pd.to_datetime(["2026-06-05 14:00:00", "2026-06-07 22:00:00"])
    })
    result = add_time_features(df)
    assert "hour" in result.columns
    assert "day_of_week" in result.columns
    assert "month" in result.columns
    assert "is_weekend" in result.columns
    assert result.iloc[0]["hour"] == 14
    assert result.iloc[0]["month"] == 6
    # June 7, 2026 is a Sunday (dayofweek=6), so is_weekend should be 1
    assert result.iloc[1]["is_weekend"] == 1
