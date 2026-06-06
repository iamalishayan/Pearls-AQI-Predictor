"""Tests for the training pipeline utilities."""

import pandas as pd
import numpy as np
from src.config import FEATURE_GROUP_NAME, FEATURE_GROUP_VERSION, FORECAST_DAYS


def test_feature_group_config():
    """Feature group config should be set correctly."""
    assert FEATURE_GROUP_NAME == "aqi_features"
    assert isinstance(FEATURE_GROUP_VERSION, int)
    assert FEATURE_GROUP_VERSION >= 1


def test_forecast_days_config():
    """Forecast horizon should be 3 days."""
    assert FORECAST_DAYS == 3


def test_target_shift_logic():
    """Verify that shifting pm25 by N hours creates valid future targets."""
    # Create a simple time-ordered DataFrame
    dates = pd.date_range("2026-01-01", periods=100, freq="h")
    df = pd.DataFrame({
        "timestamp": dates,
        "pm25": np.random.uniform(10, 100, size=100),
    })

    # Shift by 24 hours (the target for +24h forecast)
    df["target_24h"] = df["pm25"].shift(-24)

    # First 76 rows should have valid targets, last 24 should be NaN
    assert df["target_24h"].isna().sum() == 24
    assert df["target_24h"].notna().sum() == 76
