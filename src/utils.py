"""
Shared utility functions for the AQI Predictor project.

Provides logging setup, date helpers, and common data operations.
"""

import logging
import sys
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Create a configured logger with console output."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(
            "%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def get_date_range(
    start_date: str,
    end_date: Optional[str] = None,
    fmt: str = "%Y-%m-%d",
) -> list[datetime]:
    """Generate a list of dates between start_date and end_date (inclusive)."""
    start = datetime.strptime(start_date, fmt)
    end = datetime.strptime(end_date, fmt) if end_date else datetime.now()
    dates = []
    current = start
    while current <= end:
        dates.append(current)
        current += timedelta(days=1)
    return dates


def classify_aqi(aqi_value: float) -> str:
    """Classify an AQI value into its EPA category."""
    from src.config import AQI_CATEGORIES

    for category, (low, high) in AQI_CATEGORIES.items():
        if low <= aqi_value <= high:
            return category
    return "Hazardous" if aqi_value > 500 else "Unknown"


def pm25_to_aqi(pm25_val: float) -> int:
    """Convert PM2.5 concentration (ug/m3) to US EPA AQI."""
    breakpoints = [
        (0.0, 12.0, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 350.4, 301, 400),
        (350.5, 500.4, 401, 500)
    ]
    pm25_val = round(pm25_val, 1)
    for c_low, c_high, i_low, i_high in breakpoints:
        if c_low <= pm25_val <= c_high:
            return int(round(((i_high - i_low) / (c_high - c_low)) * (pm25_val - c_low) + i_low))
    if pm25_val > 500.4:
        return 500
    return 0


def add_time_features(df: pd.DataFrame, datetime_col: str = "datetime") -> pd.DataFrame:
    """Add time-based features (hour, day_of_week, month) to a DataFrame."""
    df = df.copy()
    dt = pd.to_datetime(df[datetime_col])
    df["hour"] = dt.dt.hour
    df["day_of_week"] = dt.dt.dayofweek
    df["month"] = dt.dt.month
    df["is_weekend"] = dt.dt.dayofweek.isin([5, 6]).astype(int)
    return df
