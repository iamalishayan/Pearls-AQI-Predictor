"""
Backfill Pipeline — Run once to populate historical training data for V3.

MLOps Pattern (no local files):
  API (Open-Meteo)  →  Vectorized Feature Engineering (Lags)  →  Hopsworks Feature Store V3
"""

import argparse
import warnings
from datetime import datetime, timedelta, timezone
import pandas as pd
import hopsworks
import openmeteo_requests
import requests_cache
from retry_requests import retry

from src.config import (
    CITY_NAME,
    CITY_LAT,
    CITY_LON,
    FEATURE_GROUP_NAME,
    FEATURE_GROUP_VERSION,
    HOPSWORKS_API_KEY,
    HOPSWORKS_PROJECT_NAME,
)
from src.feature_pipeline import engineer_features, get_feature_group
from src.utils import setup_logger

logger = setup_logger("backfill_pipeline")


def build_historical_df(days: int = 90) -> pd.DataFrame:
    """Fetch and merge historical weather and air quality data from Open-Meteo."""
    logger.info(f"Fetching {days} days of historical data from Open-Meteo...")
    
    # Setup Open-Meteo client
    cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    # We fetch slightly more than 'days' to ensure we have lags for the first day of our target range
    end_date = datetime.now(timezone.utc).date() - timedelta(days=1)
    start_date = end_date - timedelta(days=days + 1) # Extra 2 days for lags
    
    logger.info(f"Time range for fetch: {start_date} to {end_date}")

    # 1. Fetch Air Quality
    url_aq = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params_aq = {
        "latitude": CITY_LAT,
        "longitude": CITY_LON,
        "hourly": ["pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide", "ozone"],
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
    }
    response_aq = openmeteo.weather_api(url_aq, params=params_aq)[0]
    hourly_aq = response_aq.Hourly()
    
    date_range = pd.date_range(
        start=pd.to_datetime(hourly_aq.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly_aq.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly_aq.Interval()),
        inclusive="left"
    )
    
    df_aq = pd.DataFrame({"timestamp": date_range})
    df_aq["pm10"] = hourly_aq.Variables(0).ValuesAsNumpy()
    df_aq["pm25"] = hourly_aq.Variables(1).ValuesAsNumpy()
    df_aq["co"] = hourly_aq.Variables(2).ValuesAsNumpy()
    df_aq["no2"] = hourly_aq.Variables(3).ValuesAsNumpy()
    df_aq["so2"] = hourly_aq.Variables(4).ValuesAsNumpy()
    df_aq["o3"] = hourly_aq.Variables(5).ValuesAsNumpy()

    # 2. Fetch Weather Data
    url_wx = "https://archive-api.open-meteo.com/v1/archive"
    params_wx = {
        "latitude": CITY_LAT,
        "longitude": CITY_LON,
        "hourly": ["temperature_2m", "relative_humidity_2m", "surface_pressure", "wind_speed_10m", "precipitation"],
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
    }
    response_wx = openmeteo.weather_api(url_wx, params=params_wx)[0]
    hourly_wx = response_wx.Hourly()
    
    df_wx = pd.DataFrame({"timestamp": date_range})
    df_wx["temperature"] = hourly_wx.Variables(0).ValuesAsNumpy()
    df_wx["humidity"] = hourly_wx.Variables(1).ValuesAsNumpy()
    df_wx["pressure"] = hourly_wx.Variables(2).ValuesAsNumpy()
    df_wx["wind_speed"] = hourly_wx.Variables(3).ValuesAsNumpy()
    df_wx["rain"] = hourly_wx.Variables(4).ValuesAsNumpy()

    # Merge DataFrames
    df_raw = pd.merge(df_wx, df_aq, on="timestamp")
    df_raw["city_name"] = CITY_NAME
    
    logger.info(f"Raw data merged: {df_raw.shape[0]} rows")

    # Engineer Features (includes Lags)
    logger.info("Engineering features for the historical dataset (V3)...")
    df_processed = engineer_features(df_raw)
    
    # DROP NaNs created by shifts (First 24 hours of the fetch won't have 24h lags)
    initial_len = len(df_processed)
    df_processed = df_processed.dropna().reset_index(drop=True)
    logger.info(f"Dropped {initial_len - len(df_processed)} rows due to lag-induced NaNs.")
    
    logger.info(f"Processed dataset ready: {df_processed.shape[0]} rows × {df_processed.shape[1]} cols")
    return df_processed


def push_batch_to_feature_store(df: pd.DataFrame) -> None:
    """Insert the DataFrame directly into Hopsworks."""
    logger.info("Connecting to Hopsworks Feature Store...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        project = hopsworks.login(
            api_key_value=HOPSWORKS_API_KEY,
            project=HOPSWORKS_PROJECT_NAME,
        )

    fs = project.get_feature_store()

    df_hw = df.copy()
    df_hw.columns = [c.lower().replace(" ", "_") for c in df_hw.columns]
    df_hw["timestamp"] = pd.to_datetime(df_hw["timestamp"]).dt.tz_localize(None)

    int_cols = ["hour", "day_of_week", "month", "is_weekend", "is_night"]
    for col in int_cols:
        if col in df_hw.columns:
            df_hw[col] = df_hw[col].astype("int32")
            
    float_cols = df_hw.select_dtypes(include=["float32", "float64"]).columns
    for col in float_cols:
        df_hw[col] = df_hw[col].astype("float64")

    fg = get_feature_group(fs)
    fg.insert(df_hw, write_options={"wait_for_job": False})

    logger.info(
        f"  ✅ Inserted {len(df_hw)} rows directly into "
        f"Feature Group '{FEATURE_GROUP_NAME}' v{FEATURE_GROUP_VERSION}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill PM2.5 V3 (Lags) into Hopsworks")
    parser.add_argument("--days", type=int, default=90, help="Number of past days to backfill")
    parser.add_argument("--no-push", action="store_true", help="Skip Feature Store push")
    args = parser.parse_args()

    logger.info("=" * 55)
    logger.info(f"Backfill Pipeline V3 starting | days={args.days}")
    logger.info("=" * 55)

    df = build_historical_df(days=args.days)

    if not args.no_push:
        push_batch_to_feature_store(df)
    else:
        logger.info("  --no-push flag set: skipping Feature Store push")

    logger.info("Backfill Pipeline V3 complete ✅")

if __name__ == "__main__":
    main()
