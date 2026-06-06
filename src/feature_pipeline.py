"""
Feature Pipeline — Runs hourly via CI/CD.

MLOps Pattern:
  Open-Meteo APIs  →  Vectorized Feature Engineering (with Lags)  →  Hopsworks Feature Store
Target ML Variable: pm25
"""

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
from src.utils import setup_logger

logger = setup_logger("feature_pipeline")


def fetch_live_data() -> pd.DataFrame:
    """
    Fetch last 25 hours of weather and air quality data from Open-Meteo 
    to enable lag feature engineering for the current hour.
    """
    logger.info(f"Fetching recent history for {CITY_NAME} from Open-Meteo (to calculate lags)...")
    
    cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)
    
    # We fetch a 2-day window to be safe and ensure we have at least 24 hours of history
    end_date = datetime.now(timezone.utc).date()
    start_date = end_date - timedelta(days=1)
    
    params_aq = {
        "latitude": CITY_LAT,
        "longitude": CITY_LON,
        "hourly": ["pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide", "ozone"],
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
    }
    
    params_wx = {
        "latitude": CITY_LAT,
        "longitude": CITY_LON,
        "hourly": ["temperature_2m", "relative_humidity_2m", "surface_pressure", "wind_speed_10m", "precipitation"],
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
    }
    
    resp_aq = openmeteo.weather_api("https://air-quality-api.open-meteo.com/v1/air-quality", params=params_aq)[0]
    resp_wx = openmeteo.weather_api("https://api.open-meteo.com/v1/forecast", params=params_wx)[0]
    
    hourly_aq = resp_aq.Hourly()
    hourly_wx = resp_wx.Hourly()
    
    date_range = pd.date_range(
        start=pd.to_datetime(hourly_aq.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly_aq.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly_aq.Interval()),
        inclusive="left"
    )
    
    df = pd.DataFrame({"timestamp": date_range})
    df["pm10"] = hourly_aq.Variables(0).ValuesAsNumpy()
    df["pm25"] = hourly_aq.Variables(1).ValuesAsNumpy()
    df["co"] = hourly_aq.Variables(2).ValuesAsNumpy()
    df["no2"] = hourly_aq.Variables(3).ValuesAsNumpy()
    df["so2"] = hourly_aq.Variables(4).ValuesAsNumpy()
    df["o3"] = hourly_aq.Variables(5).ValuesAsNumpy()
    df["temperature"] = hourly_wx.Variables(0).ValuesAsNumpy()
    df["humidity"] = hourly_wx.Variables(1).ValuesAsNumpy()
    df["pressure"] = hourly_wx.Variables(2).ValuesAsNumpy()
    df["wind_speed"] = hourly_wx.Variables(3).ValuesAsNumpy()
    df["rain"] = hourly_wx.Variables(4).ValuesAsNumpy()
    df["city_name"] = CITY_NAME
    
    # We only care about the latest row after calculating lags
    logger.info(f"Fetched {len(df)} recent hours.")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorized feature engineering on a Pandas DataFrame.
    Includes Lag and Rolling features for intelligence.
    """
    df = df.copy().sort_values("timestamp")
    
    # 1. Time-based features
    dt = df["timestamp"].dt
    df["hour"] = dt.hour
    df["day_of_week"] = dt.dayofweek
    df["month"] = dt.month
    df["is_weekend"] = dt.dayofweek.isin([5, 6]).astype(int)
    df["is_night"] = ((dt.hour < 6) | (dt.hour >= 21)).astype(int)
    
    # 2. Physics/Interaction features
    T = df["temperature"]
    H = df["humidity"]
    df["heat_index"] = T + 0.33 * (H / 100 * 6.105 * (17.27 * T / (237.7 + T))) - 4.0
    df["humidity_wind_interaction"] = df["humidity"] * df["wind_speed"]
    
    # 3. Lag Features (The "Intelligence" Layer)
    # We shift 'pm25' to give the model context of the recent past
    df["pm25_lag_1h"] = df["pm25"].shift(1)
    df["pm25_lag_6h"] = df["pm25"].shift(6)
    df["pm25_lag_24h"] = df["pm25"].shift(24)
    df["pm25_change_24h"] = df["pm25"] - df["pm25_lag_24h"]
    
    # Rolling mean for the last 24 hours
    df["pm25_rolling_mean_24h"] = df["pm25"].rolling(window=24).mean()
    
    # Trend feature: Temperature rate of change over 6 hours
    df["temp_trend_6h"] = df["temperature"].diff(6)
    
    # Drop rows that don't have enough history for lags (will be N/A)
    # This is important for backfill; for live, we just take the last row.
    return df


def get_feature_group(fs):
    """Return the Hopsworks Feature Group, creating it if it doesn't exist."""
    from hsfs.feature import Feature

    features = [
        Feature("city_name",                 type="string"),
        Feature("timestamp",                 type="timestamp"),
        Feature("hour",                      type="int"),
        Feature("day_of_week",               type="int"),
        Feature("month",                     type="int"),
        Feature("is_weekend",                type="int"),
        Feature("is_night",                  type="int"),
        Feature("temperature",               type="double"),
        Feature("humidity",                  type="double"),
        Feature("pressure",                  type="double"),
        Feature("wind_speed",                type="double"),
        Feature("rain",                      type="double"),
        Feature("heat_index",                type="double"),
        Feature("humidity_wind_interaction", type="double"),
        Feature("pm10",                      type="double"),
        Feature("no2",                       type="double"),
        Feature("so2",                       type="double"),
        Feature("co",                        type="double"),
        Feature("o3",                        type="double"),
        # Intelligence Features
        Feature("pm25_lag_1h",               type="double"),
        Feature("pm25_lag_6h",               type="double"),
        Feature("pm25_lag_24h",              type="double"),
        Feature("pm25_change_24h",           type="double"),
        Feature("pm25_rolling_mean_24h",     type="double"),
        Feature("temp_trend_6h",             type="double"),
        Feature("pm25",                      type="double"), # Target variable
    ]

    fg = fs.get_or_create_feature_group(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION,
        primary_key=["city_name", "timestamp"],
        description="V3: Hourly AQI & Weather with 24h Lag Features for Islamabad",
        features=features,
        online_enabled=False,
    )
    return fg


def push_to_feature_store(df: pd.DataFrame) -> None:
    """Insert the latest row into Hopsworks."""
    logger.info("Connecting to Hopsworks Feature Store...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        project = hopsworks.login(
            api_key_value=HOPSWORKS_API_KEY,
            project=HOPSWORKS_PROJECT_NAME,
        )

    fs = project.get_feature_store()

    # Take only the latest row (the current hour)
    latest_row = df.iloc[[-1]].copy()
    
    # Verify no NaNs in intelligence features for the latest row
    if latest_row.isnull().any().any():
        logger.warning("Latest row contains NaNs in lag features. This is expected if the API didn't return enough history.")

    latest_row.columns = [c.lower().replace(" ", "_") for c in latest_row.columns]
    latest_row["timestamp"] = pd.to_datetime(latest_row["timestamp"]).dt.tz_localize(None)
    
    int_cols = ["hour", "day_of_week", "month", "is_weekend", "is_night"]
    for col in int_cols:
        if col in latest_row.columns:
            latest_row[col] = latest_row[col].astype("int32")

    float_cols = latest_row.select_dtypes(include=["float32", "float64"]).columns
    for col in float_cols:
        latest_row[col] = latest_row[col].astype("float64")

    fg = get_feature_group(fs)
    fg.insert(latest_row, write_options={"wait_for_job": False})
    
    logger.info(f"  ✅ Pushed live row (with lags) to Feature Group '{FEATURE_GROUP_NAME}' v{FEATURE_GROUP_VERSION}")


def main() -> None:
    logger.info("=" * 55)
    logger.info("Feature Pipeline starting (V3 with Lags)")
    logger.info("=" * 55)

    df_raw = fetch_live_data()
    df_features = engineer_features(df_raw)
    push_to_feature_store(df_features)

    logger.info("Feature Pipeline complete ✅")

if __name__ == "__main__":
    main()
