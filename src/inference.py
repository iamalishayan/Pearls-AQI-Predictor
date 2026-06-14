"""
Inference Module — The production core of the AQI Predictor.

Responsibilities:
  1. Load the best V3 forecasting model from Hopsworks Model Registry.
  2. Fetch the last 25 hours of live data from Open-Meteo to calculate lag features.
  3. Engineer features identically to training pipeline.
  4. Predict PM2.5 for t+24h.
  5. Convert to AQI and return explainability (SHAP top drivers).
"""

import os
import warnings
import joblib
import numpy as np
import pandas as pd
import shap
import hopsworks
import openmeteo_requests
import requests_cache
from retry_requests import retry
from datetime import datetime, timedelta, timezone

from src.config import (
    CITY_NAME, CITY_LAT, CITY_LON,
    HOPSWORKS_API_KEY, HOPSWORKS_PROJECT_NAME,
    MODEL_NAME, FEATURE_GROUP_NAME, FEATURE_GROUP_VERSION
)
from src.utils import setup_logger, pm25_to_aqi, classify_aqi
from src.alerts import check_alerts
from src.explainability import get_top_drivers

logger = setup_logger("inference")

_cached_models = {"24h": None, "48h": None, "72h": None}

def _load_model(horizon_label):
    """Download and cache the specific horizon model from Hopsworks."""
    global _cached_models
    if _cached_models[horizon_label] is not None:
        return _cached_models[horizon_label]

    logger.info(f"Loading '{horizon_label}' model from Hopsworks...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        project = hopsworks.login(
            api_key_value=HOPSWORKS_API_KEY,
            project=HOPSWORKS_PROJECT_NAME,
        )
    mr = project.get_model_registry()

    model_name = f"{MODEL_NAME}_{horizon_label}"

    # Dynamically fetch the best model version by lowest RMSE
    # (avoids hardcoding a version number that becomes stale after retraining)
    try:
        hw_model = mr.get_best_model(model_name, "rmse", "min")
        logger.info(f"Selected best model '{model_name}' v{hw_model.version} (lowest RMSE).")
    except Exception as e:
        logger.warning(f"get_best_model failed ({e}), falling back to version=1.")
        hw_model = mr.get_model(model_name, version=1)

    model_dir = hw_model.download()
    model_path = os.path.join(model_dir, f"forecast_model_{horizon_label}.pkl")

    _cached_models[horizon_label] = joblib.load(model_path)
    logger.info(f"✅ Model '{model_name}' v{hw_model.version} loaded successfully.")
    return _cached_models[horizon_label]

def fetch_recent_data() -> pd.DataFrame:
    """Fetch last 48 hours of weather + AQ data. Try Hopsworks FS first, fallback to Open-Meteo."""
    logger.info(f"Fetching recent data for {CITY_NAME}...")

    try:
        project = hopsworks.login(
            api_key_value=HOPSWORKS_API_KEY,
            project=HOPSWORKS_PROJECT_NAME,
        )
        fs = project.get_feature_store()
        fg = fs.get_feature_group(name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
        df = fg.read()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.sort_values("timestamp")
        
        # Use a wide window (90 days) so we always find data even if the
        # hourly feature pipeline hasn't run recently.
        cutoff = pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=90)
        df_recent = df[df["timestamp"] >= cutoff]
        
        if len(df_recent) > 2:
            # Take only the latest 48 rows for lag calculations
            df_recent = df_recent.tail(48)
            logger.info(f"✅ Successfully fetched {len(df_recent)} rows from Hopsworks Feature Store.")
            base_cols = ["timestamp", "pm10", "pm25", "co", "no2", "so2", "o3", 
                         "temperature", "humidity", "pressure", "wind_speed", "rain", "city_name"]
            if all(c in df_recent.columns for c in base_cols):
                return df_recent[base_cols].copy()
            else:
                logger.warning("Hopsworks data missing base columns, falling back to Open-Meteo.")
        else:
            logger.warning("Hopsworks returned insufficient data, falling back to Open-Meteo.")
    except Exception as e:
        logger.warning(f"⚠️ Failed to fetch from Hopsworks FS: {e}. Falling back to Open-Meteo API.")

    logger.info("Hitting Open-Meteo API directly...")
    try:
        cache_session = requests_cache.CachedSession('.cache', expire_after=1800)  # 30 min cache
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
        openmeteo = openmeteo_requests.Client(session=retry_session)

        end_date = datetime.now(timezone.utc).date()
        start_date = end_date - timedelta(days=2)

        # Air Quality
        resp_aq = openmeteo.weather_api(
            "https://air-quality-api.open-meteo.com/v1/air-quality",
            params={
                "latitude": CITY_LAT, "longitude": CITY_LON,
                "hourly": ["pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide", "ozone"],
                "start_date": start_date.strftime("%Y-%m-%d"),
                "end_date": end_date.strftime("%Y-%m-%d"),
            }
        )[0]

        # Weather
        resp_wx = openmeteo.weather_api(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": CITY_LAT, "longitude": CITY_LON,
                "hourly": ["temperature_2m", "relative_humidity_2m", "surface_pressure", "wind_speed_10m", "precipitation"],
                "start_date": start_date.strftime("%Y-%m-%d"),
                "end_date": end_date.strftime("%Y-%m-%d"),
            }
        )[0]

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

        return df

    except Exception as e:
        logger.error(f"❌ Open-Meteo API also failed: {e}")
        raise RuntimeError(
            "Could not fetch data from either Hopsworks or Open-Meteo. "
            "Open-Meteo may have hit its daily API rate limit. Please try again later."
        )

def _engineer_inference_features(df: pd.DataFrame) -> pd.DataFrame:
    """Mirror the training feature engineering exactly."""
    df = df.copy().sort_values("timestamp")

    dt = df["timestamp"].dt
    df["hour"] = dt.hour
    df["day_of_week"] = dt.dayofweek
    df["month"] = dt.month
    df["is_weekend"] = dt.dayofweek.isin([5, 6]).astype(int)
    df["is_night"] = ((dt.hour < 6) | (dt.hour >= 21)).astype(int)

    T = df["temperature"]
    H = df["humidity"]
    df["heat_index"] = T + 0.33 * (H / 100 * 6.105 * (17.27 * T / (237.7 + T))) - 4.0
    df["humidity_wind_interaction"] = df["humidity"] * df["wind_speed"]

    # Lag features
    df["pm25_lag_1h"] = df["pm25"].shift(1)
    df["pm25_lag_6h"] = df["pm25"].shift(6)
    df["pm25_lag_24h"] = df["pm25"].shift(24)
    df["pm25_change_24h"] = df["pm25"] - df["pm25_lag_24h"]
    df["pm25_rolling_mean_24h"] = df["pm25"].rolling(window=24).mean()
    df["temp_trend_6h"] = df["temperature"].diff(6)

    return df

def get_forecasts() -> dict:
    """
    Generate a 3-day PM2.5 forecast with explainability.

    Returns a dict with:
      - current_pm25: latest observed PM2.5
      - current_aqi: latest observed AQI
      - forecasts: list of dicts for 24h, 48h, 72h containing predictions and SHAP top drivers.
    """
    df_raw = fetch_recent_data()
    df_feat = _engineer_inference_features(df_raw)

    latest = df_feat.dropna().iloc[[-1]].copy()
    current_pm25 = float(latest["pm25"].iloc[0])
    current_aqi = pm25_to_aqi(current_pm25)
    base_time = pd.to_datetime(latest["timestamp"].iloc[0])

    drop_cols = ["pm25", "city_name", "timestamp"]
    X_raw = latest.drop(columns=[c for c in drop_cols if c in latest.columns])

    result = {
        "city": CITY_NAME.title(),
        "current_pm25": round(current_pm25, 2),
        "current_aqi": current_aqi,
        "alert": check_alerts(current_aqi),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "all_features": {k: round(float(v), 4) for k, v in X_raw.iloc[0].items()},
        "forecasts": []
    }

    horizons = {"24h": 24, "48h": 48, "72h": 72}

    for label, hours in horizons.items():
        try:
            model = _load_model(label)
            
            # Align features
            if hasattr(model, "feature_names_in_"):
                X = X_raw[model.feature_names_in_]
            else:
                expected_cols = [
                    "hour", "day_of_week", "month", "is_weekend", "is_night",
                    "temperature", "humidity", "pressure", "wind_speed", "rain",
                    "heat_index", "humidity_wind_interaction", "pm10", "no2", "so2",
                    "co", "o3", "pm25_lag_1h", "pm25_lag_6h", "pm25_lag_24h",
                    "pm25_change_24h", "pm25_rolling_mean_24h", "temp_trend_6h"
                ]
                X = X_raw[[c for c in expected_cols if c in X_raw.columns]]

            pm25_pred = float(model.predict(X)[0])
            pm25_pred = max(0.0, pm25_pred)
            aqi_pred = pm25_to_aqi(pm25_pred)
            category = classify_aqi(aqi_pred)
            forecast_time = base_time + timedelta(hours=hours)

            # SHAP
            top_drivers = get_top_drivers(model, X, top_n=5)

            result["forecasts"].append({
                "horizon": label,
                "forecast_time": forecast_time.isoformat(),
                "pm25_forecast": round(pm25_pred, 2),
                "aqi_forecast": aqi_pred,
                "aqi_category": category,
                "top_drivers": top_drivers
            })
            logger.info(f"🚀 {label} Forecast: PM2.5={pm25_pred:.1f} | AQI={aqi_pred}")
        except Exception as e:
            logger.error(f"Failed to generate forecast for {label}: {e}")

    return result

if __name__ == "__main__":
    import json
    result = get_forecasts()
    print(json.dumps(result, indent=2))
