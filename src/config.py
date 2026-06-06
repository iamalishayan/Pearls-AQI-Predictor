"""
Centralized configuration for the AQI Predictor project.

Loads environment variables and defines project-wide constants.

MLOps principle: no local data storage.
Hopsworks Feature Store is the single source of truth for all data.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env
load_dotenv()

# Project root (used for notebook paths, docs, etc.)
ROOT_DIR = Path(__file__).resolve().parent.parent

# API Keys
AQICN_API_KEY = os.getenv("AQICN_API_KEY", "")

# Hopsworks
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY", "")
HOPSWORKS_PROJECT_NAME = os.getenv("HOPSWORKS_PROJECT_NAME", "aqi_predictor")

# Target City
CITY_NAME = os.getenv("CITY_NAME", "islamabad")
CITY_LAT = float(os.getenv("CITY_LAT", "33.6844"))
CITY_LON = float(os.getenv("CITY_LON", "73.0479"))

# AQI Thresholds (EPA Standard)
AQI_CATEGORIES = {
    "Good":                         (0, 50),
    "Moderate":                     (51, 100),
    "Unhealthy for Sensitive Groups": (101, 150),
    "Unhealthy":                    (151, 200),
    "Very Unhealthy":               (201, 300),
    "Hazardous":                    (301, 500),
}

# MLflow
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT_NAME = "aqi-predictor"

# Feature Pipeline
FEATURE_GROUP_NAME = "aqi_features"
FEATURE_GROUP_VERSION = 4 # V4 includes pm25_change_24h

# Training Pipeline
MODEL_NAME = "aqi_predictor_model"
FORECAST_DAYS = 3
TEST_SIZE = 0.2
RANDOM_STATE = 42
