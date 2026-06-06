"""
FastAPI Backend — AQI Predictor API

Endpoints:
  /predict  → 24h PM2.5 forecast with explainability
  /explain  → SHAP feature importance breakdown
  /health   → Model version and system status
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timezone
import traceback

app = FastAPI(
    title="AQI Predictor — Islamabad",
    description="24-hour PM2.5 forecasting with SHAP explainability",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cache the last forecast to avoid hammering APIs on every request
_forecast_cache = {"data": None, "expires": None}
CACHE_TTL_SECONDS = 900  # 15 minutes


@app.get("/")
def root():
    return {
        "service": "AQI Predictor — Islamabad",
        "version": "3.0.0",
        "endpoints": ["/predict", "/explain", "/health"],
    }


@app.get("/predict")
def predict():
    """Return a 3-day PM2.5 forecast with AQI conversion and top drivers."""
    from src.inference import get_forecasts

    now = datetime.now(timezone.utc)

    # Return cached result if fresh
    if _forecast_cache["data"] and _forecast_cache["expires"] and now < _forecast_cache["expires"]:
        return _forecast_cache["data"]

    try:
        result = get_forecasts()
        _forecast_cache["data"] = result
        _forecast_cache["expires"] = now.replace(second=0, microsecond=0).__add__(
            __import__("datetime").timedelta(seconds=CACHE_TTL_SECONDS)
        )
        return result
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/explain")
def explain():
    """Return detailed SHAP explainability for the latest 3-day forecast."""
    from src.inference import get_forecasts

    try:
        result = get_forecasts()
        return result
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Explanation failed: {str(e)}")


@app.get("/health")
def health():
    """Return system health and model metadata."""
    return {
        "status": "healthy",
        "model_name": "aqi_predictor_model_v3",
        "forecast_horizon": "24h",
        "target_variable": "PM2.5 (µg/m³)",
        "city": "Islamabad",
        "feature_group_version": 3,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.app.api:app", host="0.0.0.0", port=8000, reload=True)
