# Pearls AQI Predictor

> **3-Day Air Quality Index Forecasting System** — A fully automated, serverless ML pipeline that predicts AQI using real-time weather and pollutant data.

> **🟢 Live Dashboard:** [https://aqi-predictor-uwpy.onrender.com/](https://aqi-predictor-uwpy.onrender.com/)

[![Feature Pipeline](https://github.com/iamalishayan/Pearls-AQI-Predictor/actions/workflows/feature_pipeline.yml/badge.svg)](https://github.com/iamalishayan/Pearls-AQI-Predictor/actions/workflows/feature_pipeline.yml)
[![Training Pipeline](https://github.com/iamalishayan/Pearls-AQI-Predictor/actions/workflows/training_pipeline.yml/badge.svg)](https://github.com/iamalishayan/Pearls-AQI-Predictor/actions/workflows/training_pipeline.yml)

---

## Overview

This project builds an end-to-end MLOps system that:

1. **Fetches** real-time weather & pollutant data from Open-Meteo APIs (with AQICN for live AQI)
2. **Engineers** features (lag features, rolling statistics, AQI change rate, temporal encoding)
3. **Trains** multiple models (Ridge, Random Forest, XGBoost) and experiments with LSTM/GRU
4. **Evaluates** using RMSE, MAE, and R² metrics with TimeSeriesSplit cross-validation
5. **Deploys** the best model per horizon to a live Streamlit dashboard via Docker
6. **Alerts** users when AQI reaches hazardous levels with EPA-approved health actions

## Architecture

```
External APIs (Open-Meteo) → Feature Pipeline (Hourly) → Feature Store (Hopsworks)
                                                              ↓
                                                    Training Pipeline (Daily)
                                                              ↓
                                                      Model Registry (Hopsworks)
                                                              ↓
                                          FastAPI Backend + Streamlit Dashboard
                                                         (Docker)
```

See [docs/architecture.md](docs/architecture.md) for the full architecture diagram.

## Project Structure

```
├── .github/workflows/       # CI/CD (hourly feature, daily training)
├── docker-compose.yml        # Docker Compose configuration
├── Dockerfile.app            # Dockerfile for FastAPI & Streamlit
├── data/                     # Raw & processed data (gitignored)
├── docs/                     # Architecture, final report, assets
├── models/                   # Local model artifacts (gitignored)
├── notebooks/                # EDA & experimentation
├── src/                      # All Python source code
│   ├── feature_pipeline.py   # Hourly data ingestion & feature engineering
│   ├── backfill_pipeline.py  # Historical data backfill
│   ├── training_pipeline.py  # Daily model training & evaluation
│   ├── inference.py          # Model loading & prediction
│   ├── explainability.py     # SHAP explainability analysis
│   ├── alerts.py             # AQI threshold alerting system
│   ├── config.py             # Centralized configuration
│   ├── utils.py              # Shared utilities
│   └── app/                  # Web application
│       ├── api.py            # FastAPI backend
│       └── dashboard.py      # Streamlit frontend
└── tests/                    # Unit tests
```

## Quick Start

### 1. Clone & Setup

```bash
git clone https://github.com/iamalishayan/Pearls-AQI-Predictor.git
cd Pearls-AQI-Predictor

# Create virtual environment
python -m venv venv
source venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your API keys (AQICN, Hopsworks)
```

### 3. Run Locally with Docker

```bash
docker compose up -d --build
```

- **Streamlit Dashboard:** http://localhost:8501
- **FastAPI Docs:** http://localhost:8000/docs

## Running Tests

```bash
pytest tests/ -v
```

## Tech Stack

| Category | Tools |
|---|---|
| ML | scikit-learn, XGBoost |
| Deep Learning | PyTorch (LSTM, GRU) |
| Feature Store | Hopsworks |
| Experiment Tracking | MLflow |
| Web App | FastAPI + Streamlit |
| Explainability | SHAP |
| CI/CD | GitHub Actions |
| Containerization | Docker |

## Models Evaluated

| Model | Type | Status |
|---|---|---|
| Ridge Regression | Baseline | ✅ Production (72h) |
| Random Forest | Ensemble | ✅ Production (24h, 48h) |
| XGBoost | Gradient Boosting | ✅ Evaluated |
| LSTM | Deep Learning (PyTorch) | 📓 Notebook experiment |
| GRU | Deep Learning (PyTorch) | 📓 Notebook experiment |

## Documentation

- [Architecture](docs/architecture.md)
- [Final Report](docs/final_report.md)

## Author

**Ali Shayan** — [10Pearls](https://10pearls.com)

## License

This project is for educational purposes as part of the 10Pearls internship program.
