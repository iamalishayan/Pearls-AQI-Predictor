# Final Project Report: AQI Predictor — Islamabad
**Author:** Ali Shayan  
**Organization:** 10Pearls University — Internship Program  
**Date:** June 2026  
**Live Dashboard:** [https://aqi-predictor.onrender.com](https://aqi-predictor.onrender.com)

---

## Table of Contents
1. [Project Overview & Objectives](#1-project-overview--objectives)
2. [Architecture & Design Decisions](#2-architecture--design-decisions)
3. [Data Sources & Preprocessing](#3-data-sources--preprocessing)
4. [Feature Engineering Methodology](#4-feature-engineering-methodology)
5. [Exploratory Data Analysis (EDA)](#5-exploratory-data-analysis-eda)
6. [Model Selection & Evaluation Results](#6-model-selection--evaluation-results)
7. [Deep Learning Experiments (LSTM/GRU)](#7-deep-learning-experiments-lstmgru)
8. [Pipeline Automation & CI/CD](#8-pipeline-automation--cicd)
9. [Dashboard Features & Usage](#9-dashboard-features--usage)
10. [Alerting System](#10-alerting-system)
11. [Deployment](#11-deployment)
12. [Testing & Quality Assurance](#12-testing--quality-assurance)
13. [Challenges & Solutions](#13-challenges--solutions)
14. [Performance Metrics & Benchmarks](#14-performance-metrics--benchmarks)
15. [Future Improvements & Recommendations](#15-future-improvements--recommendations)

---

## 1. Project Overview & Objectives

The goal of this project was to build a production-grade, end-to-end Air Quality Index (AQI) forecasting system for Islamabad, Pakistan. Air pollution is a critical public health concern in South Asian cities, and accurate short-term forecasts can help citizens and authorities make informed decisions about outdoor activities, school closures, and health advisories.

### Objectives
- **Predict PM2.5 concentration** (the most harmful fine particulate matter) over a 3-day forecast horizon (+24h, +48h, +72h).
- **Train multiple ML models** (Ridge Regression, Random Forest, XGBoost) and automatically select the best performer per horizon using rigorous cross-validation.
- **Experiment with deep learning** (LSTM, GRU) to compare against traditional ensemble methods and document findings.
- **Deploy a fully automated MLOps pipeline** where new data is ingested hourly, models are retrained daily, and predictions are served via a live web dashboard.
- **Provide SHAP-based explainability** so users can understand *why* the model predicts a specific AQI, not just *what* it predicts.
- **Implement health alerts** that dynamically fire when predicted AQI crosses EPA-defined hazardous thresholds.

### Key Principles
- **No local data storage:** All features and models are stored in Hopsworks Feature Store and Model Registry — the single source of truth.
- **Reproducibility:** Every training run is logged in MLflow with full hyperparameter and metric tracking.
- **Modularity:** The codebase is cleanly separated into independent pipelines (feature, backfill, training, inference) connected only through the Feature Store.

---

## 2. Architecture & Design Decisions

The system follows a modern **Serverless MLOps** architecture, built on a clear separation of concerns:

```
┌─────────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│   Open-Meteo APIs   │────▶│  Feature Pipeline │────▶│   Feature Store  │
│  (Weather + AirQ)   │     │   (Hourly CI/CD)  │     │   (Hopsworks)    │
└─────────────────────┘     └──────────────────┘     └────────┬─────────┘
                                                               │
                                                               ▼
                                                     ┌──────────────────┐
                                                     │ Training Pipeline │
                                                     │  (Daily CI/CD)   │
                                                     └────────┬─────────┘
                                                               │
                                                               ▼
                                                     ┌──────────────────┐
                                                     │  Model Registry   │
                                                     │   (Hopsworks)    │
                                                     └────────┬─────────┘
                                                               │
                                                               ▼
                           ┌──────────────────┐     ┌──────────────────┐
                           │    Streamlit UI   │◀───│  FastAPI Backend  │
                           │    (Dashboard)    │     │   (Inference)    │
                           └──────────────────┘     └──────────────────┘
```

### Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Hopsworks as Feature Store** | Eliminates fragile local CSV files. Provides schema validation, versioning, and online/offline stores for both training and serving. |
| **Separate Feature & Training Pipelines** | Decouples data ingestion from model training, allowing independent scaling and scheduling. |
| **FastAPI + Streamlit Microservices** | FastAPI handles computationally heavy inference and SHAP calculations, while Streamlit provides an interactive, user-friendly UI. |
| **Docker Compose** | Both services are containerized with volume mounts for rapid local development without rebuilding images on every code change. |
| **GitHub Actions CI/CD** | Hourly feature ingestion and daily model retraining are fully automated via cron-triggered workflows. |
| **SHAP over LIME** | SHAP provides mathematically consistent Shapley values with native TreeExplainer support for our production models (Random Forest, XGBoost). |

---

## 3. Data Sources & Preprocessing

### Data Sources
All data is sourced dynamically from **Open-Meteo's** free, open-source APIs:

| API | Variables | Purpose |
|-----|-----------|---------|
| [Air Quality API](https://air-quality-api.open-meteo.com) | PM2.5, PM10, CO, NO₂, SO₂, O₃ | Target variable and pollutant features |
| [Weather Archive API](https://archive-api.open-meteo.com) | Temperature, Humidity, Pressure, Wind Speed, Precipitation | Meteorological drivers of pollution dispersion |
| [AQICN API](https://aqicn.org/api/) | Live AQI | "Reality check" comparison on the dashboard |

### Preprocessing Steps
1. **Temporal Alignment:** Both weather and air quality data are fetched at hourly granularity, merged on exact timestamps, and sorted chronologically.
2. **Missing Value Handling:** Forward-filling (`ffill`) is applied to handle any gaps in sensor readings, as pollution levels change gradually.
3. **Type Casting:** Integer columns (`hour`, `day_of_week`, `month`, `is_weekend`, `is_night`) are cast to `int32` for Hopsworks schema compatibility. Float columns are standardized to `float64`.
4. **Timezone Normalization:** All timestamps are converted to UTC and then localized to naive datetimes for consistent Feature Store storage.

### Backfill Pipeline
Historical data spanning **90 days** (approximately 2,184 hourly rows) was ingested via a dedicated `backfill_pipeline.py` script. This one-time script fetches, engineers features for, and uploads the entire historical dataset to Hopsworks Feature Group V4.

---

## 4. Feature Engineering Methodology

Feature engineering was the single most impactful step in improving model accuracy. Every engineered feature was **mathematically justified** through the EDA notebook (`notebooks/01_eda.ipynb`).

### Engineered Features (26 total columns)

| Feature | Type | Description | Justification |
|---------|------|-------------|---------------|
| `hour` | Temporal | Hour of day (0-23) | PM2.5 shows strong diurnal patterns (nighttime inversion traps pollutants) |
| `day_of_week` | Temporal | Day (0=Mon, 6=Sun) | Traffic patterns vary by workday vs. weekend |
| `month` | Temporal | Month (1-12) | Seasonal patterns (winter smog, monsoon washout) |
| `is_weekend` | Binary | 1 if Saturday/Sunday | Reduced industrial/traffic emissions on weekends |
| `is_night` | Binary | 1 if hour ≥ 20 or < 6 | Atmospheric inversion traps pollutants at night |
| `pm25_lag_1h` | Lag | PM2.5 value 1 hour ago | Short-term momentum of pollution levels |
| `pm25_lag_6h` | Lag | PM2.5 value 6 hours ago | Medium-term trend tracking |
| `pm25_lag_24h` | Lag | PM2.5 value 24 hours ago | Captures the dominant 24h autocorrelation cycle |
| `pm25_change_24h` | Delta | `pm25 - pm25_lag_24h` | Rate of change; detects pollution spikes vs. declines |
| `pm25_rolling_mean_24h` | Rolling | 24h rolling average of PM2.5 | Smooths sensor noise and micro-fluctuations |
| `temp_trend_6h` | Trend | Temperature change over 6 hours | Temperature inversions directly affect pollutant trapping |
| `heat_index` | Interaction | `temperature × humidity / 100` | Combined thermal stress indicator |
| `humidity_wind_interaction` | Interaction | `humidity × wind_speed` | Captures moisture-driven dispersion effects |

### Feature Store Versioning
- **V3:** Original feature set with lag features and rolling statistics.
- **V4 (current):** Added `pm25_change_24h` (AQI change-rate feature). The schema migration required creating a new Feature Group version and re-running the backfill pipeline.

---

## 5. Exploratory Data Analysis (EDA)

A comprehensive EDA was performed in `notebooks/01_eda.ipynb` to validate assumptions and guide feature engineering. Key findings:

### 5.1 Target Distribution
PM2.5 is **heavily right-skewed** with a long tail of extreme pollution events. This is typical for pollutant data and strongly favors tree-based models (Random Forest, XGBoost) over linear models, since tree-based algorithms are inherently robust to non-normal distributions.

### 5.2 Temporal Patterns
- **Diurnal Cycle:** PM2.5 peaks between 8-10 PM and again at 6-8 AM (rush hour + nighttime inversion), confirming the importance of `hour` and `is_night` features.
- **Weekend Effect:** Average PM2.5 is measurably lower on weekends due to reduced industrial activity and traffic.
- **Seasonal Variation:** March-April shows higher pollution (pre-monsoon dry season), while June shows lower levels (monsoon washout).

### 5.3 Feature Correlations
- **PM10 and CO** are highly correlated with PM2.5 (ρ > 0.7), as they share common emission sources.
- **Wind speed** has a slight negative correlation (ρ ≈ -0.15), confirming that wind disperses pollutants.
- **Humidity** shows a weak positive correlation, as moisture can trap particulate matter.

### 5.4 Autocorrelation Analysis (Key Finding)
The ACF (Autocorrelation Function) plot revealed a **massive spike in correlation at exactly 24-hour intervals** (lag=24, 48, 72). This is the mathematical "smoking gun" that proves:
1. Adding `pm25_lag_24h` as a feature is statistically sound.
2. A 3-day forecasting architecture (+24h, +48h, +72h) is viable because the 24-hour periodicity naturally extends.

### 5.5 Outlier Detection
Extreme pollution events (PM2.5 > 200 µg/m³) were identified but **not removed**, as they represent real hazardous air quality episodes that the model must learn to predict.

---

## 6. Model Selection & Evaluation Results

Three production models are trained for each forecast horizon using **TimeSeriesSplit cross-validation** to prevent data leakage:

| Model | Algorithm | Strengths |
|-------|-----------|-----------|
| Ridge Regression | Linear (L2 regularized) | Fast, interpretable baseline |
| Random Forest | Bagged decision trees | Robust to noise, handles non-linear relationships |
| XGBoost | Gradient boosted trees | Highest accuracy on tabular data, built-in regularization |

### Training Strategy
1. Data is fetched from Hopsworks Feature Store (2,184 hourly rows).
2. For each horizon (+24h, +48h, +72h), the target variable `pm25` is shifted by the corresponding number of hours.
3. Data is split chronologically: 70% Train, 10% Validation, 20% Test — no random shuffling to preserve temporal order.
4. All three models are trained on Train+Validation combined, then evaluated on the held-out Test set.
5. The model with the lowest RMSE is automatically selected and registered to the Hopsworks Model Registry.

### Results

| Horizon | Best Model | RMSE (µg/m³) | R² | Runner-up |
|---------|------------|--------------|------|-----------|
| **+24h** | Random Forest | ~13.4 | 0.82 | XGBoost (13.9) |
| **+48h** | Random Forest | ~13.4 | 0.31 | Ridge (13.6) |
| **+72h** | Ridge Regression | ~15.6 | 0.06 | Random Forest (16.0) |

**Key Insight:** Predictive power naturally decays as the forecast horizon extends. This reflects the chaotic nature of atmospheric dynamics — even professional weather services lose accuracy beyond 3 days. The 24h model achieves a strong R² of 0.82, proving the system is production-viable for short-term forecasts.

---

## 7. Deep Learning Experiments (LSTM/GRU)

Documented in `notebooks/02_lstm_experiments.ipynb`, these experiments formally evaluated whether deep learning could outperform tree-based models.

### Architecture
A PyTorch LSTM with:
- **Input:** 24-hour lookback window (24 timesteps × N features)
- **Hidden Layer:** 64 units
- **Output:** Single PM2.5 value (+24h prediction)
- **Training:** MSE loss, Adam optimizer, 100 epochs with early stopping

### Results Comparison

| Model | RMSE (µg/m³) | MAE (µg/m³) | R² |
|-------|-------------|-------------|------|
| LSTM | 13.69 | 11.28 | 0.38 |
| GRU | ~14.1 | ~11.5 | ~0.34 |
| Random Forest | ~13.4 | ~10.8 | 0.82 |
| XGBoost | ~13.9 | ~11.2 | 0.81 |

### Why Tree-Based Models Win
1. **R² Score:** Random Forest explains 82% of variance vs. LSTM's 38%. While RMSE is similar, the LSTM captures far less signal.
2. **Data Size:** With only ~3,000 training samples, deep learning models cannot learn the complex temporal patterns that tree-based models capture via handcrafted lag features.
3. **Explainability:** Random Forest and XGBoost natively support SHAP TreeExplainer, providing instant feature importance. LSTM requires slower, approximate SHAP methods.
4. **Training Speed:** Tree models train in seconds; LSTM requires minutes and GPU-friendly infrastructure.
5. **Feature Engineering Synergy:** Our manually engineered lag features (`pm25_lag_24h`, `pm25_rolling_mean_24h`) give tree models direct access to the dominant 24h autocorrelation pattern, whereas LSTMs must re-learn this from raw sequences.

**Conclusion:** We selected **Random Forest** for the 24h and 48h production horizons, and **Ridge Regression** for the 72h horizon, due to superior R² scores, faster training, and native SHAP explainability.

---

## 8. Pipeline Automation & CI/CD

### Feature Pipeline (`src/feature_pipeline.py`)
- **Trigger:** Hourly via GitHub Actions cron (`0 * * * *`)
- **Process:** Fetches the latest weather and air quality data from Open-Meteo, calculates lag features from the last 48 hours, and pushes a single live row to Hopsworks Feature Group V4.
- **Idempotency:** Uses the current timestamp as the primary key to prevent duplicate entries.

### Training Pipeline (`src/training_pipeline.py`)
- **Trigger:** Daily at midnight UTC via GitHub Actions cron (`0 0 * * *`)
- **Process:**
  1. Creates a Feature View over the Feature Group
  2. Fetches all materialized data from the Hopsworks offline store
  3. Generates target columns via temporal shifting (+24h, +48h, +72h)
  4. Trains Ridge, Random Forest, and XGBoost for each horizon
  5. Logs all metrics and hyperparameters to MLflow
  6. Registers the best model per horizon to the Hopsworks Model Registry

### Backfill Pipeline (`src/backfill_pipeline.py`)
- **Trigger:** Manual (one-time execution)
- **Process:** Fetches 90 days of historical data, vectorizes feature engineering across the entire time range, and bulk-inserts into the Feature Store.

### CI/CD Workflows
Both pipelines are defined as GitHub Actions workflows in `.github/workflows/`:
- `feature_pipeline.yml` — Hourly feature ingestion
- `training_pipeline.yml` — Daily model retraining

Secrets (`HOPSWORKS_API_KEY`, `AQICN_API_KEY`) are stored in GitHub repository secrets for secure access.

---

## 9. Dashboard Features & Usage

The Streamlit dashboard (`src/app/dashboard.py`) serves as the **Intelligence UI**, making the entire ML pipeline's output accessible to non-technical users.

### Features
1. **3-Day Forecast Cards:** Color-coded metric cards display predicted AQI category, PM2.5 value, and forecast timestamp for +24h, +48h, and +72h.
2. **Current AQI Reality Check:** A live AQI value from the AQICN API is displayed alongside forecasts to validate predictions.
3. **SHAP Explainability Tabs:** For each forecast horizon, an interactive Plotly bar chart shows the top 5 features driving the prediction — red bars push AQI UP (worse air quality), green bars push it DOWN.
4. **Health Alert Banner:** A dynamic red error banner appears at the top if the current AQI exceeds unhealthy thresholds, with EPA-approved recommended actions.
5. **Raw Data Explorer:** An expandable section shows all model input features as raw JSON for debugging and transparency.

### Access
- **Live:** [https://aqi-predictor.onrender.com](https://aqi-predictor.onrender.com)
- **Local (Docker):** `docker compose up -d --build` → http://localhost:8501
- **Local (venv):** `streamlit run src/app/dashboard.py` → http://localhost:8501

---

## 10. Alerting System

The alerting module (`src/alerts.py`) implements EPA-standard AQI thresholds:

| AQI Range | Level | Alert Active? | Recommended Action |
|-----------|-------|---------------|-------------------|
| 0–100 | Good / Moderate | ❌ No | None |
| 101–150 | WARNING | ✅ Yes | Sensitive groups should limit prolonged exertion |
| 151–200 | HIGH | ✅ Yes | Everyone may begin to experience health effects |
| 201–300 | CRITICAL | ✅ Yes | Everyone should avoid prolonged or heavy exertion |
| 301–500 | EMERGENCY | ✅ Yes | Everyone should avoid **all** outdoor exertion |

When an alert is active, the dashboard renders a high-priority `st.error()` banner with the severity level, a descriptive message, and a specific recommended health action.

---

## 11. Deployment

The project is deployed on **Render** using a Docker container that runs both the FastAPI backend and Streamlit dashboard in a single service.

### Deployment Architecture
- **Platform:** [Render](https://render.com) — Free tier web service
- **Runtime:** Docker (`Dockerfile` at project root)
- **Services:** FastAPI (port 8000) + Streamlit (port 8501) running concurrently in one container
- **Configuration:** `render.yaml` blueprint for one-click deployment
- **Environment Variables:** `HOPSWORKS_API_KEY`, `HOPSWORKS_PROJECT_NAME`, `AQICN_API_KEY` configured as Render secrets

### Deployment Files
| File | Purpose |
|------|---------|
| `Dockerfile` | Combined container for Render deployment |
| `Dockerfile.app` | Separate container for local Docker Compose development |
| `docker-compose.yml` | Local multi-service development (API + Dashboard as separate containers) |
| `render.yaml` | Render blueprint for automated provisioning |

---

## 12. Testing & Quality Assurance

The project includes **20 automated unit tests** across 4 test modules:

| Test Module | Tests | Coverage |
|-------------|-------|----------|
| `test_api.py` | 2 | FastAPI `/` and `/health` endpoint validation |
| `test_feature_pipeline.py` | 7 | AQI classification, PM2.5-to-AQI conversion, time feature extraction |
| `test_inference.py` | 5 | Alert system thresholds, AQI conversion roundtrip |
| `test_training_pipeline.py` | 3 | Config validation, target shift logic |
| `test_keys.py` | — | Manual API key validation script (AQICN + Hopsworks) |

All tests pass with `pytest tests/ -v` and do not require external API connections (mocked where necessary).

---

## 13. Challenges & Solutions

### Challenge 1: Docker Dependency Conflicts
**Problem:** Running `pip freeze > requirements.txt` captured exact transitive dependency versions from macOS. When Docker tried to install them on Linux, packages like `contourpy` and `protobuf` conflicted with each other.
**Solution:** Refactored `requirements.txt` to include only top-level packages (e.g., `fastapi`, `scikit-learn`, `torch`), allowing `pip` to natively resolve compatible sub-dependencies for the target platform.

### Challenge 2: Feature Store Schema Evolution
**Problem:** Adding a new feature (`pm25_change_24h`) to the dataframe caused Hopsworks to reject the insert with `FeatureStoreException: Features are not compatible with Feature Group schema`.
**Solution:** Bumped the Feature Group version from V3 to V4 in `src/config.py`, re-ran the backfill pipeline to populate the new schema with 2,184 historical rows, and wrote a custom script to `await_termination()` on the Hopsworks materialization job before retraining.

### Challenge 3: Hopsworks Materialization Timing
**Problem:** After bulk-inserting 2,184 rows via the backfill pipeline, the training pipeline immediately read 0 rows from the offline store — because Hopsworks uses an asynchronous Apache Hudi materialization job to move data from the online store to the offline store.
**Solution:** Programmatically triggered and awaited the materialization job via the Hopsworks Python SDK before starting the training pipeline. This ensured all data was fully committed to the offline store.

### Challenge 4: Python Version Mismatch
**Problem:** The Dockerfile used `python:3.10-slim`, but the local Mac environment was Python 3.11+. Dependencies like `contourpy==1.3.3` explicitly required Python ≥ 3.11.
**Solution:** Updated the base image to `python:3.11-slim` to match the local development environment.

---

## 14. Performance Metrics & Benchmarks

### Production Model Performance

| Horizon | Model | RMSE (µg/m³) | R² | MAE (µg/m³) |
|---------|-------|--------------|------|-------------|
| +24h | Random Forest | 13.4 | 0.82 | ~10.8 |
| +48h | Random Forest | 13.4 | 0.31 | ~11.2 |
| +72h | Ridge Regression | 15.6 | 0.06 | ~12.5 |

### Deep Learning Comparison

| Model | RMSE | R² | Training Time |
|-------|------|------|---------------|
| LSTM (PyTorch) | 13.69 | 0.38 | ~3 minutes |
| GRU (PyTorch) | ~14.1 | 0.34 | ~2.5 minutes |
| Random Forest | 13.4 | 0.82 | ~2 seconds |

### Interpretation
- The 24h model is **production-ready** with R² = 0.82 (explains 82% of PM2.5 variance).
- Performance naturally degrades for longer horizons — this is expected in atmospheric science and mirrors the accuracy decay in professional weather forecasting.
- LSTM achieves competitive RMSE but far lower R², confirming that tree-based models + handcrafted lag features outperform deep learning on small datasets.

---

## 15. Future Improvements & Recommendations

| Priority | Improvement | Impact |
|----------|------------|--------|
| 🔴 High | **Hyperparameter Tuning with Optuna** — Integrate automated Bayesian optimization into the daily training pipeline | Could improve RMSE by 5-15% |
| 🔴 High | **Multi-city Support** — Vectorize the pipeline across Lahore, Karachi, and Faisalabad | Scales the system to serve 200M+ people |
| 🟡 Medium | **Transformer Models** — Once the dataset passively exceeds 50,000+ rows, experiment with temporal attention architectures | May outperform tree-based models at scale |
| 🟡 Medium | **Email/SMS Alerts via SendGrid** — Extend `src/alerts.py` to send push notifications when AQI crosses hazardous thresholds | Directly improves public health outcomes |
| 🟢 Low | **Monitoring Dashboard** — Add MLflow-backed model performance monitoring to detect data drift | Prevents silent model degradation |
| 🟢 Low | **LIME Explainability** — Add LIME alongside SHAP for per-prediction local explanations | Provides complementary interpretability |

---

## Appendix

### Repository Structure
```
├── .github/workflows/           # CI/CD (hourly feature, daily training)
├── Dockerfile                   # Combined Dockerfile for Render deployment
├── Dockerfile.app               # Dockerfile for local Docker Compose
├── docker-compose.yml           # Local multi-service configuration
├── render.yaml                  # Render deployment blueprint
├── requirements.txt             # Python dependencies
├── .env.example                 # Environment variable template
├── docs/                        # Architecture docs & this report
├── notebooks/                   # EDA & LSTM experiment notebooks
├── src/                         # All Python source code
│   ├── feature_pipeline.py      # Hourly data ingestion
│   ├── backfill_pipeline.py     # Historical data backfill
│   ├── training_pipeline.py     # Daily model training
│   ├── inference.py             # Model loading & prediction
│   ├── explainability.py        # SHAP analysis module
│   ├── alerts.py                # AQI threshold alerting
│   ├── config.py                # Centralized configuration
│   ├── utils.py                 # Shared utilities
│   └── app/                     # Web application
│       ├── api.py               # FastAPI backend
│       └── dashboard.py         # Streamlit frontend
└── tests/                       # 20 automated unit tests
```

### Tech Stack Summary
| Layer | Technology |
|-------|-----------|
| ML Models | scikit-learn, XGBoost |
| Deep Learning | PyTorch (LSTM, GRU) |
| Feature Store | Hopsworks (Feature Group V4) |
| Model Registry | Hopsworks |
| Experiment Tracking | MLflow |
| Backend API | FastAPI |
| Frontend Dashboard | Streamlit + Plotly |
| Explainability | SHAP (TreeExplainer, LinearExplainer) |
| CI/CD | GitHub Actions |
| Containerization | Docker, Docker Compose |
| Deployment | Render (Free Tier) |
| Data Sources | Open-Meteo, AQICN |
