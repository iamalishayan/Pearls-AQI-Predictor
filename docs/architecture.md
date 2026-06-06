# System Architecture

## High-Level Overview

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   External APIs  │────▶│  Feature Pipeline │────▶│  Feature Store  │
│  (Open-Meteo)   │     │   (Hourly CI/CD)  │     │  (Hopsworks)    │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
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
                                                 │  (Hopsworks)     │
                                                 └────────┬─────────┘
                                                          │
                                                          ▼
                        ┌──────────────────┐     ┌──────────────────┐
                        │ Streamlit Dashboard│◀───│  FastAPI Backend │
                        │     (Docker)       │     │    (Docker)      │
                        └──────────────────┘     └──────────────────┘
```

## Components

| Component | Technology | Trigger |
|---|---|---|
| Feature Pipeline | Python + GitHub Actions | Hourly cron |
| Backfill Pipeline | Python (one-time) | Manual |
| Training Pipeline | Python + GitHub Actions | Daily cron |
| Model Registry | Hopsworks | Updated by training pipeline |
| Feature Store | Hopsworks | Updated by feature pipeline |
| Backend API | FastAPI | Always-on (Docker) |
| Dashboard | Streamlit | Always-on (Docker) |
| Experiment Tracking | MLflow | During training |
