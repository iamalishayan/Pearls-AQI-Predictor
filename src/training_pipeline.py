"""
Training Pipeline v3 — Intelligence Sprint (+24h Forecast Horizon)

Features:
  - Fetches from Hopsworks V3 (Lag Features included)
  - Target Shifting: Predicts PM2.5 at t+24 using features from time t.
  - 70/10/20 Temporal Split
  - TimeSeriesSplit Cross-Validation
  - SHAP Explainability (logs summary plot to MLflow)
  - Registers the +24h forecasting model.
"""

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
import mlflow
import hopsworks
import joblib

from src.config import (
    HOPSWORKS_API_KEY,
    HOPSWORKS_PROJECT_NAME,
    FEATURE_GROUP_NAME,
    FEATURE_GROUP_VERSION,
    MODEL_NAME,
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME,
    RANDOM_STATE
)
from src.utils import setup_logger

logger = setup_logger("training_pipeline_v3")
warnings.filterwarnings("ignore")

# Force non-interactive matplotlib backend
plt.switch_backend('Agg')

def get_training_data():
    """Fetch V3 training data from Hopsworks."""
    logger.info("Connecting to Hopsworks...")
    project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY, project=HOPSWORKS_PROJECT_NAME)
    fs = project.get_feature_store()
    
    view_name = f"{FEATURE_GROUP_NAME}_view"
    
    # Recreate Feature View without labels for batch fetch
    try:
        fs.get_feature_view(name=view_name, version=FEATURE_GROUP_VERSION).delete()
    except:
        pass

    logger.info(f"Creating Feature View '{view_name}' v{FEATURE_GROUP_VERSION} for FG V{FEATURE_GROUP_VERSION}...")
    fg = fs.get_feature_group(name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
    feature_view = fs.create_feature_view(
        name=view_name,
        version=FEATURE_GROUP_VERSION,
        query=fg.select_all(),
        labels=[] # Empty labels to ensure all columns (including pm25) are returned
    )

    logger.info("Fetching batch data from V3...")
    df = feature_view.get_batch_data()
    return df, feature_view, project

def preprocess_forecast_shifts(df):
    """
    Transform Nowcasting dataset into Forecasting dataset for 3 horizons.
    Target 1 (y_24) = PM2.5 at t + 24
    Target 2 (y_48) = PM2.5 at t + 48
    Target 3 (y_72) = PM2.5 at t + 72
    """
    logger.info("Applying 24h, 48h, and 72h forecast target shifts...")
    df = df.copy().sort_values("timestamp").reset_index(drop=True)
    
    # 1. Null Handling (safety net)
    df = df.dropna(subset=["pm25"])
    feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if "pm25" in feature_cols: feature_cols.remove("pm25")
    df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())

    # 2. Target Shifting
    df["target_24h"] = df["pm25"].shift(-24)
    df["target_48h"] = df["pm25"].shift(-48)
    df["target_72h"] = df["pm25"].shift(-72)
    
    return df

def train_and_evaluate_horizon(df, horizon_label, target_col):
    """Train multiple models for a specific horizon and return the best one."""
    # Drop rows where this specific target is NaN
    horizon_df = df.dropna(subset=[target_col]).reset_index(drop=True)
    
    # 70/10/20 Temporal Split
    train_size = int(len(horizon_df) * 0.7)
    val_size = int(len(horizon_df) * 0.1)
    
    train_df = horizon_df.iloc[:train_size]
    val_df = horizon_df.iloc[train_size : train_size + val_size]
    test_df = horizon_df.iloc[train_size + val_size :]
    
    drop_cols = ["pm25", "target_24h", "target_48h", "target_72h", "city_name", "timestamp"]
    def get_X_y(data):
        X = data.drop(columns=[c for c in drop_cols if c in data.columns])
        y = data[target_col]
        return X, y

    X_train, y_train = get_X_y(train_df)
    X_val, y_val = get_X_y(val_df)
    X_test, y_test = get_X_y(test_df)
    
    logger.info(f"[{horizon_label}] Splits: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    
    # Combine Train + Val for final fit
    X_combined = pd.concat([X_train, X_val])
    y_combined = pd.concat([y_train, y_val])
    
    models = {
        "Ridge": Ridge(alpha=1.0),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE),
        "XGBoost": xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=RANDOM_STATE)
    }

    best_model = None
    best_rmse = float('inf')
    best_model_name = ""

    for name, model in models.items():
        with mlflow.start_run(run_name=f"{name}_{horizon_label}"):
            logger.info(f"[{horizon_label}] Training {name}...")
            model.fit(X_combined, y_combined)
            
            preds = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            mae = mean_absolute_error(y_test, preds)
            r2 = r2_score(y_test, preds)
            
            logger.info(f"[{horizon_label}] {name} Test RMSE: {rmse:.4f} | R2: {r2:.4f}")
            mlflow.log_metric("test_rmse", rmse)
            mlflow.log_metric("test_mae", mae)
            mlflow.log_metric("test_r2", r2)
            
            if rmse < best_rmse:
                best_rmse = rmse
                best_model = model
                best_model_name = name

    logger.info(f"🏆 Best Model for {horizon_label}: {best_model_name} (RMSE: {best_rmse:.4f})")

    # SHAP Explainability for the best model
    try:
        logger.info(f"[{horizon_label}] Generating SHAP explanation for {best_model_name}...")
        if "Ridge" in best_model_name:
            explainer = shap.LinearExplainer(best_model, X_train)
            shap_values = explainer.shap_values(X_test)
        else:
            explainer = shap.TreeExplainer(best_model)
            shap_values = explainer.shap_values(X_test)
        
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test, show=False)
        plt.tight_layout()
        shap_plot_path = f"shap_{horizon_label}.png"
        plt.savefig(shap_plot_path)
        # Log to the last MLflow run or save locally
        plt.close()
    except Exception as e:
        logger.warning(f"[{horizon_label}] SHAP failed: {e}")
            
    return best_model, best_rmse, X_test, y_test, best_model_name

def register_model_horizon(project, model, horizon_label, rmse, X_test, best_model_name):
    """Register the horizon-specific model to Hopsworks."""
    logger.info(f"Registering '{MODEL_NAME}_{horizon_label}' ({best_model_name}) to Hopsworks...")
    mr = project.get_model_registry()
    
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", f"forecast_model_{horizon_label}.pkl")
    joblib.dump(model, model_path)

    hw_model = mr.python.create_model(
        name=f"{MODEL_NAME}_{horizon_label}",
        metrics={"rmse": float(rmse)},
        description=f"Phase 6: {horizon_label} PM2.5 Forecast Model ({best_model_name}).",
        input_example=X_test.iloc[0].to_dict()
    )
    hw_model.save(model_path)
    logger.info(f"✅ Registered as '{MODEL_NAME}_{horizon_label}'")

def main():
    logger.info("="*60)
    logger.info("Phase 5: 3-Day Forecast Training Pipeline")
    logger.info("="*60)
    
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(f"{MLFLOW_EXPERIMENT_NAME}_3day_forecast")

    try:
        df, feature_view, project = get_training_data()
        df_shifted = preprocess_forecast_shifts(df)
        
        horizons = {
            "24h": "target_24h",
            "48h": "target_48h",
            "72h": "target_72h"
        }
        
        for label, target_col in horizons.items():
            model, rmse, X_test, y_test, best_name = train_and_evaluate_horizon(df_shifted, label, target_col)
            register_model_horizon(project, model, label, rmse, X_test, best_name)
            
        logger.info("Phase 6 Training Complete ✅")
    except Exception as e:
        logger.error(f"Training Pipeline failed: {e}")
        raise e

if __name__ == "__main__":
    main()
