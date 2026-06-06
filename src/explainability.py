"""
Explainability Module — SHAP & LIME analysis.

Responsibilities:
  1. Generate SHAP values for the best model
  2. Generate LIME explanations for individual predictions
  3. Create feature importance visualizations

Usage:
  Called by the training pipeline and the Streamlit dashboard.
"""

import shap
import numpy as np
import pandas as pd

def get_top_drivers(model, X: pd.DataFrame, top_n: int = 5) -> list:
    """
    Generate SHAP explanation for a prediction.
    Returns a list of the top N driving features and their impacts.
    """
    top_drivers = []
    
    # Determine the correct explainer based on model type
    if hasattr(model, 'feature_importances_'):
        explainer = shap.TreeExplainer(model)
    else:
        explainer = shap.LinearExplainer(model, X)
        
    shap_values = explainer.shap_values(X)

    feature_names = X.columns.tolist()
    
    # Handle SHAP output shapes gracefully
    shap_vals = shap_values[0] if isinstance(shap_values, list) else shap_values.flatten()
    
    # Get indices of top N absolute impacts
    indices = np.argsort(np.abs(shap_vals))[::-1][:top_n]
    
    for idx in indices:
        top_drivers.append({
            "feature": feature_names[idx],
            "impact": round(float(shap_vals[idx]), 3),
            "value": round(float(X.iloc[0, idx]), 3),
        })
        
    return top_drivers
