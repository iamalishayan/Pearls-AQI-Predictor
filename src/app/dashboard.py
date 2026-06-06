"""
Streamlit Dashboard — Phase 4 Intelligence UI

Displays:
  1. Headline Forecast: Tomorrow's PM2.5 and AQI Category
  2. Explainability: SHAP Feature Importance Drivers
  3. Reality Check: Live vs Forecast
"""

import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import time
from datetime import datetime

# Configure page
st.set_page_config(
    page_title="Islamabad Air Quality Forecast",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .metric-card {
        background-color: #1e1e1e;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        text-align: center;
        border-left: 5px solid #00c853; /* Default green, updated dynamically */
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 10px 0;
    }
    .metric-label {
        font-size: 1rem;
        color: #aaaaaa;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .category-badge {
        display: inline-block;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: bold;
        margin-top: 10px;
    }
    /* AQI Colors */
    .aqi-good { background-color: #00e400; color: #000; }
    .aqi-moderate { background-color: #ffff00; color: #000; }
    .aqi-sg { background-color: #ff7e00; color: #fff; }
    .aqi-unhealthy { background-color: #ff0000; color: #fff; }
    .aqi-very { background-color: #8f3f97; color: #fff; }
    .aqi-haz { background-color: #7e0023; color: #fff; }
</style>
""", unsafe_allow_html=True)
import os

# API URL: configurable for Docker Compose (api:8000), Render (localhost:8000), or local dev
API_URL = os.getenv("API_URL", "http://api:8000")

def get_aqi_style(category: str) -> tuple:
    """Return color and CSS class based on AQI category."""
    mapping = {
        "Good": ("#00e400", "aqi-good"),
        "Moderate": ("#ffff00", "aqi-moderate"),
        "Unhealthy for Sensitive Groups": ("#ff7e00", "aqi-sg"),
        "Unhealthy": ("#ff0000", "aqi-unhealthy"),
        "Very Unhealthy": ("#8f3f97", "aqi-very"),
        "Hazardous": ("#7e0023", "aqi-haz"),
    }
    return mapping.get(category, ("#888888", ""))

def fetch_prediction():
    """Fetch data from FastAPI backend."""
    try:
        # For local development outside docker, try localhost if api fails
        url = f"{API_URL}/predict"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        try:
            url = "http://localhost:8000/predict"
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"Failed to connect to backend API: {e}")
            return None
    except Exception as e:
        st.error(f"API Error: {e}")
        return None

# Sidebar
with st.sidebar:
    st.title("AQI Predictor")
    st.write("Intelligent 24-hour Air Quality Forecasting System for Islamabad, Pakistan.")
    
    st.markdown("---")
    st.subheader("System Status")
    status = st.empty()
    
    if st.button("🔄 Refresh Forecast", use_container_width=True):
        st.rerun()
        
    st.markdown("---")
    st.caption("Built with FastAPI, Streamlit, and Hopsworks ML")

# Main Content
st.title("Islamabad Air Quality Forecast")
st.markdown("Predicting pollution levels for the next 3 days using Machine Learning & SHAP Explainability.")

# Fetch Data
with st.spinner("Analyzing atmospheric data and generating 3-day forecast..."):
    data = fetch_prediction()

if data:
    status.success("🟢 API Connected")
    
    # Check for Active Alerts
    alert_info = data.get("alert", {})
    if alert_info.get("is_active"):
        st.error(f"**🚨 {alert_info.get('level')} ALERT:** {alert_info.get('message')}  \n**Recommended Action:** {alert_info.get('recommended_action')}", icon="🚨")
        
    st.markdown(f"### Current AQI (Reality Check): **{data['current_aqi']}** (PM2.5: {data['current_pm25']} µg/m³)")
    st.markdown("---")
    
    # Extract forecasts
    forecasts = data.get("forecasts", [])
    if not forecasts:
        st.warning("No forecast data available.")
    else:
        # 1. Headline Metrics (3 Columns for 3 Days)
        cols = st.columns(3)
        for i, fcast in enumerate(forecasts):
            horizon_label = fcast["horizon"]
            forecast_time = datetime.fromisoformat(fcast["forecast_time"]).strftime("%A, %b %d (%I %p)")
            aqi_cat = fcast["aqi_category"]
            color, css_class = get_aqi_style(aqi_cat)
            
            with cols[i]:
                st.markdown(f"#### +{horizon_label} Forecast")
                st.caption(f"{forecast_time}")
                st.markdown(f"""
                <div class="metric-card" style="border-left-color: {color};">
                    <div class="metric-label">Forecasted AQI</div>
                    <div class="metric-value">{fcast['aqi_forecast']}</div>
                    <div class="category-badge {css_class}">{aqi_cat}</div>
                    <div style="margin-top:10px; color:#aaa; font-size:0.9rem;">PM2.5: {fcast['pm25_forecast']} µg/m³</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")
        
        # 2. Intelligence Layer: Explainability (Tabs for each day)
        st.subheader("System Intelligence: What's driving the forecast?")
        st.write("The model analyzed these top atmospheric drivers for each forecast horizon:")
        
        tabs = st.tabs([f"+{f['horizon']}" for f in forecasts])
        
        for i, fcast in enumerate(forecasts):
            with tabs[i]:
                drivers = fcast.get("top_drivers", [])
                if drivers:
                    df_drivers = pd.DataFrame(drivers)
                    df_drivers['abs_impact'] = df_drivers['impact'].abs()
                    df_drivers = df_drivers.sort_values('abs_impact', ascending=True)
                    
                    colors = ['#ff4b4b' if x > 0 else '#00c853' for x in df_drivers['impact']]
                    
                    fig = go.Figure(go.Bar(
                        x=df_drivers['impact'],
                        y=df_drivers['feature'],
                        orientation='h',
                        marker_color=colors,
                        text=[f"{'+' if x>0 else ''}{x:.2f} µg/m³" for x in df_drivers['impact']],
                        textposition='auto'
                    ))
                    
                    fig.update_layout(
                        title=f"SHAP Feature Impact for {fcast['horizon']}",
                        xaxis_title="Impact on PM2.5 (µg/m³)",
                        yaxis_title="Feature",
                        height=400,
                        margin=dict(l=20, r=20, t=40, b=20),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.info(f"💡 **Interpretation:** The biggest driver for the {fcast['horizon']} forecast is **{df_drivers.iloc[-1]['feature']}**. "
                            f"It is pushing the PM2.5 prediction {'UP' if df_drivers.iloc[-1]['impact'] > 0 else 'DOWN'} "
                            f"by {abs(df_drivers.iloc[-1]['impact']):.1f} µg/m³.")
                else:
                    st.warning(f"Explainability data not available for {fcast['horizon']}.")

    # 3. Raw Data Expander
    with st.expander("🔬 View Raw Model Inputs"):
        st.json(data["all_features"])

else:
    status.error("🔴 API Disconnected")
    st.error("Could not fetch the forecast. Please ensure the FastAPI backend is running.")

