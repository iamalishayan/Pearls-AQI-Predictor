"""
Alerting System — AQI threshold monitoring & notifications.

Responsibilities:
  1. Check predicted AQI against EPA category thresholds
  2. Generate color-coded warnings for the dashboard
  3. (Optional) Send email/SMS alerts via SendGrid

AQI Categories (EPA Standard):
  - Good:                          0 - 50
  - Moderate:                      51 - 100
  - Unhealthy for Sensitive Groups: 101 - 150
  - Unhealthy:                     151 - 200
  - Very Unhealthy:                201 - 300
  - Hazardous:                     301 - 500

Usage:
  Called by src/app/dashboard.py
"""

def check_alerts(predicted_aqi: float) -> dict:
    """Check AQI value and return alert info (category, color, message)."""
    aqi = int(predicted_aqi)
    alert = {
        "is_active": False,
        "level": "NONE",
        "message": "Air quality is acceptable.",
        "recommended_action": "None"
    }

    if aqi > 300:
        alert = {
            "is_active": True,
            "level": "EMERGENCY",
            "message": f"HAZARDOUS AIR QUALITY (AQI: {aqi}). Health warning of emergency conditions.",
            "recommended_action": "Everyone should avoid all outdoor exertion."
        }
    elif aqi > 200:
        alert = {
            "is_active": True,
            "level": "CRITICAL",
            "message": f"VERY UNHEALTHY AIR QUALITY (AQI: {aqi}). Health alert.",
            "recommended_action": "Everyone should avoid prolonged or heavy exertion."
        }
    elif aqi > 150:
        alert = {
            "is_active": True,
            "level": "HIGH",
            "message": f"UNHEALTHY AIR QUALITY (AQI: {aqi}).",
            "recommended_action": "Everyone may begin to experience health effects; sensitive groups may experience more serious effects."
        }
    elif aqi > 100:
        alert = {
            "is_active": True,
            "level": "WARNING",
            "message": f"Air quality is UNHEALTHY FOR SENSITIVE GROUPS (AQI: {aqi}).",
            "recommended_action": "People with respiratory or heart disease, the elderly, and children should limit prolonged exertion."
        }

    return alert
