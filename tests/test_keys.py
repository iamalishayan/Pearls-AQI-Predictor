"""
API Key Tester — Validates AQICN and Hopsworks credentials.
Run: python test_keys.py
"""

import os
import json
from dotenv import load_dotenv
import requests
import warnings
warnings.filterwarnings("ignore")

load_dotenv()

# ── Colors ────────────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

def ok(msg):    print(f"  {GREEN}✅ {msg}{RESET}")
def fail(msg):  print(f"  {RED}❌ {msg}{RESET}")
def warn(msg):  print(f"  {YELLOW}⚠️  {msg}{RESET}")
def header(msg):print(f"\n{BOLD}{msg}{RESET}")

# ══════════════════════════════════════════════════════════════════════════════
# 1. AQICN API
# ══════════════════════════════════════════════════════════════════════════════
header("1. Testing AQICN API Key...")

aqicn_key = os.getenv("AQICN_API_KEY", "")
city_name = os.getenv("CITY_NAME", "islamabad")

if not aqicn_key:
    fail("AQICN_API_KEY is not set in .env")
else:
    url = f"https://api.waqi.info/feed/{city_name}/?token={aqicn_key}"
    try:
        resp = requests.get(url, timeout=10)
        data = resp.json()
        status = data.get("status")
        if status == "ok":
            aqi  = data["data"]["aqi"]
            city = data["data"]["city"]["name"]
            ok(f"AQICN key is VALID")
            ok(f"City resolved: {city}")
            ok(f"Current live AQI: {aqi}")
        elif status == "error":
            msg = data.get("data", "Unknown error")
            fail(f"AQICN API returned error: {msg}")
        else:
            warn(f"Unexpected response: {data}")
    except Exception as e:
        fail(f"Request failed: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# 2. Hopsworks
# ══════════════════════════════════════════════════════════════════════════════
header("2. Testing Hopsworks API Key...")

hw_key     = os.getenv("HOPSWORKS_API_KEY", "")
hw_project = os.getenv("HOPSWORKS_PROJECT_NAME", "")

if not hw_key:
    fail("HOPSWORKS_API_KEY is not set in .env")
elif not hw_project:
    fail("HOPSWORKS_PROJECT_NAME is not set in .env")
else:
    try:
        import hopsworks
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            project = hopsworks.login(api_key_value=hw_key)
        ok(f"Hopsworks key is VALID")
        ok(f"Connected to project: {project.name}")
        fs = project.get_feature_store()
        ok(f"Feature Store accessible: {fs.name}")
    except ModuleNotFoundError as e:
        warn(f"Missing dependency: {e} — run: pip install 'hopsworks[python]'")
    except Exception as e:
        fail(f"Hopsworks connection failed: {e}")

# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{BOLD}Done.{RESET}\n")
