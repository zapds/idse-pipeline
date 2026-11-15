"""Central configuration: paths, seeds, date splits, region and thresholds.

Every other module imports from here so that a single edit changes the whole pipeline.
"""
from pathlib import Path

import pandas as pd

SEED = 42

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
RAW_FIRMS = RAW / "firms"
RAW_WEATHER = RAW / "weather"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
METRICS = REPORTS / "metrics"
MODELS_DIR = ROOT / "models"

AQI_RAW = RAW / "city_day.csv"
WEATHER_RAW = RAW_WEATHER / "open_meteo_delhi.json"
AQI_CLEAN = INTERIM / "aqi_delhi.parquet"
FIRES_POINTS = INTERIM / "fires_points.parquet"
FIRES_DAILY = INTERIM / "fires_daily.parquet"
WEATHER_CLEAN = INTERIM / "weather_delhi.parquet"
MODEL_TABLE = PROCESSED / "delhi_daily.parquet"
DAILY_INPUTS = PROCESSED / "daily_inputs.parquet"

for _p in (RAW_FIRMS, RAW_WEATHER, INTERIM, PROCESSED, FIGURES, METRICS, MODELS_DIR):
    _p.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- time
START = "2015-01-01"
END = "2020-07-01"
YEARS = list(range(2015, 2021))

TRAIN_END = pd.Timestamp("2017-12-31")
VAL_END = pd.Timestamp("2018-12-31")
TEST_END = pd.Timestamp("2019-12-31")

# Fire (burning) season used for the seasonal metrics: 15 Oct - 30 Nov.
FIRE_SEASON = ((10, 15), (11, 30))

# ---------------------------------------------------------------- location
DELHI_LAT, DELHI_LON = 28.61, 77.21
TZ = "Asia/Kolkata"
# Punjab + Haryana bounding box (approximate; includes slivers of neighbouring states).
FIRE_BBOX = dict(lat_min=27.6, lat_max=32.6, lon_min=73.8, lon_max=77.6)
FIRE_MIN_CONFIDENCE = 30

# ---------------------------------------------------------------- thresholds
PM25_MAX_PLAUSIBLE = 1000.0
SEVERE = 250.0
VERY_POOR = 120.0
# CPCB PM2.5 (24h) categories: upper bound inclusive.
PM25_BINS = [-0.1, 30, 60, 90, 120, 250, float("inf")]
PM25_LABELS = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]

# Diwali (Lakshmi Puja) dates, checked against published calendars.
DIWALI = pd.to_datetime([
    "2015-11-11", "2016-10-30", "2017-10-19", "2018-11-07", "2019-10-27", "2020-11-14",
])

N_BOOTSTRAP = 2000
