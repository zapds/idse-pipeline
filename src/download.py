"""Phase 1: fetch raw data.

- NASA FIRMS MODIS C6.1 yearly country files for India (no key needed).
- Open-Meteo ERA5 archive for Delhi: daily weather + hourly boundary-layer height.
- CPCB city_day.csv from Kaggle: only checked, because it needs the user's Kaggle token.

Already-downloaded files are skipped so the script is cheap to rerun.
"""
import json
import sys

import pandas as pd
import requests

from config import (AQI_RAW, DELHI_LAT, DELHI_LON, END, RAW_FIRMS, START, TZ,
                    WEATHER_RAW, YEARS)

FIRMS_URL = "https://firms.modaps.eosdis.nasa.gov/data/country/modis/{y}/modis_{y}_India.csv"
METEO_URL = "https://archive-api.open-meteo.com/v1/archive"
DAILY_VARS = [
    "temperature_2m_mean", "relative_humidity_2m_mean", "wind_speed_10m_max",
    "wind_direction_10m_dominant", "precipitation_sum",
]


def download_firms() -> None:
    """Download one MODIS CSV per year into data/raw/firms/ (skip if present)."""
    for y in YEARS:
        out = RAW_FIRMS / f"modis_{y}_India.csv"
        if out.exists() and out.stat().st_size > 0:
            continue
        print(f"  downloading FIRMS {y} ...")
        r = requests.get(FIRMS_URL.format(y=y), timeout=300)
        r.raise_for_status()
        out.write_bytes(r.content)


def download_weather() -> None:
    """Download Delhi weather from the Open-Meteo historical archive as raw JSON."""
    if WEATHER_RAW.exists():
        return
    print("  downloading Open-Meteo weather ...")
    params = {
        "latitude": DELHI_LAT, "longitude": DELHI_LON,
        "start_date": START, "end_date": END,
        "daily": ",".join(DAILY_VARS),
        "hourly": "boundary_layer_height",
        "timezone": TZ,
        "wind_speed_unit": "kmh",
    }
    r = requests.get(METEO_URL, params=params, timeout=300)
    r.raise_for_status()
    WEATHER_RAW.write_text(json.dumps(r.json()))


def check_aqi() -> None:
    """Stop with a clear instruction if the Kaggle file has not been downloaded."""
    if not AQI_RAW.exists():
        print(f"Missing {AQI_RAW}. Download it with:\n"
              "  kaggle datasets download -d rohanrao/air-quality-data-in-india "
              f"-f city_day.csv -p {AQI_RAW.parent} && unzip {AQI_RAW}.zip -d {AQI_RAW.parent}")
        sys.exit(1)


def log_raw_files() -> None:
    """Print row counts and date ranges of every raw file."""
    aqi = pd.read_csv(AQI_RAW, usecols=["City", "Date"])
    d = aqi[aqi.City == "Delhi"].Date
    print(f"  city_day.csv: {len(aqi):,} rows, Delhi {len(d):,} rows, {d.min()} -> {d.max()}")
    for y in YEARS:
        f = pd.read_csv(RAW_FIRMS / f"modis_{y}_India.csv", usecols=["acq_date"])
        print(f"  modis_{y}_India.csv: {len(f):,} rows, {f.acq_date.min()} -> {f.acq_date.max()}")
    w = json.loads(WEATHER_RAW.read_text())
    print(f"  open_meteo: {len(w['daily']['time'])} days, {len(w['hourly']['time'])} hours, "
          f"{w['daily']['time'][0]} -> {w['daily']['time'][-1]}")


def main() -> None:
    check_aqi()
    download_firms()
    download_weather()
    log_raw_files()


if __name__ == "__main__":
    main()
