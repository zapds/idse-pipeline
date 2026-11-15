"""Phase 2c: turn the Open-Meteo JSON into a daily weather table for Delhi.

Hourly boundary-layer height (BLH) is aggregated to a daily mean and minimum: a shallow
mixing layer traps pollution near the ground, and the night-time minimum captures inversions.
"""
import json

import pandas as pd

from config import INTERIM, WEATHER_CLEAN, WEATHER_RAW

RENAME = {
    "temperature_2m_mean": "temp", "relative_humidity_2m_mean": "humidity",
    "wind_speed_10m_max": "wind_speed", "wind_direction_10m_dominant": "wind_dir",
    "precipitation_sum": "precip",
}


def clean_weather() -> pd.DataFrame:
    raw = json.loads(WEATHER_RAW.read_text())
    daily = pd.DataFrame(raw["daily"]).rename(columns=RENAME)
    daily["date"] = pd.to_datetime(daily.pop("time"))
    daily = daily.set_index("date").astype("float64")

    hourly = pd.DataFrame(raw["hourly"])
    hourly["time"] = pd.to_datetime(hourly.time)
    blh = hourly.set_index("time").boundary_layer_height.resample("D").agg(["mean", "min"])
    blh.columns = ["blh_mean", "blh_min"]
    df = daily.join(blh)

    gaps = df.isna().sum()
    full = pd.date_range(df.index.min(), df.index.max(), freq="D")
    assert len(full) == len(df), "missing calendar days in ERA5"
    assert gaps.sum() == 0, f"unexpected gaps in ERA5: {gaps[gaps > 0].to_dict()}"

    log = {"days": len(df), "start": str(df.index.min().date()), "end": str(df.index.max().date()),
           "missing_values": int(gaps.sum())}
    pd.Series(log).to_json(INTERIM / "weather_cleaning_log.json", indent=2)
    print("  Weather:", log)
    df.to_parquet(WEATHER_CLEAN)
    return df


if __name__ == "__main__":
    clean_weather()
