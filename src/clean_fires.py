"""Phase 2b: clean NASA FIRMS MODIS detections and aggregate to daily Punjab+Haryana totals.

Decisions:
- Keep type == 0 (presumed vegetation fire) and confidence >= 30.
- Filter to the Punjab + Haryana bounding box.
- Convert UTC acquisition time to the IST calendar date before aggregating
  (an evening UTC overpass can belong to the next IST day).
- Drop exact duplicate detections (same lat, lon, date, time, satellite).
- Days with no detections are 0, not NaN: absence of fire is information.
"""
import pandas as pd

from config import (END, FIRE_BBOX, FIRE_MIN_CONFIDENCE, FIRES_DAILY, FIRES_POINTS, INTERIM,
                    RAW_FIRMS, START, TZ, YEARS)


def load_raw() -> pd.DataFrame:
    frames = [pd.read_csv(RAW_FIRMS / f"modis_{y}_India.csv", dtype={"acq_time": "string"})
              for y in YEARS]
    return pd.concat(frames, ignore_index=True)


def clean_fires() -> pd.DataFrame:
    df = load_raw()
    n_raw = len(df)
    df["acq_time"] = df.acq_time.str.zfill(4)
    df["confidence"] = pd.to_numeric(df.confidence, errors="coerce")

    df = df[(df.type == 0) & (df.confidence >= FIRE_MIN_CONFIDENCE)]
    n_veg = len(df)
    b = FIRE_BBOX
    df = df[df.latitude.between(b["lat_min"], b["lat_max"])
            & df.longitude.between(b["lon_min"], b["lon_max"])].copy()
    n_box = len(df)

    n_before = len(df)
    df = df.drop_duplicates(["latitude", "longitude", "acq_date", "acq_time", "satellite"])
    n_dups = n_before - len(df)

    utc = pd.to_datetime(df.acq_date + " " + df.acq_time, format="%Y-%m-%d %H%M", utc=True)
    df["date"] = utc.dt.tz_convert(TZ).dt.tz_localize(None).dt.normalize()
    n_shifted = int((df.date.dt.strftime("%Y-%m-%d") != df.acq_date).sum())

    df[["latitude", "longitude", "frp", "confidence", "date", "satellite", "daynight"]].to_parquet(
        FIRES_POINTS)

    daily = df.groupby("date").agg(fire_count=("frp", "size"), fire_frp_sum=("frp", "sum"))
    full = pd.date_range(START, END, freq="D")
    daily = daily.reindex(full, fill_value=0)
    daily.index.name = "date"
    daily["fire_count"] = daily.fire_count.astype("int64")

    log = {"raw_rows": n_raw, "after_type_conf_filter": n_veg, "in_bbox": n_box,
           "exact_duplicates_dropped": n_dups, "ist_date_differs_from_utc": n_shifted,
           "days": len(daily), "days_with_zero_fires": int((daily.fire_count == 0).sum())}
    pd.Series(log).to_json(INTERIM / "fires_cleaning_log.json", indent=2)
    print("  Fires:", log)
    daily.to_parquet(FIRES_DAILY)
    return daily


if __name__ == "__main__":
    clean_fires()
