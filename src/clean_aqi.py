"""Phase 2a: clean the CPCB city_day.csv for Delhi.

Decisions (each justified in the report):
- Duplicated dates: dropped (asserted none remain).
- Physically impossible values (negative, PM2.5/PM10 > 1000): sensor errors -> NaN.
- Extreme but plausible values (Diwali, smog episodes): kept, they are what we want to predict.
- Pollutant columns > 40% missing: dropped.
- Feature imputation is NOT done here: forward-fill (limit 3) happens in features.py on the
  feature copies only, so the target is never imputed.
"""
import pandas as pd

from config import AQI_CLEAN, AQI_RAW, INTERIM, PM25_BINS, PM25_LABELS, PM25_MAX_PLAUSIBLE

POLLUTANTS = ["PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2", "O3",
              "Benzene", "Toluene", "Xylene"]
MAX_MISSING = 0.40


def pm25_category(s: pd.Series) -> pd.Series:
    """Map PM2.5 (ug/m3) to the six CPCB categories."""
    return pd.cut(s, bins=PM25_BINS, labels=PM25_LABELS)


def clean_aqi() -> pd.DataFrame:
    df = pd.read_csv(AQI_RAW, dtype={"City": "string", "AQI_Bucket": "string"})
    df = df[df.City == "Delhi"].copy()
    df["Date"] = pd.to_datetime(df.Date, format="%Y-%m-%d")
    df[POLLUTANTS + ["AQI"]] = df[POLLUTANTS + ["AQI"]].astype("float64")

    n_dup = df.Date.duplicated().sum()
    df = df.drop_duplicates("Date").sort_values("Date").set_index("Date")
    assert not df.index.duplicated().any()

    # Reindex to a complete daily calendar so that missing days are explicit NaN rows.
    full = pd.date_range(df.index.min(), df.index.max(), freq="D")
    n_missing_days = len(full) - len(df)
    df = df.reindex(full)
    df.index.name = "date"

    # Impossible values -> NaN (count them for the report).
    n_neg = int((df[POLLUTANTS] < 0).sum().sum())
    df[POLLUTANTS] = df[POLLUTANTS].mask(df[POLLUTANTS] < 0)
    n_high = int((df[["PM2.5", "PM10"]] > PM25_MAX_PLAUSIBLE).sum().sum())
    df[["PM2.5", "PM10"]] = df[["PM2.5", "PM10"]].mask(df[["PM2.5", "PM10"]] > PM25_MAX_PLAUSIBLE)

    missing = df[POLLUTANTS].isna().mean().sort_values(ascending=False).rename("frac_missing")
    missing.to_frame().to_csv(INTERIM / "aqi_missingness.csv")
    drop = [c for c in POLLUTANTS if missing[c] > MAX_MISSING and c != "PM2.5"]
    df = df.drop(columns=drop + ["City", "AQI_Bucket"])

    df = df.rename(columns={"PM2.5": "pm25", "PM10": "pm10"}).rename(columns=str.lower)
    df["pm25_category"] = pm25_category(df.pm25)

    log = {
        "rows": len(df), "start": str(df.index.min().date()), "end": str(df.index.max().date()),
        "duplicate_dates_dropped": int(n_dup), "calendar_days_added": int(n_missing_days),
        "negative_set_nan": n_neg, "above_1000_set_nan": n_high,
        "columns_dropped_gt40pct_missing": drop,
        "pm25_missing_frac": round(float(df.pm25.isna().mean()), 4),
    }
    pd.Series(log, dtype=object).to_json(INTERIM / "aqi_cleaning_log.json", indent=2)
    print("  AQI:", log)
    df.to_parquet(AQI_CLEAN)
    return df


if __name__ == "__main__":
    clean_aqi()
