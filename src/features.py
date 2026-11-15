"""Phase 4: merge the three clean tables and engineer past-only features.

Row t holds features known by the END of day t and the target pm25 on day t+1.
Rules that prevent leakage:
- Lags and rolling windows only look backwards (shift >= 0, trailing windows).
- Feature gaps are forward-filled (limit 3 days); there is no interpolation, which would
  use future values.
- The target is never imputed: rows whose next-day PM2.5 is missing are dropped.
- Weather is day-t observed weather, not day-t+1 weather (that would be a perfect forecast).
- Calendar features describe the target day t+1. The calendar is known in advance, so this
  is not leakage, and Diwali night is the day the fireworks happen.
"""
import numpy as np
import pandas as pd

from config import (AQI_CLEAN, DAILY_INPUTS, DIWALI, FIRE_SEASON, FIRES_DAILY, MODEL_TABLE,
                    WEATHER_CLEAN)

FFILL_LIMIT = 3

PERSISTENCE = ["pm25_t", "pm25_lag1", "pm25_lag2", "pm25_lag3", "pm25_roll3_mean",
               "pm25_roll7_mean", "pm10_t", "no2_t", "co_t"]
WEATHER = ["wind_speed", "wind_dir_sin", "wind_dir_cos", "nw_wind", "temp", "humidity",
           "precip", "blh_mean", "blh_min"]
CALENDAR = ["month_sin", "month_cos", "dow", "is_diwali_window", "is_fire_season"]
FIRES = ["fire_count_t", "fire_count_lag1", "fire_count_lag2", "fire_count_lag3",
         "fire_roll3_sum", "fire_roll7_sum", "log_fire_count_t", "log_fire_roll3_sum",
         "fire_frp_sum_t", "fire_roll3_x_nw"]
BASE_FEATURES = PERSISTENCE + WEATHER + CALENDAR
ALL_FEATURES = BASE_FEATURES + FIRES
TARGET = "pm25_next"


def in_fire_season(dates: pd.DatetimeIndex) -> np.ndarray:
    """True for dates between 15 Oct and 30 Nov (inclusive)."""
    (m0, d0), (m1, d1) = FIRE_SEASON
    md = dates.month * 100 + dates.day
    return (md >= m0 * 100 + d0) & (md <= m1 * 100 + d1)


def diwali_window(dates: pd.DatetimeIndex, days: int = 2) -> np.ndarray:
    """True within +/- `days` of a Diwali date."""
    dist = np.min(np.abs((dates.values[:, None] - DIWALI.values[None, :])
                         .astype("timedelta64[D]").astype(int)), axis=1)
    return dist <= days


def build_features(aqi: pd.DataFrame, fires: pd.DataFrame, weather: pd.DataFrame,
                   horizon: int = 1) -> pd.DataFrame:
    """Merge the daily tables and return features (day t) plus the target (pm25 at t+horizon).

    The main analysis uses horizon=1; robustness.py also uses 2 and 3 days ahead.
    The target column is always called TARGET so downstream code does not change.

    Pure function of its inputs so the leakage test can perturb the future and re-run it.
    """
    idx = pd.date_range(max(aqi.index.min(), weather.index.min()),
                        min(aqi.index.max(), weather.index.max()), freq="D", name="date")
    aqi, fires, weather = aqi.reindex(idx), fires.reindex(idx), weather.reindex(idx)
    out = pd.DataFrame(index=idx)

    # Target: raw (never imputed) PM2.5 `horizon` days ahead.
    out[TARGET] = aqi.pm25.shift(-horizon)
    out["pm25_t_raw"] = aqi.pm25  # unimputed, kept for reference / EDA

    # --- persistence (forward-filled copies only)
    pm = aqi.pm25.ffill(limit=FFILL_LIMIT)
    out["pm25_t"] = pm
    for k in (1, 2, 3):
        out[f"pm25_lag{k}"] = pm.shift(k)
    out["pm25_roll3_mean"] = pm.rolling(3, min_periods=2).mean()
    out["pm25_roll7_mean"] = pm.rolling(7, min_periods=4).mean()
    for col in ("pm10", "no2", "co"):
        out[f"{col}_t"] = aqi[col].ffill(limit=FFILL_LIMIT) if col in aqi else np.nan

    # --- weather on day t
    rad = np.deg2rad(weather.wind_dir)
    out["wind_speed"] = weather.wind_speed
    out["wind_dir_sin"], out["wind_dir_cos"] = np.sin(rad), np.cos(rad)
    # Wind blowing FROM the north-west quadrant, i.e. from Punjab/Haryana towards Delhi.
    out["nw_wind"] = weather.wind_dir.between(270, 360).astype(int)
    for c in ("temp", "humidity", "precip", "blh_mean", "blh_min"):
        out[c] = weather[c]

    # --- calendar of the target day t+horizon (known in advance)
    tgt = idx + pd.Timedelta(days=horizon)
    out["month_sin"] = np.sin(2 * np.pi * tgt.month / 12)
    out["month_cos"] = np.cos(2 * np.pi * tgt.month / 12)
    out["dow"] = tgt.day_of_week
    out["is_diwali_window"] = diwali_window(tgt).astype(int)
    out["is_fire_season"] = in_fire_season(tgt).astype(int)

    # --- fires (the tested group); absence of detections is a real 0
    fc = fires.fire_count.fillna(0)
    out["fire_count_t"] = fc
    for k in (1, 2, 3):
        out[f"fire_count_lag{k}"] = fc.shift(k)
    out["fire_roll3_sum"] = fc.rolling(3, min_periods=1).sum()
    out["fire_roll7_sum"] = fc.rolling(7, min_periods=1).sum()
    out["log_fire_count_t"] = np.log1p(fc)
    out["log_fire_roll3_sum"] = np.log1p(out.fire_roll3_sum)
    out["fire_frp_sum_t"] = fires.fire_frp_sum.fillna(0)
    # Smoke only reaches Delhi when the wind carries it there.
    out["fire_roll3_x_nw"] = out.log_fire_roll3_sum * out.nw_wind

    out["target_date"] = tgt
    return out


def make_model_table() -> pd.DataFrame:
    """Build features from the interim tables and keep rows usable for modelling."""
    aqi = pd.read_parquet(AQI_CLEAN)
    fires = pd.read_parquet(FIRES_DAILY)
    weather = pd.read_parquet(WEATHER_CLEAN)
    df = build_features(aqi, fires, weather)
    # Raw daily inputs (what a user would type in) for the inference website's "load a real day".
    aqi[["pm25", "pm10", "no2", "co"]].join(fires).join(weather).to_parquet(DAILY_INPUTS)
    n0 = len(df)
    # Drop rows with no target (never impute it) or no current PM2.5 (persistence undefined),
    # so every model, including the baseline, is scored on exactly the same days.
    df = df.dropna(subset=[TARGET, "pm25_t"])
    print(f"  Model table: {len(df)} rows ({n0 - len(df)} dropped for missing target/pm25_t), "
          f"{df.index.min().date()} -> {df.index.max().date()}")
    df.to_parquet(MODEL_TABLE)
    return df


if __name__ == "__main__":
    make_model_table()
