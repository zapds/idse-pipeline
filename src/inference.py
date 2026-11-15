"""Live inference for the website: raw daily inputs -> features -> forecasts from every saved model.

Features are computed by the SAME `build_features` used in training, on a small 7-day window,
so the website cannot drift from the pipeline (tests/test_inference.py checks this).
"""
import json
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from clean_aqi import pm25_category
from config import DAILY_INPUTS, METRICS, MODELS_DIR, SEVERE
from features import ALL_FEATURES, BASE_FEATURES, build_features

WINDOW = 7  # days of history needed (7-day rolling features)
FAMILIES = ["Ridge", "RandomForest", "HistGB"]
VARIANT_FEATURES = {"no_fires": BASE_FEATURES, "with_fires": ALL_FEATURES}
WEATHER_KEYS = ["temp", "humidity", "wind_speed", "wind_dir", "precip", "blh_mean", "blh_min"]
FIRE_MULTIPLIERS = [0, 0.25, 0.5, 1, 2, 3, 4]


@lru_cache(maxsize=1)
def load_models() -> dict:
    """{(horizon, family, variant): model}; horizon 1 also has the two baselines."""
    models = {}
    for h in (1, 2, 3):
        prefix = "" if h == 1 else f"h{h}_"
        for fam in FAMILIES:
            for v in VARIANT_FEATURES:
                path = MODELS_DIR / f"{prefix}{fam}_{v}.joblib"
                if path.exists():
                    models[(h, fam, v)] = joblib.load(path)
    models[(1, "SeasonalMean", "baseline")] = joblib.load(MODELS_DIR / "SeasonalMean.joblib")
    # Single-row requests: thread start-up costs more than it saves, and gunicorn already
    # runs several workers, so predict each forest on one core (about 2.5x faster here).
    for m in models.values():
        est = getattr(m, "regressor_", None)
        if est is not None and "n_jobs" in est[-1].get_params():
            est[-1].set_params(n_jobs=1)
    return models


@lru_cache(maxsize=1)
def selection() -> dict:
    return json.loads((METRICS / "model_selection.json").read_text())


@lru_cache(maxsize=1)
def history() -> pd.DataFrame:
    return pd.read_parquet(DAILY_INPUTS)


def _f(x):
    """Optional numeric input -> float or NaN."""
    return np.nan if x is None or x == "" else float(x)


def validate(inp: dict) -> None:
    for k in ("pm25", "fires"):
        if len(inp.get(k, [])) != WINDOW:
            raise ValueError(f"'{k}' needs {WINDOW} daily values (oldest first, today last)")
    if any(v is None or v == "" for v in inp["pm25"][-1:]):
        raise ValueError("today's PM2.5 is required")
    pm = [_f(v) for v in inp["pm25"]]
    if any(v < 0 or v > 1000 for v in pm if not np.isnan(v)):
        raise ValueError("PM2.5 values must be between 0 and 1000 ug/m3")
    if any(_f(v) < 0 for v in inp["fires"]):
        raise ValueError("fire counts cannot be negative")
    w = inp.get("weather", {})
    missing = [k for k in WEATHER_KEYS if w.get(k) in (None, "")]
    if missing:
        raise ValueError(f"missing weather inputs: {', '.join(missing)}")
    pd.Timestamp(inp["date"])


def features_from_inputs(inp: dict, horizon: int = 1) -> pd.DataFrame:
    """Build the one-row feature frame for day t from raw inputs."""
    validate(inp)
    t = pd.Timestamp(inp["date"]).normalize()
    idx = pd.date_range(t - pd.Timedelta(days=WINDOW - 1), t, freq="D")
    aqi = pd.DataFrame({"pm25": [_f(v) for v in inp["pm25"]],
                        "pm10": np.nan, "no2": np.nan, "co": np.nan}, index=idx)
    for k in ("pm10", "no2", "co"):
        aqi.loc[t, k] = _f(inp.get(k))
    fc = np.array([_f(v) for v in inp["fires"]], dtype=float)
    frp = inp.get("fire_frp_today")
    # FRP is optional: if missing, scale today's count by a typical 25 MW per detection.
    frp_today = _f(frp) if frp not in (None, "") else fc[-1] * 25.0
    fires = pd.DataFrame({"fire_count": fc, "fire_frp_sum": 0.0}, index=idx)
    fires.loc[t, "fire_frp_sum"] = frp_today
    w = inp["weather"]
    weather = pd.DataFrame({k: _f(w[k]) for k in WEATHER_KEYS}, index=idx)
    feats = build_features(aqi, fires, weather, horizon)
    return feats.loc[[t]]


def predict(inp: dict) -> dict:
    """Forecasts for horizons 1-3 from every model, plus the fire what-if curve."""
    models = load_models()
    sel = selection()
    best = sel["selected_model"]
    threshold = sel["severe_alert_threshold_from_val"]["with_fires"]
    out = {"selected_model": best, "alert_threshold": threshold, "horizons": {}}
    for h in (1, 2, 3):
        X = features_from_inputs(inp, h)
        preds = {}
        for (hh, fam, v), m in models.items():
            if hh == h:
                feats = ALL_FEATURES if v == "baseline" else VARIANT_FEATURES[v]
                preds[f"{fam}|{v}"] = max(0.0, float(m.predict(X[feats])[0]))
        preds["Persistence|baseline"] = float(X.pm25_t.iloc[0])
        main = preds.get(f"{best}|with_fires")
        out["horizons"][str(h)] = {
            "target_date": str(X.target_date.iloc[0].date()),
            "predictions": preds,
            "main": main,
            "main_no_fires": preds.get(f"{best}|no_fires"),
            "category": str(pm25_category(pd.Series([main])).iloc[0]) if main is not None else None,
            "is_fire_season": bool(X.is_fire_season.iloc[0]),
            "is_diwali_window": bool(X.is_diwali_window.iloc[0]),
        }
    h1 = out["horizons"]["1"]
    h1["severe_alert"] = h1["main"] > threshold
    out["fire_what_if"] = fire_what_if(inp, best)
    return out


def fire_what_if(inp: dict, best: str) -> list[dict]:
    """1-day forecast if every fire count (and FRP) were scaled by a multiplier."""
    m = load_models()[(1, best, "with_fires")]
    rows = []
    for k in FIRE_MULTIPLIERS:
        alt = {**inp, "fires": [_f(v) * k for v in inp["fires"]]}
        if inp.get("fire_frp_today") not in (None, ""):
            alt["fire_frp_today"] = _f(inp["fire_frp_today"]) * k
        X = features_from_inputs(alt, 1)
        rows.append({"multiplier": k, "fires_today": float(alt["fires"][-1]),
                     "prediction": float(m.predict(X[ALL_FEATURES])[0])})
    return rows


def inputs_for_date(date: str) -> dict:
    """Raw inputs for a real historical day t, plus the observed PM2.5 on t+1..t+3."""
    hist = history()
    t = pd.Timestamp(date).normalize()
    win = hist.loc[t - pd.Timedelta(days=WINDOW - 1): t]
    if len(win) != WINDOW or t not in hist.index:
        raise ValueError(f"no complete 7-day history for {date} "
                         f"(data covers {hist.index[WINDOW - 1].date()} to {hist.index[-1].date()})")
    clean = lambda v: None if pd.isna(v) else round(float(v), 2)
    today = win.loc[t]
    actual = {str(h): clean(hist.pm25.get(t + pd.Timedelta(days=h))) for h in (1, 2, 3)}
    return {
        "inputs": {
            "date": str(t.date()),
            "pm25": [clean(v) for v in win.pm25],
            "fires": [int(v) for v in win.fire_count],
            "fire_frp_today": clean(today.fire_frp_sum),
            "pm10": clean(today.pm10), "no2": clean(today.no2), "co": clean(today.co),
            "weather": {k: clean(today[k]) for k in WEATHER_KEYS},
        },
        "actual": actual,
        "split": ("train" if t.year <= 2017 else "validation" if t.year == 2018
                  else "test" if t.year == 2019 else "extra (2020)"),
    }
