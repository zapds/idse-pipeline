"""Robustness analysis: rolling-origin evaluation over three burning seasons, at 1-3 day horizons.

The main analysis (train.py / evaluate.py) tests on one season (2019, 47 days). That is too few
days to tell a ~5% effect from noise, so here every model is re-tuned and re-tested three times:

    fold 2017: train on targets in 2015-2016          -> test on 2017
    fold 2018: train on targets in 2015-2017          -> test on 2018
    fold 2019: train on targets in 2015-2018          -> test on 2019

Within each fold, hyperparameters are tuned by TimeSeriesSplit CV on that fold's training years
only, so no fold ever sees its own test year. Burning-season test days from the three folds are
pooled (about 140 days) for one fire-effect estimate with a bootstrap CI.

The same is repeated for horizons of 1, 2 and 3 days ahead. Features always stop at day t;
only the target moves further away.

Note: the 2019 / 1-day fold differs slightly from the main analysis, because there the model was
tuned on 2015-2017 and then refitted on 2015-2018, while here tuning also uses 2018.
"""
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

from config import AQI_CLEAN, FIRES_DAILY, METRICS, MODELS_DIR, WEATHER_CLEAN
from evaluate import paired_bootstrap
from features import TARGET, build_features
from models import MODELS
from train import VARIANTS

HORIZONS = (1, 2, 3)
TEST_YEARS = (2017, 2018, 2019)
OUT_PRED = METRICS / "robustness_predictions.parquet"
OUT_SUMMARY = METRICS / "robustness_summary.json"


def model_table(horizon: int) -> pd.DataFrame:
    df = build_features(pd.read_parquet(AQI_CLEAN), pd.read_parquet(FIRES_DAILY),
                        pd.read_parquet(WEATHER_CLEAN), horizon)
    return df.dropna(subset=[TARGET, "pm25_t"])


def run_predictions() -> pd.DataFrame:
    """Tune, fit and predict every (horizon, fold, model, variant). Returns long-format rows."""
    cv = TimeSeriesSplit(n_splits=5)
    frames = []
    for h in HORIZONS:
        df = model_table(h)
        for year in TEST_YEARS:
            ty = df.target_date.dt.year
            tr, te = df[ty < year], df[ty == year]
            out = te[[TARGET, "target_date", "is_fire_season", "pm25_t", "nw_wind"]].copy()
            out["horizon"], out["fold"] = h, year
            out["Persistence"] = te.pm25_t
            t0 = time.time()
            for name, (factory, grid) in MODELS.items():
                for variant, feats in VARIANTS.items():
                    gs = GridSearchCV(factory(), grid, cv=cv, scoring="neg_mean_absolute_error",
                                      n_jobs=-1).fit(tr[feats], tr[TARGET])
                    out[f"{name}_{variant}"] = gs.predict(te[feats])
                    if year == TEST_YEARS[-1] and h > 1:
                        # Same training window as the main 1-day models (2015-2018); used by the website.
                        joblib.dump(gs.best_estimator_, MODELS_DIR / f"h{h}_{name}_{variant}.joblib")
            print(f"  horizon {h}  fold {year}: train {len(tr)}  test {len(te)}  ({time.time()-t0:.0f}s)")
            frames.append(out)
    pred = pd.concat(frames)
    pred.to_parquet(OUT_PRED)
    return pred


def summarise(pred: pd.DataFrame) -> dict:
    """Pooled and per-fold fire effect for every horizon and model family."""
    mae = lambda d, c: float(np.mean(np.abs(d[TARGET] - d[c])))
    summary = {}
    for h, ph in pred.groupby("horizon"):
        fs = ph[ph.is_fire_season == 1].sort_values("target_date")
        res = {"n_fire_season_days": len(fs), "n_days": len(ph),
               "persistence": {"overall": mae(ph, "Persistence"), "fire_season": mae(fs, "Persistence")}}
        for m in MODELS:
            a, b = f"{m}_no_fires", f"{m}_with_fires"
            args = (fs[TARGET].to_numpy(), fs[a].to_numpy(), fs[b].to_numpy())
            per_fold = {int(y): {"mae_without": mae(d, a), "mae_with": mae(d, b),
                                 "rel_reduction": 1 - mae(d, b) / mae(d, a), "n": len(d)}
                        for y, d in fs.groupby("fold")}
            res[m] = {
                "overall": {"mae_without": mae(ph, a), "mae_with": mae(ph, b)},
                "fire_season_pooled": {"iid": paired_bootstrap(*args),
                                       "block7": paired_bootstrap(*args, block=7)},
                "per_fold": per_fold,
                "folds_improved": sum(v["rel_reduction"] > 0 for v in per_fold.values()),
            }
        summary[f"h{h}"] = res
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2))
    return summary


def tidy(summary: dict) -> pd.DataFrame:
    """One row per horizon x model: the headline numbers."""
    rows = []
    for h, res in summary.items():
        for m in MODELS:
            r = res[m]["fire_season_pooled"]
            rows.append({
                "horizon": int(h[1:]), "model": m,
                "season MAE no fires": r["iid"]["mae_without"], "season MAE with fires": r["iid"]["mae_with"],
                "gain %": 100 * r["iid"]["rel_reduction"],
                "iid CI low": r["iid"]["ci95"][0], "iid CI high": r["iid"]["ci95"][1],
                "block7 CI low": r["block7"]["ci95"][0], "block7 CI high": r["block7"]["ci95"][1],
                "folds improved (of 3)": res[m]["folds_improved"],
                **{f"gain % {y}": 100 * v["rel_reduction"] for y, v in res[m]["per_fold"].items()},
                "persistence season MAE": res["persistence"]["fire_season"],
            })
    t = pd.DataFrame(rows)
    t.to_csv(METRICS / "robustness_table.csv", index=False)
    return t


def main() -> None:
    pred = run_predictions()
    t = tidy(summarise(pred))
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(t.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
