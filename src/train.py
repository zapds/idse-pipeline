"""Phase 5: tune on 2015-2017 with TimeSeriesSplit CV, select on 2018, refit on 2015-2018.

Every learned model is trained twice: without and with the fire features (the ablation).
The model family is selected on the validation year using the MEAN validation MAE of its two
variants, so the selection does not favour either side of the fire comparison.
"""
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

from config import METRICS, MODEL_TABLE, MODELS_DIR, SEVERE, TEST_END, TRAIN_END, VAL_END
from features import ALL_FEATURES, BASE_FEATURES, TARGET
from models import BASELINES, MODELS

VARIANTS = {"no_fires": BASE_FEATURES, "with_fires": ALL_FEATURES}


def load_splits() -> dict[str, pd.DataFrame]:
    """Split by the date being predicted (t+1), never randomly."""
    df = pd.read_parquet(MODEL_TABLE)
    td = df.target_date
    return {
        "train": df[td <= TRAIN_END],
        "val": df[(td > TRAIN_END) & (td <= VAL_END)],
        "trainval": df[td <= VAL_END],
        "test": df[(td > VAL_END) & (td <= TEST_END)],
        "extra_2020": df[td > TEST_END],
    }


def score(y, pred) -> dict:
    """MAE, RMSE and R2 in ug/m3."""
    y, pred = np.asarray(y), np.asarray(pred)
    return {"MAE": mean_absolute_error(y, pred),
            "RMSE": float(np.sqrt(mean_squared_error(y, pred))),
            "R2": r2_score(y, pred), "n": int(len(y))}


def score_by_season(df: pd.DataFrame, pred) -> dict:
    """Scores overall and on the 15 Oct - 30 Nov burning-season days."""
    fs = df.is_fire_season.to_numpy() == 1
    pred = np.asarray(pred)
    out = {"overall": score(df[TARGET], pred)}
    if fs.any():  # the 2020 stress-test period has no burning season
        out["fire_season"] = score(df[TARGET][fs], pred[fs])
    return out


def main() -> None:
    s = load_splits()
    for k, v in s.items():
        print(f"  {k:10s} {len(v):5d} rows  {v.target_date.min().date()} -> {v.target_date.max().date()}")
    tr, va, tv = s["train"], s["val"], s["trainval"]
    val_results, cv_results = [], {}
    val_preds = va[[TARGET, "target_date", "is_fire_season"]].copy()

    for name, cls in BASELINES.items():
        m = cls().fit(tr[ALL_FEATURES], tr[TARGET])
        val_results.append({"model": name, "variant": "baseline",
                            **_flatten(score_by_season(va, m.predict(va[ALL_FEATURES])))})
        joblib.dump(cls().fit(tv[ALL_FEATURES], tv[TARGET]), MODELS_DIR / f"{name}.joblib")

    cv = TimeSeriesSplit(n_splits=5)
    for name, (factory, grid) in MODELS.items():
        for variant, feats in VARIANTS.items():
            t0 = time.time()
            gs = GridSearchCV(factory(), grid, cv=cv, scoring="neg_mean_absolute_error", n_jobs=-1)
            gs.fit(tr[feats], tr[TARGET])
            val_pred = gs.best_estimator_.predict(va[feats])
            val_preds[f"{name}_{variant}"] = val_pred
            val_results.append({"model": name, "variant": variant,
                                **_flatten(score_by_season(va, val_pred))})
            cv_results[f"{name}/{variant}"] = {
                "best_params": {k.split("__")[-1]: v for k, v in gs.best_params_.items()},
                "cv_mae": -gs.best_score_, "cv_mae_std": float(gs.cv_results_["std_test_score"][gs.best_index_]),
            }
            # Refit the tuned configuration on train + validation for the final test.
            final = factory().set_params(**gs.best_params_).fit(tv[feats], tv[TARGET])
            joblib.dump(final, MODELS_DIR / f"{name}_{variant}.joblib")
            print(f"  {name:12s} {variant:10s} cv MAE {-gs.best_score_:6.2f}  "
                  f"val MAE {val_results[-1]['overall_MAE']:6.2f}  "
                  f"val fire-season MAE {val_results[-1]['fire_season_MAE']:6.2f}  ({time.time()-t0:.0f}s)")

    val = pd.DataFrame(val_results)
    val.to_csv(METRICS / "validation_metrics.csv", index=False)
    learned = val[val.variant != "baseline"]
    family_mae = learned.groupby("model").overall_MAE.mean().sort_values()
    best = family_mae.index[0]
    val_preds.to_parquet(METRICS / "val_predictions.parquet")
    thresholds = {v: alert_threshold(val_preds[TARGET], val_preds[f"{best}_{v}"]) for v in VARIANTS}
    selection = {"selected_model": best, "rule": "lowest mean validation MAE over both variants",
                 "family_val_mae": family_mae.round(3).to_dict(),
                 "severe_alert_threshold_from_val": thresholds, "cv": cv_results,
                 "features": VARIANTS}
    (METRICS / "model_selection.json").write_text(json.dumps(selection, indent=2, default=str))
    print(f"  Selected model family: {best}")


def alert_threshold(y, pred, target_recall: float = 0.7) -> float:
    """Highest forecast threshold that still flags >= target_recall of the Severe (>250) days
    in the validation year. A regression model shrinks extremes towards the mean, so the
    best cut-off on its forecast is lower than 250. Chosen on validation only, never on test."""
    actual = np.asarray(y) > SEVERE
    pred = np.asarray(pred)
    for t in np.arange(SEVERE, 99, -5.0):
        if (pred[actual] > t).mean() >= target_recall:
            return float(t)
    return 100.0


def _flatten(d: dict) -> dict:
    return {f"{scope}_{k}": v for scope, m in d.items() for k, v in m.items()}


if __name__ == "__main__":
    main()
