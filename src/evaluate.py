"""Phase 6: score every model ONCE on the 2019 test year, test the fire hypothesis (G2),
evaluate Severe-day alerts (G3), and run the error analysis and interpretation.

Outputs go to reports/metrics/.
"""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix, precision_score, recall_score

from config import (METRICS, MODELS_DIR, N_BOOTSTRAP, PM25_LABELS, SEED, SEVERE, VERY_POOR)
from clean_aqi import pm25_category
from features import ALL_FEATURES, FIRES, TARGET
from models import BASELINES, MODELS
from train import VARIANTS, load_splits, score_by_season


def predict_all(df: pd.DataFrame) -> pd.DataFrame:
    """One prediction column per model/variant, plus the actual target and context columns."""
    out = df[[TARGET, "target_date", "pm25_t", "is_fire_season", "is_diwali_window", "nw_wind",
              "precip", "wind_speed", "fire_count_t", "fire_roll3_sum"]].copy()
    for name in BASELINES:
        out[name] = joblib.load(MODELS_DIR / f"{name}.joblib").predict(df[ALL_FEATURES])
    for name in MODELS:
        for variant, feats in VARIANTS.items():
            out[f"{name}_{variant}"] = joblib.load(MODELS_DIR / f"{name}_{variant}.joblib").predict(df[feats])
    return out


def metrics_table(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    cols = list(BASELINES) + [f"{m}_{v}" for m in MODELS for v in VARIANTS]
    for c in cols:
        for scope, s in score_by_season(pred, pred[c]).items():
            model, _, variant = c.partition("_")
            rows.append({"model": model, "variant": variant or "baseline", "scope": scope, **s})
    return pd.DataFrame(rows)


def paired_bootstrap(y, p_without, p_with, block: int = 1, n: int = N_BOOTSTRAP,
                     save_to=None) -> dict:
    """Bootstrap CI of MAE(without) - MAE(with). Positive = fires help.

    block=1 is the ordinary paired bootstrap from the plan. block>1 resamples contiguous
    blocks of days (moving-block bootstrap), which respects day-to-day autocorrelation and
    gives a more honest, wider interval.
    """
    rng = np.random.default_rng(SEED)
    e0 = np.abs(np.asarray(y) - np.asarray(p_without))
    e1 = np.abs(np.asarray(y) - np.asarray(p_with))
    m = len(e0)
    n_blocks = int(np.ceil(m / block))
    diffs, rels = np.empty(n), np.empty(n)
    for i in range(n):
        starts = rng.integers(0, m - block + 1, size=n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:m]
        a, b = e0[idx].mean(), e1[idx].mean()
        diffs[i], rels[i] = a - b, (a - b) / a
    if save_to is not None:
        np.save(save_to, diffs)  # reused by plots.py so the histogram matches the reported CI
    return {"mae_without": float(e0.mean()), "mae_with": float(e1.mean()),
            "diff": float(e0.mean() - e1.mean()),
            "rel_reduction": float((e0.mean() - e1.mean()) / e0.mean()),
            "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
            "rel_ci95": [float(np.percentile(rels, 2.5)), float(np.percentile(rels, 97.5))],
            "p_fires_not_helping": float(np.mean(diffs <= 0)), "n_days": m, "block": block}


def alert_metrics(y, pred, threshold: float, forecast_threshold: float | None = None) -> dict:
    """Alert = forecast above `forecast_threshold` (default: same as the event threshold)."""
    ft = threshold if forecast_threshold is None else forecast_threshold
    a, p = np.asarray(y) > threshold, np.asarray(pred) > ft
    cm = confusion_matrix(a, p, labels=[False, True])
    return {"threshold": threshold, "forecast_threshold": ft, "n_actual": int(a.sum()), "n_predicted": int(p.sum()),
            "recall": float(recall_score(a, p, zero_division=0)),
            "precision": float(precision_score(a, p, zero_division=0)),
            "confusion_matrix [[TN, FP], [FN, TP]]": cm.tolist()}


def group_permutation(model, X, y, group, n_repeats=30) -> dict:
    """Increase in MAE when ALL columns in `group` are shuffled together (same row permutation)."""
    rng = np.random.default_rng(SEED)
    base = np.mean(np.abs(y - model.predict(X)))
    inc = []
    for _ in range(n_repeats):
        Xp = X.copy()
        Xp[group] = X[group].to_numpy()[rng.permutation(len(X))]
        inc.append(np.mean(np.abs(y - model.predict(Xp))) - base)
    return {"mae_increase_mean": float(np.mean(inc)), "mae_increase_std": float(np.std(inc))}


def main() -> None:
    s = load_splits()
    sel = json.loads((METRICS / "model_selection.json").read_text())
    best = sel["selected_model"]

    pred = predict_all(s["test"])
    pred.to_parquet(METRICS / "test_predictions.parquet")
    table = metrics_table(pred)
    table.to_csv(METRICS / "test_metrics.csv", index=False)
    print(table.pivot_table(index=["model", "variant"], columns="scope", values="MAE").round(2))

    extra = predict_all(s["extra_2020"])
    extra.to_parquet(METRICS / "extra2020_predictions.parquet")
    metrics_table(extra).query("scope == 'overall'").to_csv(METRICS / "extra2020_metrics.csv", index=False)

    y = pred[TARGET].to_numpy()
    fs = pred.is_fire_season.to_numpy() == 1
    mae = lambda c, mask=slice(None): float(np.mean(np.abs(y[mask] - pred[c].to_numpy()[mask])))
    bw, bn = f"{best}_with_fires", f"{best}_no_fires"

    # ---- G1: beat persistence
    g1 = {"best_with_fires_overall": mae(bw), "persistence_overall": mae("Persistence"),
          "best_with_fires_fire_season": mae(bw, fs), "persistence_fire_season": mae("Persistence", fs)}
    g1["passed"] = bool(g1["best_with_fires_overall"] < g1["persistence_overall"]
                        and g1["best_with_fires_fire_season"] < g1["persistence_fire_season"])

    # ---- G2: value of fire data (fire-season test days), for every family
    g2 = {}
    for m in MODELS:
        args = (y[fs], pred[f"{m}_no_fires"].to_numpy()[fs], pred[f"{m}_with_fires"].to_numpy()[fs])
        g2[m] = {"iid": paired_bootstrap(*args, save_to=METRICS / f"bootstrap_diffs_{m}.npy"),
                 "block7": paired_bootstrap(*args, block=7)}
    r = g2[best]["iid"]
    g2_pass = bool(r["rel_reduction"] >= 0.10 and r["ci95"][0] > 0)

    # ---- G3: alerts
    g3 = {f"{v}": {"severe": alert_metrics(y, pred[f"{best}_{v}"], SEVERE),
                   "very_poor_or_worse": alert_metrics(y, pred[f"{best}_{v}"], VERY_POOR)}
          for v in VARIANTS}
    # Secondary (added after the first test run, threshold chosen on 2018 validation only).
    for v, t in sel["severe_alert_threshold_from_val"].items():
        g3[v]["severe_val_calibrated"] = alert_metrics(y, pred[f"{best}_{v}"], SEVERE, t)
    g3["Persistence"] = {"severe": alert_metrics(y, pred.Persistence, SEVERE),
                         "very_poor_or_worse": alert_metrics(y, pred.Persistence, VERY_POOR)}
    g3_pass = g3["with_fires"]["severe"]["recall"] >= 0.7
    cat = confusion_matrix(pm25_category(pred[TARGET]).astype(str), pm25_category(pred[bw]).astype(str),
                           labels=PM25_LABELS)
    pd.DataFrame(cat, index=[f"actual_{l}" for l in PM25_LABELS],
                 columns=[f"pred_{l}" for l in PM25_LABELS]).to_csv(METRICS / "category_confusion.csv")

    # ---- error analysis on the selected with-fires model
    pred["error"] = pred[bw] - pred[TARGET]
    pred["abs_error"] = pred.error.abs()
    pred["month"] = pred.target_date.dt.month
    pred.groupby("month").agg(MAE=("abs_error", "mean"), bias=("error", "mean"),
                              mean_pm25=(TARGET, "mean")).round(2).to_csv(METRICS / "error_by_month.csv")
    pred["pm25_level"] = pm25_category(pred[TARGET])
    pred.groupby("pm25_level", observed=True).agg(
        n=("error", "size"), MAE=("abs_error", "mean"), bias=("error", "mean")).round(2) \
        .to_csv(METRICS / "error_by_level.csv")
    wind = pred.groupby("nw_wind").agg(n=("error", "size"), MAE=("abs_error", "mean"),
                                       bias=("error", "mean")).round(2)
    wind.index = wind.index.map({0: "other wind", 1: "NW wind"})
    wind.to_csv(METRICS / "error_by_wind.csv")
    worst = pred.nlargest(10, "abs_error")[["target_date", TARGET, bw, "error", "pm25_t",
                                             "is_diwali_window", "precip", "wind_speed",
                                             "nw_wind", "fire_count_t"]].round(1)
    worst.to_csv(METRICS / "worst_days.csv", index=False)

    # ---- interpretation
    ridge = joblib.load(MODELS_DIR / "Ridge_with_fires.joblib")
    coefs = pd.Series(ridge.regressor_.named_steps["model"].coef_, index=ALL_FEATURES)
    coefs.sort_values(key=np.abs, ascending=False).rename("std_coef_log_pm25") \
        .to_csv(METRICS / "ridge_coefficients.csv")

    tree = best if best != "Ridge" else "HistGB"
    tm = joblib.load(MODELS_DIR / f"{tree}_with_fires.joblib")
    Xt = s["test"][ALL_FEATURES]
    pi = permutation_importance(tm, Xt, y, scoring="neg_mean_absolute_error",
                                n_repeats=20, random_state=SEED, n_jobs=-1)
    pd.DataFrame({"feature": ALL_FEATURES, "mae_increase": pi.importances_mean,
                  "std": pi.importances_std}).sort_values("mae_increase", ascending=False) \
        .to_csv(METRICS / "permutation_importance.csv", index=False)
    Xfs = Xt[fs]
    groups = {"fire_group_all_test": group_permutation(tm, Xt, y, FIRES),
              "fire_group_fire_season": group_permutation(tm, Xfs, y[fs], FIRES)}

    summary = {
        "selected_model": best, "tree_model_for_importance": tree,
        "G1_beat_persistence": g1,
        "G2_fire_value": {"selected_model": r, "passed": g2_pass,
                          "criterion": ">=10% lower fire-season MAE and iid 95% CI excluding 0",
                          "all_models": g2},
        "G3_alerts": {"passed": bool(g3_pass), "criterion": "recall >= 0.7 for Severe (>250)",
                      "results": g3},
        "fire_group_permutation": groups,
    }
    (METRICS / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"  G1 passed={g1['passed']}  G2 passed={g2_pass} "
          f"(rel {r['rel_reduction']:+.1%}, CI {r['ci95'][0]:.2f}..{r['ci95'][1]:.2f})  "
          f"G3 passed={g3_pass} (recall {g3['with_fires']['severe']['recall']:.2f}; "
          f"val-calibrated: {g3['with_fires']['severe_val_calibrated']})")


if __name__ == "__main__":
    main()
