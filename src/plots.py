"""All figures -> reports/figures/. EDA figures (Phase 3) and results figures (Phase 6).

Colour roles are fixed across every figure: blue = no-fires model, orange = fire data /
with-fires model, aqua = persistence baseline, near-black = observed PM2.5.
Two measures on different scales are drawn as stacked panels, never on a twin y-axis.
"""
import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from config import (AQI_CLEAN, DIWALI, FIGURES, FIRE_BBOX, FIRES_DAILY, METRICS,  # noqa: E402
                    MODEL_TABLE, MODELS_DIR, RAW_FIRMS, SEVERE, WEATHER_CLEAN)
from features import ALL_FEATURES, FIRES, TARGET  # noqa: E402

BLUE, ORANGE, AQUA, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#8a8984", "#e6e5e1"
SEASON_SHADE = "#f6d9cb"

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.labelcolor": "#52514e",
    "xtick.color": "#52514e", "ytick.color": "#52514e", "font.size": 10,
    "lines.linewidth": 1.6, "legend.frameon": False,
})


def _save(fig, name):
    fig.savefig(FIGURES / f"{name}.png")
    plt.close(fig)


def _shade_seasons(ax, years=range(2015, 2021)):
    for y in years:
        ax.axvspan(pd.Timestamp(f"{y}-10-15"), pd.Timestamp(f"{y}-11-30"), color=SEASON_SHADE, lw=0)


def _load():
    return (pd.read_parquet(AQI_CLEAN), pd.read_parquet(FIRES_DAILY),
            pd.read_parquet(WEATHER_CLEAN), pd.read_parquet(MODEL_TABLE))


# =============================================================== EDA (Phase 3)
def fig_missingness(aqi):
    cols = [c for c in aqi.columns if c != "pm25_category"]
    m = aqi[cols].isna().resample("MS").mean().T
    fig, ax = plt.subplots(figsize=(11, 3.2))
    sns.heatmap(m, cmap="Blues", vmin=0, vmax=1, ax=ax, cbar_kws={"label": "fraction missing"})
    ax.set_xticks(range(0, m.shape[1], 6))
    ax.set_xticklabels([d.strftime("%Y-%m") for d in m.columns[::6]], rotation=45, ha="right")
    ax.set(title="Missing values per month, Delhi CPCB data", xlabel="", ylabel="")
    _save(fig, "00_missingness")


def fig_timeseries(aqi):
    fig, ax = plt.subplots(figsize=(11, 3.6))
    _shade_seasons(ax)
    ax.plot(aqi.index, aqi.pm25, color=INK, lw=0.9)
    ax.axhline(SEVERE, color=MUTED, ls="--", lw=1)
    ax.text(aqi.index[5], SEVERE + 8, "Severe (250)", color="#52514e", fontsize=9)
    ax.set(title="1. Delhi daily PM2.5, 2015 - mid-2020 (shaded: 15 Oct - 30 Nov)",
           ylabel="PM2.5 (µg/m³)")
    _save(fig, "01_pm25_timeseries")


def fig_monthly_box(aqi):
    d = aqi.assign(month=aqi.index.month)
    fig, ax = plt.subplots(figsize=(9, 3.8))
    sns.boxplot(data=d, x="month", y="pm25", color="#c9dcf3", fliersize=2, linewidth=1, ax=ax)
    ax.set(title="2. PM2.5 by month: a strong winter peak", xlabel="month", ylabel="PM2.5 (µg/m³)")
    _save(fig, "02_monthly_box")


def fig_fires_vs_pm(aqi, fires):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
    for ax in (a1, a2):
        _shade_seasons(ax)
    a1.plot(aqi.index, aqi.pm25, color=INK, lw=0.9)
    a1.set(ylabel="PM2.5 (µg/m³)", title="3. Delhi PM2.5 (top) and Punjab+Haryana fire detections (bottom)")
    a2.bar(fires.index, fires.fire_count, color=ORANGE, width=1.0)
    a2.set(ylabel="fires / day (MODIS)")
    _save(fig, "03_fires_vs_pm25")


def fig_fire_map():
    """Fire detections Oct-Nov in a wider window, with the analysis bounding box drawn."""
    pts = []
    for y in range(2015, 2020):
        f = pd.read_csv(RAW_FIRMS / f"modis_{y}_India.csv",
                        usecols=["latitude", "longitude", "acq_date", "type", "confidence"])
        f = f[(f.type == 0) & (f.confidence >= 30) & f.acq_date.str[5:7].isin(["10", "11"])]
        pts.append(f[f.latitude.between(24, 34) & f.longitude.between(70, 82)])
    pts = pd.concat(pts)
    b = FIRE_BBOX
    inside = pts.latitude.between(b["lat_min"], b["lat_max"]) & pts.longitude.between(b["lon_min"], b["lon_max"])
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.scatter(pts.longitude[~inside], pts.latitude[~inside], s=1, color=MUTED, alpha=0.3, lw=0)
    ax.scatter(pts.longitude[inside], pts.latitude[inside], s=1, color=ORANGE, alpha=0.3, lw=0)
    ax.add_patch(plt.Rectangle((b["lon_min"], b["lat_min"]), b["lon_max"] - b["lon_min"],
                               b["lat_max"] - b["lat_min"], fill=False, ec=INK, lw=1.5))
    ax.plot(77.21, 28.61, marker="*", ms=14, color=BLUE, mec="white")
    ax.annotate("Delhi", (77.21, 28.61), xytext=(77.6, 28.2), fontsize=10)
    ax.set(title=f"4. Oct-Nov fire detections 2015-19\n(box = Punjab+Haryana region; {inside.mean():.0%} of points inside)",
           xlabel="longitude", ylabel="latitude", aspect="equal")
    _save(fig, "04_fire_map")


def fig_lag_corr(fires, aqi):
    d = pd.DataFrame({"pm": aqi.pm25, "fire": np.log1p(fires.fire_count.reindex(aqi.index))})
    oct_nov = d.index.month.isin([10, 11])
    rows = []
    for lag in range(6):
        sh = d.fire.shift(lag)
        rows.append({"lag": lag, "all days": d.pm.corr(sh),
                     "Oct-Nov only": d.pm[oct_nov].corr(sh[oct_nov])})
    r = pd.DataFrame(rows).set_index("lag")
    r.to_csv(METRICS / "eda_lag_correlation.csv")
    fig, ax = plt.subplots(figsize=(7, 3.6))
    x = np.arange(6)
    ax.bar(x - 0.2, r["all days"], 0.38, color=BLUE, label="all days")
    ax.bar(x + 0.2, r["Oct-Nov only"], 0.38, color=ORANGE, label="Oct-Nov only")
    ax.set_xticks(x, [f"t-{k}" for k in x])
    ax.set(title="5. Correlation of log(fires on day t-k) with PM2.5 on day t", xlabel="fire lag",
           ylabel="Pearson r")
    ax.legend()
    _save(fig, "05_lag_correlation")
    return r


def fig_weather(aqi, weather):
    d = weather.join(aqi.pm25).dropna(subset=["pm25"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, col, lab in zip(axes, ["wind_speed", "blh_mean"],
                            ["max wind speed (km/h)", "mean boundary-layer height (m)"]):
        ax.scatter(d[col], d.pm25, s=6, color=BLUE, alpha=0.35, lw=0)
        q = d.groupby(pd.qcut(d[col], 10), observed=True).agg(x=(col, "median"), y=("pm25", "median"))
        ax.plot(q.x, q.y, color=INK, marker="o", ms=4, label="decile median")
        ax.set(xlabel=lab, title=f"r = {d[col].corr(d.pm25):.2f} (Spearman {d[col].corr(d.pm25, method='spearman'):.2f})")
        ax.legend()
    axes[0].set_ylabel("PM2.5 (µg/m³)")
    fig.suptitle("6. Weather confounders: calm air and a shallow mixing layer trap pollution",
                 fontweight="bold", x=0.02, y=1.03, ha="left")
    _save(fig, "06_weather_vs_pm25")


def fig_wind_direction(aqi, weather):
    d = weather.join(aqi.pm25).dropna(subset=["pm25"])
    d = d[d.index.month.isin([10, 11, 12, 1])]  # compare within the cool season only
    sectors = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    d["sector"] = pd.Categorical([sectors[int(((w + 22.5) % 360) // 45)] for w in d.wind_dir], sectors)
    d["group"] = np.where(d.pm25 > SEVERE, "Severe days (>250)", "other days")
    share = d.groupby("group").sector.value_counts(normalize=True).unstack(0).reindex(sectors)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    x = np.arange(len(sectors))
    ax.bar(x - 0.2, share["other days"], 0.38, color=BLUE, label="other days")
    ax.bar(x + 0.2, share["Severe days (>250)"], 0.38, color=ORANGE, label="Severe days (>250)")
    ax.set_xticks(x, sectors)
    ax.set(title="7. Dominant wind direction (from), Oct-Jan days", ylabel="share of days",
           xlabel="wind from")
    ax.legend()
    _save(fig, "07_wind_direction")
    return share


def fig_corr_heatmap(table):
    cols = [TARGET, "pm25_t", "pm25_roll7_mean", "pm10_t", "no2_t", "co_t", "wind_speed", "nw_wind",
            "temp", "humidity", "precip", "blh_mean", "blh_min", "log_fire_count_t",
            "log_fire_roll3_sum", "fire_roll3_x_nw"]
    c = table[cols].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(9, 7.5))
    sns.heatmap(c, cmap="RdBu_r", center=0, vmin=-1, vmax=1, annot=True, fmt=".1f",
                annot_kws={"size": 7}, ax=ax, cbar_kws={"label": "Spearman ρ"})
    ax.set_title("8. Feature correlations (Spearman)")
    _save(fig, "08_corr_heatmap")


def fig_diwali(aqi):
    fig, axes = plt.subplots(1, 5, figsize=(14, 3.4), sharey=True)
    for ax, y in zip(axes, range(2015, 2020)):
        w = aqi.loc[f"{y}-10-01":f"{y}-11-30", "pm25"]
        ax.axvspan(pd.Timestamp(f"{y}-10-15"), pd.Timestamp(f"{y}-11-30"), color=SEASON_SHADE, lw=0)
        ax.plot(w.index, w, color=INK, lw=1.2)
        dw = DIWALI[DIWALI.year == y][0]
        ax.axvline(dw, color=BLUE, ls="--", lw=1.2)
        ax.text(dw, ax.get_ylim()[1] * 0.02 + 20, " Diwali", color=BLUE, fontsize=8)
        ax.set_title(str(y))
        ax.tick_params(axis="x", rotation=45)
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%d %b"))
    axes[0].set_ylabel("PM2.5 (µg/m³)")
    fig.suptitle("9. Oct-Nov PM2.5 with Diwali marked: fireworks spike sits inside the burning season",
                 fontweight="bold", x=0.02, y=1.03, ha="left")
    _save(fig, "09_diwali")


def eda_figures():
    aqi, fires, weather, table = _load()
    fig_missingness(aqi)
    fig_timeseries(aqi)
    fig_monthly_box(aqi)
    fig_fires_vs_pm(aqi, fires)
    fig_fire_map()
    fig_lag_corr(fires, aqi)
    fig_weather(aqi, weather)
    fig_wind_direction(aqi, weather)
    fig_corr_heatmap(table)
    fig_diwali(aqi)


# =============================================================== results (Phase 6)
def results_figures():
    sel = json.loads((METRICS / "model_selection.json").read_text())
    summ = json.loads((METRICS / "summary.json").read_text())
    best = sel["selected_model"]
    pred = pd.read_parquet(METRICS / "test_predictions.parquet")
    tm = pd.read_csv(METRICS / "test_metrics.csv")

    # R1: model comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)
    for ax, scope, title in zip(axes, ["overall", "fire_season"], ["2019 test year: all days", "2019 burning season (15 Oct - 30 Nov)"]):
        t = tm[tm.scope == scope]
        base = t[t.variant == "baseline"]
        learned = t[t.variant != "baseline"].pivot(index="model", columns="variant", values="MAE")
        names = list(base.model) + list(learned.index)
        x = np.arange(len(names))
        ax.bar(x[:len(base)], base.MAE, 0.6, color=MUTED, label="baseline")
        xl = x[len(base):]
        ax.bar(xl - 0.18, learned.no_fires, 0.34, color=BLUE, label="without fires")
        ax.bar(xl + 0.18, learned.with_fires, 0.34, color=ORANGE, label="with fires")
        for xi, v in zip(x, list(base.MAE)):
            ax.text(xi, v, f"{v:.0f}", ha="center", va="bottom", fontsize=8)
        for xi, a, b in zip(xl, learned.no_fires, learned.with_fires):
            ax.text(xi - 0.18, a, f"{a:.0f}", ha="center", va="bottom", fontsize=8)
            ax.text(xi + 0.18, b, f"{b:.0f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x, [n.replace("SeasonalMean", "Seasonal\nmean") for n in names])
        ax.set(title=title, ylabel="MAE (µg/m³)")
    axes[0].legend(loc="upper right")
    fig.suptitle("R1. Next-day PM2.5 forecast error by model", fontweight="bold", x=0.02, y=1.03, ha="left")
    _save(fig, "R1_model_comparison")

    # R2: test-year forecast during the burning season
    w = pred.set_index("target_date").loc["2019-10-01":"2019-12-15"]
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.axvspan(pd.Timestamp("2019-10-15"), pd.Timestamp("2019-11-30"), color=SEASON_SHADE, lw=0)
    ax.plot(w.index, w[TARGET], color=INK, lw=2, label="observed")
    ax.plot(w.index, w.Persistence, color=AQUA, lw=1.2, label="persistence")
    ax.plot(w.index, w[f"{best}_no_fires"], color=BLUE, lw=1.4, label=f"{best} without fires")
    ax.plot(w.index, w[f"{best}_with_fires"], color=ORANGE, lw=1.4, label=f"{best} with fires")
    ax.axhline(SEVERE, color=MUTED, ls="--", lw=1)
    ax.set(title="R2. 2019 test season: observed vs next-day forecasts", ylabel="PM2.5 (µg/m³)")
    ax.legend(ncol=4, loc="upper right")
    _save(fig, "R2_test_season_forecast")

    # R3: bootstrap distribution of the fire effect
    r = summ["G2_fire_value"]["selected_model"]
    diffs = np.load(METRICS / f"bootstrap_diffs_{best}.npy")
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.hist(diffs, bins=50, color=ORANGE, alpha=0.85)
    ax.axvline(0, color=INK, lw=1.2)
    for v in r["ci95"]:
        ax.axvline(v, color=MUTED, ls="--")
    ax.set(title=f"R3. Bootstrap: MAE(without) - MAE(with fires), {best}\n"
                 f"mean {r['diff']:.1f} µg/m³ ({r['rel_reduction']:+.0%}), 95% CI [{r['ci95'][0]:.1f}, {r['ci95'][1]:.1f}]",
           xlabel="MAE improvement from fire features (µg/m³)", ylabel="resamples")
    _save(fig, "R3_bootstrap")

    # R4: residual analysis
    pred["err"] = pred[f"{best}_with_fires"] - pred[TARGET]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    ax = axes[0]
    ax.scatter(pred[TARGET], pred[f"{best}_with_fires"], s=8, color=ORANGE, alpha=0.6, lw=0)
    lim = [0, max(pred[TARGET].max(), pred[f"{best}_with_fires"].max()) * 1.05]
    ax.plot(lim, lim, color=INK, lw=1)
    ax.set(xlim=lim, ylim=lim, xlabel="observed PM2.5", ylabel="predicted PM2.5", title="predicted vs observed")
    bm = pd.read_csv(METRICS / "error_by_month.csv")
    axes[1].bar(bm.month, bm.bias, color=[ORANGE if v > 0 else BLUE for v in bm.bias])
    axes[1].axhline(0, color=INK, lw=1)
    axes[1].set(xlabel="month", ylabel="mean error (pred - obs)", title="bias by month", xticks=range(1, 13))
    bl = pd.read_csv(METRICS / "error_by_level.csv")
    axes[2].bar(bl.pm25_level, bl.bias, color=[ORANGE if v > 0 else BLUE for v in bl.bias])
    axes[2].axhline(0, color=INK, lw=1)
    axes[2].tick_params(axis="x", rotation=30)
    axes[2].set(ylabel="mean error (pred - obs)", title="bias by observed level")
    fig.suptitle(f"R4. Error analysis, {best} with fires, 2019", fontweight="bold", x=0.02, y=1.03, ha="left")
    _save(fig, "R4_error_analysis")

    # R5: permutation importance
    pi = pd.read_csv(METRICS / "permutation_importance.csv").head(15)[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(pi.feature, pi.mae_increase, xerr=pi["std"], color=[ORANGE if f in FIRES else BLUE for f in pi.feature])
    ax.set(title=f"R5. Permutation importance on 2019 ({summ['tree_model_for_importance']}, with fires)\norange = fire feature",
           xlabel="MAE increase when shuffled (µg/m³)")
    _save(fig, "R5_permutation_importance")

    # R6: partial dependence on recent fires, split by wind direction
    tree = summ["tree_model_for_importance"]
    model = joblib.load(MODELS_DIR / f"{tree}_with_fires.joblib")
    table = pd.read_parquet(MODEL_TABLE)
    X = table[table.is_fire_season == 1][ALL_FEATURES]
    grid = np.linspace(0, np.percentile(table.fire_roll3_sum, 99.5), 25)
    fig, ax = plt.subplots(figsize=(7, 4))
    for nw, col, lab in [(1, ORANGE, "NW wind (from Punjab)"), (0, BLUE, "other wind")]:
        Xs = X[X.nw_wind == nw].copy()
        ys = []
        for g in grid:
            Xs["fire_roll3_sum"] = g
            Xs["log_fire_roll3_sum"] = np.log1p(g)
            Xs["fire_roll3_x_nw"] = np.log1p(g) * nw
            ys.append(model.predict(Xs).mean())
        ax.plot(grid, ys, color=col, marker="o", ms=3, label=f"{lab} (n={len(Xs)})")
    ax.set(title=f"R6. Partial dependence: 3-day fire count -> predicted PM2.5\n({tree}, burning-season days 2015-19)",
           xlabel="fires in last 3 days (Punjab+Haryana)", ylabel="mean predicted PM2.5 (µg/m³)")
    ax.legend()
    _save(fig, "R6_partial_dependence")

    # R7: standardised ridge coefficients
    rc = pd.read_csv(METRICS / "ridge_coefficients.csv", index_col=0).iloc[:, 0].head(15)[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(rc.index, rc.values, color=[ORANGE if f in FIRES else BLUE for f in rc.index])
    ax.axvline(0, color=INK, lw=1)
    ax.set(title="R7. Ridge (with fires): standardised coefficients, top 15\n(effect on log PM2.5 per 1 SD; orange = fire feature)",
           xlabel="coefficient")
    _save(fig, "R7_ridge_coefficients")


def robustness_figures():
    """R8: fire gain per season and horizon. R9: forecast error vs horizon."""
    summ = json.loads((METRICS / "robustness_summary.json").read_text())
    models = ["Ridge", "RandomForest", "HistGB"]
    hs = sorted(int(h[1:]) for h in summ)
    fig, ax = plt.subplots(figsize=(10, 4.4))
    ax.axhline(0, color=INK, lw=1)
    offs = {"Ridge": -0.25, "RandomForest": 0, "HistGB": 0.25}
    for m in models:
        for h in hs:
            r = summ[f"h{h}"][m]
            x = h + offs[m]
            pooled = r["fire_season_pooled"]["iid"]
            lo, hi = (100 * v for v in pooled["rel_ci95"])
            ax.plot([x, x], [lo, hi], color=ORANGE, lw=2)
            ax.plot(x, 100 * pooled["rel_reduction"], "o", color=ORANGE, ms=8, mec="white", mew=1.5,
                    label="pooled 2017-19 (95% CI)" if (m, h) == ("Ridge", 1) else None)
            folds = r["per_fold"]
            ax.scatter([x + 0.07] * len(folds), [100 * v["rel_reduction"] for v in folds.values()],
                       s=22, facecolors="none", edgecolors=MUTED, lw=1,
                       label="single season" if (m, h) == ("Ridge", 1) else None)
            if h == hs[0]:
                ax.text(x, -21, m.replace("RandomForest", "RF"),
                        ha="center", fontsize=8, color="#52514e")
    ax.set_xticks(hs, [f"{h} day{'s' if h > 1 else ''} ahead" for h in hs])
    ax.set_ylim(-22, 12)
    ax.set(ylabel="burning-season MAE reduction from fires (%)",
           title="R8. Fire features across three test seasons and three horizons\n"
                 "positive = fires help; models left to right: Ridge, RF, HistGB")
    ax.legend(loc="upper left")
    _save(fig, "R8_robustness_fire_gain")

    pred = pd.read_parquet(METRICS / "robustness_predictions.parquet")
    fs = pred[pred.is_fire_season == 1]
    rows = []
    for h, d in fs.groupby("horizon"):
        e = lambda c: float(np.mean(np.abs(d[TARGET] - d[c])))
        rows.append({"h": h, "Persistence": e("Persistence"),
                     "RF without fires": e("RandomForest_no_fires"), "RF with fires": e("RandomForest_with_fires")})
    t = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7, 4))
    for c, col in [("Persistence", AQUA), ("RF without fires", BLUE), ("RF with fires", ORANGE)]:
        ax.plot(t.h, t[c], marker="o", color=col, label=c, lw=2 if c != "RF with fires" else 1.4,
                ls="-" if c != "RF with fires" else "--")
    gap = [f"{h} day{'s' if h > 1 else ''}\nRF {100 * (r - p) / p:+.0f}% vs persistence"
           for h, r, p in zip(t.h, t["RF without fires"], t.Persistence)]
    ax.set_xticks(t.h, gap)
    ax.set_xlim(0.8, 3.2)
    ax.set(title="R9. Burning-season forecast error vs horizon\n(pooled 2017-19 test seasons)",
           xlabel="forecast horizon", ylabel="MAE (µg/m³)")
    ax.legend()
    _save(fig, "R9_error_vs_horizon")


if __name__ == "__main__":
    import sys
    if "--results" in sys.argv:
        results_figures()
        robustness_figures()
    else:
        eda_figures()
