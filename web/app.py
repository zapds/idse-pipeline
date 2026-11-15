"""Inference website: model statistics, reports, training details and live forecasts. No login.

Run locally:   .venv/bin/python web/app.py            (http://127.0.0.1:8000)
Production:    .venv/bin/gunicorn -w 2 -b 0.0.0.0:8000 --chdir web app:app
"""
import json
import sys
from pathlib import Path

import pandas as pd
from flask import Flask, abort, jsonify, request, send_from_directory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import inference  # noqa: E402
from config import FIGURES, INTERIM, METRICS, PM25_BINS, PM25_LABELS  # noqa: E402
from features import BASE_FEATURES, FIRES  # noqa: E402

app = Flask(__name__, static_folder="static", static_url_path="/static")
app.json.sort_keys = False

DOWNLOADS = {
    "report.pdf": ROOT / "reports", "report.md": ROOT / "reports",
    "01_eda.ipynb": ROOT / "notebooks", "02_results.ipynb": ROOT / "notebooks",
}
FIGURE_CAPTIONS = json.loads((Path(__file__).parent / "captions.json").read_text())


def _csv(name: str) -> list[dict]:
    return json.loads(pd.read_csv(METRICS / name).round(3).to_json(orient="records"))


def _json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def build_meta() -> dict:
    """Everything the static pages show, gathered once at start-up."""
    sel = _json(METRICS / "model_selection.json")
    summ = _json(METRICS / "summary.json")
    hist = inference.history()
    return {
        "selected_model": sel["selected_model"],
        "selection_rule": sel["rule"],
        "family_val_mae": sel["family_val_mae"],
        "cv": sel["cv"],
        "alert_threshold": sel["severe_alert_threshold_from_val"],
        "features": {"base": BASE_FEATURES, "fires": FIRES},
        "goals": {
            "G1": summ["G1_beat_persistence"],
            "G2": {k: v for k, v in summ["G2_fire_value"].items() if k != "all_models"},
            "G2_all": summ["G2_fire_value"]["all_models"],
            "G3": summ["G3_alerts"],
        },
        "fire_group_permutation": summ["fire_group_permutation"],
        "test_metrics": _csv("test_metrics.csv"),
        "validation_metrics": _csv("validation_metrics.csv"),
        "extra2020_metrics": _csv("extra2020_metrics.csv"),
        "robustness": _csv("robustness_table.csv"),
        "worst_days": _csv("worst_days.csv"),
        "error_by_level": _csv("error_by_level.csv"),
        "permutation_importance": _csv("permutation_importance.csv")[:12],
        "cleaning": {k: _json(INTERIM / f"{k}_cleaning_log.json") for k in ("aqi", "fires", "weather")},
        "data_range": {"first_forecast_day": str(hist.index[inference.WINDOW - 1].date()),
                       "last_day": str(hist.index[-1].date()), "days": len(hist)},
        "categories": [{"label": l, "max": None if b == float("inf") else b}
                       for l, b in zip(PM25_LABELS, PM25_BINS[1:])],
        "figures": [{"file": f.name, **FIGURE_CAPTIONS.get(f.stem, {"title": f.stem, "caption": ""})}
                    for f in sorted(FIGURES.glob("*.png"))],
    }


META = build_meta()


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/meta")
def meta():
    return jsonify(META)


@app.get("/api/example")
def example():
    try:
        return jsonify(inference.inputs_for_date(request.args.get("date", "2019-11-01")))
    except (ValueError, KeyError) as e:
        return jsonify(error=str(e)), 400


@app.post("/api/predict")
def predict():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify(error="expected a JSON object"), 400
    try:
        return jsonify(inference.predict(payload))
    except (ValueError, KeyError, TypeError) as e:
        return jsonify(error=str(e)), 400


@app.get("/figures/<path:name>")
def figure(name):
    return send_from_directory(FIGURES, name, max_age=3600)


@app.get("/download/<name>")
def download(name):
    if name not in DOWNLOADS:
        abort(404)
    return send_from_directory(DOWNLOADS[name], name, as_attachment=name != "report.pdf")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=False)
