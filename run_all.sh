#!/usr/bin/env bash
# One command: download -> clean -> features -> test -> train -> evaluate -> figures.
# Prerequisite: data/raw/city_day.csv from Kaggle (download.py prints the command if missing).
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-.venv/bin/python}
export PYTHONHASHSEED=42

echo "[1/7] download";  (cd src && ../$PY download.py)
echo "[2/7] clean";     (cd src && ../$PY clean_aqi.py && ../$PY clean_fires.py && ../$PY clean_weather.py)
echo "[3/7] features";  (cd src && ../$PY features.py)
echo "[4/7] leakage tests"; $PY -m pytest -q tests
echo "[5/7] train";     (cd src && ../$PY train.py)
echo "[6/7] evaluate";  (cd src && ../$PY evaluate.py && ../$PY robustness.py)
echo "[7/7] figures";   (cd src && ../$PY plots.py && ../$PY plots.py --results)
if [[ "${SKIP_NOTEBOOKS:-0}" != "1" ]]; then
  echo "[+] executing notebooks"
  $PY -m jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb notebooks/02_results.ipynb
fi
echo "done: see reports/figures and reports/metrics"
