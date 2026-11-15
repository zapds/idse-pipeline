# Do Punjab–Haryana farm fires help predict Delhi's smog?

Every October–November Delhi's air turns "Very Poor" or "Severe", and stubble burning in Punjab and Haryana
gets the blame. This project asks a narrower, testable question:

> **Does adding satellite-detected farm-fire activity improve *next-day* prediction of Delhi's PM2.5 beyond
> what today's pollution and weather already tell us?**

The test is an **ablation**: every model is trained twice, without and with fire features, and the two are
compared on a held-out year (2019) that is used only once.

## Answer

**Once Delhi's own PM2.5 and weather are known, satellite fire counts from Punjab and Haryana add no measurable skill
to 1–3-day PM2.5 forecasts.** This holds across three burning seasons (2017–2019), three horizons and three model types.
A single-season test suggested a 4.5–7% gain, and testing on more seasons showed it was a one-season fluctuation.
Meanwhile, the ML forecasts beat persistence by 7% (1 day) to 14% (3 days) in the burning season.

This does not mean fires don't pollute Delhi. It means their effect is already visible in the variables a forecaster has.

## Results

**Primary analysis: 2019 test year, fixed in advance**

| Goal | Criterion | Result |
|------|-----------|--------|
| **G1** Beat the naive forecast | Best model MAE < persistence ("tomorrow = today") | **Met.** Random Forest MAE 25.3 vs 28.0 overall; 50.6 vs 54.5 in the burning season |
| **G2** Value of fire data | ≥10% lower burning-season MAE, 95% CI excluding 0 | **Not met.** 4.5% (Random Forest, CI [−0.05, 5.05] µg/m³), 7.2% (HistGB), −2.4% (Ridge) |
| **G3** Severe-day alerts | Recall ≥ 0.7 for PM2.5 > 250 | **Not met at a fixed 250 threshold** (recall 0.41). **Met with a validation-calibrated threshold of 225** (recall 0.74, precision 0.66, vs persistence 0.65/0.63) |

**Robustness check: rolling-origin over the 2017, 2018 and 2019 seasons, 1–3 days ahead** (`src/robustness.py`)

| Horizon | Fire gain, pooled over 141 burning-season days (3 models) | All 95% CIs include 0? | RF vs persistence |
|---|---|:-:|---:|
| 1 day | −1.9% to +0.7% | yes | −7% |
| 2 days | −3.5% to +0.7% | yes | −8% |
| 3 days | +0.5% to +2.1% | yes | −14% |

Single-season fire gains swing from −19% to +8%. 2019 was the one season where fires helped every tree model.
The full write-up is in [`reports/report.md`](reports/report.md) ([PDF](reports/report.pdf)).

## Data (all public)

| Source | Content | Licence / access |
|--------|---------|------------------|
| CPCB via Kaggle [`rohanrao/air-quality-data-in-india`](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india), `city_day.csv` | Daily Delhi pollutants, 2015-01-01 → 2020-07-01 (2,009 days) | CC0; needs a Kaggle token |
| [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/) MODIS C6.1, India yearly files | 482k fire detections (120k in Punjab+Haryana) | Public, no key |
| [Open-Meteo](https://open-meteo.com/) historical archive (ERA5) | Delhi daily weather + hourly boundary-layer height | CC BY 4.0, no key |

## Setup and run

```bash
uv venv --python 3.12 .venv            # or: python3.12 -m venv .venv
uv pip install --python .venv -r requirements.txt   # or: .venv/bin/pip install -r requirements.txt

# AQI data (requires a Kaggle token in ~/.kaggle/):
kaggle datasets download -d rohanrao/air-quality-data-in-india -f city_day.csv -p data/raw/
unzip data/raw/city_day.csv.zip -d data/raw/ && rm data/raw/city_day.csv.zip

./run_all.sh            # download → clean → features → leakage tests → train → evaluate → robustness → figures → notebooks
.venv/bin/python reports/build_pdf.py   # optional: report.md → report.pdf (needs Chrome)
```

Runtime: about 3–4 minutes once data is downloaded, mostly the robustness grid (the FIRMS download is ~37 MB). `SKIP_NOTEBOOKS=1 ./run_all.sh`
skips re-executing the notebooks. Every random process uses `SEED = 42` (`src/config.py`), so reruns give identical numbers.

## Inference website

A no-login site for exploring the results and running live forecasts: findings, every figure with its insight,
training details and hyperparameters, downloads, and a live 1–3-day forecast from any inputs (or a real day loaded
from the dataset, shown next to what was actually observed).

```bash
.venv/bin/python web/app.py                                      # dev: http://127.0.0.1:8000
.venv/bin/gunicorn -w 2 -b 0.0.0.0:8000 --chdir web app:app      # production
```

It needs the outputs of `./run_all.sh` (`models/`, `reports/`, `data/processed/`). Forecast features are built by the
same `build_features` used in training (`src/inference.py`), and `tests/test_inference.py` checks that the two paths
give identical features. API: `GET /api/meta`, `GET /api/example?date=YYYY-MM-DD`, `POST /api/predict`.

## Repository layout

```
src/config.py         paths, SEED, date splits, bounding box, thresholds, Diwali dates
src/download.py       FIRMS + Open-Meteo fetch (cached); checks the Kaggle file
src/clean_aqi.py      Delhi filter, dtypes, duplicates, impossible values, missingness table
src/clean_fires.py    vegetation/confidence filter, bounding box, UTC→IST, daily counts (0 not NaN)
src/clean_weather.py  JSON → daily table, boundary-layer mean/min, gap assertion
src/features.py       past-only features: persistence, weather, calendar, fires; target = PM2.5 at t+1
src/models.py         Persistence, seasonal mean, Ridge, Random Forest, HistGradientBoosting (log-target pipelines)
src/train.py          TimeSeriesSplit CV tuning on 2015–17, selection on 2018, refit on 2015–18
src/evaluate.py       single test-set evaluation, paired bootstrap, alerts, error analysis, importance
src/robustness.py     rolling-origin evaluation (2017/18/19 seasons) at 1-3 day horizons
src/plots.py          all figures → reports/figures/
tests/                leakage test: scrambling the future must not change any past feature
notebooks/01_eda.ipynb      EDA with a written insight under every plot
notebooks/02_results.ipynb  model comparison, G1–G3, error analysis, interpretation, limitations
reports/              report.md, figures/, metrics/ (CSV/JSON outputs)
models/               fitted models (joblib): 1-day models + h2_/h3_ models for 2- and 3-day forecasts
src/inference.py      raw inputs -> features -> forecasts (used by the website)
web/                  Flask app (app.py) + static front end (index.html, style.css, app.js)
```

## Methodology at a glance

- **Split by time:** train 2015–2017, validation 2018, test 2019 (touched once), with 2020 H1 as an extra stress test (COVID lockdown). The split uses the date being predicted.
- **No leakage:** lags and rolling windows only look back; forward-fill only (no interpolation); the target is never imputed; preprocessing sits inside sklearn Pipelines; a unit test proves it.
- **Models:** persistence and seasonal-mean baselines; Ridge, Random Forest and HistGradientBoosting, each trained with and without the fire features. Models are fitted on log(PM2.5), and metrics are reported in µg/m³.
- **Robustness:** each 2017–2019 season is re-tested with models trained only on earlier years, at 1–3 day horizons.
- **Evaluation:** MAE (main metric), RMSE and R², overall and for the 15 Oct – 30 Nov burning season; a paired bootstrap (2,000 resamples, plus a 7-day block bootstrap for autocorrelation); Severe-day precision/recall; error analysis; permutation and group importance; partial dependence.
