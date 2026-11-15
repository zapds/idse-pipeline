# PLAN: Do Punjab–Haryana farm fires help predict Delhi's smog?

## 1. Problem statement

Every October–November, Delhi's air quality falls to "Very Poor" or "Severe". Crop-residue
(stubble) burning in Punjab and Haryana is blamed every year, but its contribution is disputed.
Winter weather (low wind, a shallow mixing layer), Diwali fireworks, traffic and dust also matter.
Authorities act on next-day forecasts: under Delhi-NCR's Graded Response Action Plan (GRAP),
restrictions are triggered by AQI levels.

**Research question:** Does adding satellite-detected farm-fire activity in Punjab and Haryana
improve **next-day** prediction of Delhi's PM2.5 beyond what yesterday's pollution and weather
already tell us? By how much does it help during the burning season?

**Measurable goals** (all judged on the held-out 2019 test year):

| ID | Goal | Success criterion |
|----|------|-------------------|
| G1 | Beat the naive forecast | Best model MAE < persistence baseline MAE ("tomorrow = today"), overall and in the burning season |
| G2 | Quantify the value of fire data | Fire-season (15 Oct – 30 Nov) MAE of the *with-fires* model ≥ 10% lower than the *no-fires* model, with a paired-bootstrap 95% CI for the difference that excludes 0 |
| G3 | Useful for alerts | Recall ≥ 0.7 on "Severe" days (PM2.5 > 250 µg/m³) when predictions are thresholded, with precision reported alongside |

Whether G2 succeeds or fails, the result answers the question. A negative result is still a valid
finding and must be reported honestly.

---

## 2. Data sources (all public and free; access checked 2026-11-15)

| Source | What | Access | Role |
|--------|------|--------|------|
| **CPCB air quality**, Kaggle "Air Quality Data in India (2015–2020)", `rohanrao/air-quality-data-in-india`, file `city_day.csv` | Daily city-level pollutants (PM2.5, PM10, NO2, CO, SO2, O3, …) and AQI for 26 cities, Jan 2015 – Jul 2020 | Kaggle login (CLI token, or manual download into `data/raw/`) | Target and lag features (filter `City == "Delhi"`) |
| **NASA FIRMS MODIS C6.1** active fires, country file for India | One row per detected fire pixel: lat, lon, date, time (UTC), confidence, FRP (fire radiative power), type | Direct download, no key: `https://firms.modaps.eosdis.nasa.gov/data/country/modis/{YEAR}/modis_{YEAR}_India.csv` (verified HTTP 200) | Fire features |
| **Open-Meteo historical archive** (ERA5 reanalysis) for Delhi (28.61 N, 77.21 E) | Daily wind speed, dominant wind direction, temperature, precipitation, humidity; hourly boundary-layer height | Free JSON API, no key: `archive-api.open-meteo.com/v1/archive` (verified) | Weather confounders |

**Why these sources:**
- **Source:** CPCB is the government monitoring agency, NASA FIRMS is the standard satellite fire product, and ERA5 is the standard reanalysis dataset. All three are authoritative, not scraped.
- **Size:** about 2,000 Delhi days, and tens of thousands of fire detections per year in Punjab and Haryana.
- **Relevance:** the three sources map directly onto the target, the hypothesised cause and the main confounder.
- **Why add weather at all:** without it, any "fire effect" could simply be winter weather. The no-fires model must be strong for the comparison to be fair.

**Why MODIS rather than VIIRS:** MODIS gives a consistent record across all years. VIIRS is more
sensitive and can be used as an optional robustness check.

---

## 3. Repository layout

```
idse_pipeline/
├── PLAN.md
├── README.md               # what, why, how to run, results summary
├── requirements.txt        # pinned versions
├── .gitignore              # data/raw, data/interim, __pycache__, .ipynb_checkpoints
├── run_all.sh              # one command: download → clean → features → train → evaluate → figures
├── data/
│   ├── raw/                # untouched downloads (not committed)
│   ├── interim/            # cleaned per-source tables
│   └── processed/          # final modelling table (delhi_daily.parquet)
├── src/
│   ├── config.py           # paths, SEED=42, date splits, region bbox, thresholds
│   ├── download.py         # FIRMS + Open-Meteo fetch; checks Kaggle file exists
│   ├── clean_aqi.py
│   ├── clean_fires.py
│   ├── clean_weather.py
│   ├── features.py         # merge + feature engineering (past-only)
│   ├── models.py           # baseline + model definitions as sklearn Pipelines
│   ├── train.py            # CV tuning, model selection on validation
│   ├── evaluate.py         # final test metrics, bootstrap, error analysis
│   └── plots.py            # all figures → reports/figures/
├── notebooks/
│   ├── 01_eda.ipynb        # EDA with a written insight under each plot
│   └── 02_results.ipynb    # model comparison, error analysis, interpretation
└── reports/
    ├── figures/
    ├── metrics/            # JSON/CSV outputs from evaluate.py
    └── report.md → report.pdf
```

Principle: the logic lives in `src/` and the notebooks only call it and explain it. That keeps the
code clean and makes the pipeline runnable without Jupyter.

---

## 4. Phase-by-phase plan

### Phase 0: Setup
- [ ] Create a venv and `requirements.txt` (pandas, numpy, scikit-learn, matplotlib, seaborn, requests, pyarrow, jupyter, nbconvert). Pin the versions.
- [ ] Put `SEED = 42` in `config.py` and pass it to every `random_state`, to the bootstrap and to `np.random.default_rng`.
- [ ] Write `.gitignore` and `git init`.

### Phase 1: Acquisition (`download.py`)
- [ ] FIRMS: download the India yearly CSVs for 2015–2020 and cache them in `data/raw/firms/`. Skip files that already exist.
- [ ] Open-Meteo: request the daily variables (`temperature_2m_mean`, `relative_humidity_2m_mean`, `wind_speed_10m_max`, `wind_direction_10m_dominant`, `precipitation_sum`) and hourly `boundary_layer_height` for 2015-01-01 → 2020-07-01 with `timezone=Asia/Kolkata`. Save the raw JSON.
- [ ] AQI: check that `data/raw/city_day.csv` exists. If it doesn't, print the exact Kaggle CLI command (`kaggle datasets download -d rohanrao/air-quality-data-in-india`) and stop.
- [ ] Log the row counts and date ranges of every raw file.

### Phase 2: Cleaning (each decision is written down and justified in the report)

**AQI (`clean_aqi.py`)**
- [ ] Filter to Delhi. Parse `Date` and set the dtype of every column explicitly.
- [ ] **Duplicates:** check for duplicate dates and assert that none remain.
- [ ] **Missing values:** produce a per-column missingness table and a heatmap over time.
  - Drop pollutant columns with more than 40% missing.
  - Fill feature columns **forward only** (ffill, limit 3 days). Never use interpolation, because it uses future values and would leak.
  - **Never impute the target.** Days with a missing next-day PM2.5 are dropped from training and evaluation.
- [ ] **Outliers:**
  - Physically impossible values (negative values, or PM2.5 above about 1,000) are treated as sensor errors and set to NaN.
  - Extreme but real values (Diwali, smog episodes) are **kept**, because they are exactly what we want to predict. The reasoning goes in the report.
- [ ] Add the CPCB PM2.5 category labels (Good ≤30, Satisfactory ≤60, Moderate ≤90, Poor ≤120, Very Poor ≤250, Severe >250).

**Fires (`clean_fires.py`)**
- [ ] Concatenate the yearly files. Fix dtypes, and read `acq_time` as a zero-padded string.
- [ ] Keep `type == 0` (presumed vegetation fire) and `confidence >= 30`, which drops low-confidence detections.
- [ ] **Region:** filter to a Punjab + Haryana bounding box (about lat 27.6–32.6 N, lon 73.8–77.6 E). Plot the points on a scatter map to check visually. Note in the limitations that the box also includes slivers of neighbouring states.
- [ ] **Time zone:** convert UTC acquisition time to an IST date before aggregating to daily.
- [ ] **Duplicates:** drop exact duplicate detections (same lat, lon, time and satellite).
- [ ] Aggregate to daily `fire_count` and `fire_frp_sum`. Days with no detections become **0, not NaN**: absence of fire is real information. Satellite outages are noted as a limitation.

**Weather (`clean_weather.py`)**
- [ ] Turn the JSON into a DataFrame. Aggregate hourly boundary-layer height to a daily mean and minimum.
- [ ] Check for gaps; ERA5 should have none, but assert it.

### Phase 3: EDA (`01_eda.ipynb`): each plot is followed by a written "What this shows" insight
1. Delhi PM2.5 time series 2015–2020 with the 15 Oct – 30 Nov windows shaded. The seasonal spike is the motivation for the project.
2. Monthly PM2.5 box plots showing the seasonality.
3. Daily fire count in Punjab and Haryana over time, overlaid on PM2.5. The two seasons line up.
4. Map of fire detections, showing that the bounding box captures the burning belt.
5. Lagged cross-correlation between fires and PM2.5 at lags of 0–5 days. This tells us which lags to engineer.
6. PM2.5 against wind speed and against boundary-layer height. This is the weather confounder, made visible.
7. Wind-rose or bar chart of dominant wind direction on high- versus low-PM2.5 days. On bad days, is the wind from the NW, i.e. from Punjab?
8. Correlation heatmap of the candidate features.
9. Diwali spikes annotated, a confounder to control for.

### Phase 4: Feature engineering (`features.py`)
**Rule:** every feature for predicting day *t+1* uses only information available by the end of day *t*.

| Group | Features |
|-------|----------|
| Persistence | `pm25_t`, `pm25_lag1..3`, `pm25_roll3_mean`, `pm25_roll7_mean`, `pm10_t`, `no2_t`, `co_t` |
| Weather (day *t*) | wind speed, dominant direction encoded as sin/cos, `nw_wind` flag (direction 270–360°, i.e. blowing from Punjab), temperature, humidity, precipitation, boundary-layer height mean and min |
| Calendar | month encoded as sin/cos, day-of-week, `is_diwali_window` (Diwali ±2 days, dates hard-coded and checked), `is_fire_season` |
| **Fires** (the tested group) | `fire_count_t`, `fire_count_lag1..3`, `fire_roll3_sum`, `fire_roll7_sum`, `log1p` of counts, `fire_frp_sum_t`, interaction `fire_roll3_sum × nw_wind` (smoke only reaches Delhi when the wind carries it) |

- **Target:** `pm25_{t+1}`. Models are fitted on `log1p(pm25)`, because PM2.5 is right-skewed and its errors scale with level. Predictions are back-transformed and metrics are computed in µg/m³.
- **Weather for day t+1:** we use day-*t* weather only. Using the actual day-*t+1* weather would mean pretending we have a perfect forecast. That could be an optional "oracle weather" variant, clearly labelled.

### Phase 5: Modelling (`models.py`, `train.py`)

**Split by time, not randomly.** A random split leaks, because neighbouring days are nearly identical.

| Set | Period | Fire seasons | Use |
|-----|--------|--------------|-----|
| Train | 2015-01-01 → 2017-12-31 | 2015, 2016, 2017 | Fitting, plus `TimeSeriesSplit(n_splits=5)` CV for tuning |
| Validation | 2018 | 2018 | Model selection |
| Test | 2019 | 2019 | **Touched once**, at the very end |
| (Extra) | 2020 H1 | none | Optional stress test: the COVID lockdown is a distribution shift |

After selecting the model, refit on 2015–2018 and evaluate once on 2019.

**No leakage:** scaling and imputation live inside an sklearn `Pipeline`, so they are fitted only
on the training folds. All lag and rolling features use `.shift()`, and a unit test asserts that
no feature uses data from *t+1*.

**Models.** Each one except the baseline is trained twice, *without* and *with* the fire
features. That ablation is the core of the project.

| Model | Why |
|-------|-----|
| B0 Persistence ("tomorrow = today") | The honest baseline. Air pollution is very autocorrelated, so it is hard to beat. |
| B1 Seasonal mean (mean PM2.5 for that calendar month in training) | Shows what seasonality alone explains |
| M1 Ridge regression | Interpretable coefficients and linear effects; the regularisation strength `alpha` is tuned by CV |
| M2 Random Forest | Captures non-linear effects and interactions such as fire × wind; `max_depth`, `min_samples_leaf` and `n_estimators` tuned |
| M3 HistGradientBoostingRegressor | Usually the strongest on tabular data, and handles NaN natively; `learning_rate`, `max_depth` and `max_iter` tuned |

Tuning: `GridSearchCV(cv=TimeSeriesSplit(5), scoring="neg_mean_absolute_error")` on the training
period. Small grids, all with seed 42.

### Phase 6: Evaluation and interpretation (`evaluate.py`, `02_results.ipynb`)
- [ ] **Metrics:**
  - MAE (main metric: interpretable in µg/m³ and robust), RMSE (penalises big misses), R².
  - Each reported **overall** and for the **fire season** separately.
- [ ] **Comparison table:** every model × {no fires, with fires} × {overall, fire season}.
- [ ] **G2 test:** paired bootstrap (2,000 resamples, seed 42) over fire-season test days, giving a 95% CI for MAE(no fires) − MAE(with fires).
- [ ] **Alerts (G3):** threshold the predictions at 250 to flag "Severe" days. Report a confusion matrix, precision and recall, and the category-level confusion (6 CPCB classes).
- [ ] **Error analysis:**
  - Residuals over time, by month and by actual PM2.5 level. We expect underprediction at extremes.
  - The 10 worst test days, looked up against events: Diwali, dust storms, rain.
  - Error on NW-wind versus other-wind days.
- [ ] **Interpretation:**
  - Ridge coefficients (standardised).
  - Permutation importance on the test set for the best tree model.
  - Partial dependence of predicted PM2.5 on `fire_roll3_sum`, split by wind direction.
- [ ] **Limitations and bias:**
  - This is a predictive association, not a causal attribution.
  - Satellites miss small fires and fires under cloud or haze, and burning has shifted to later evening times, outside overpass hours.
  - The bounding box is approximate.
  - The city-average PM2.5 hides differences between stations.
  - Only 5 fire seasons, and the data ends in 2020.
  - Weather is a single grid point.
- [ ] **Next steps:**
  - Add VIIRS fires.
  - Use proper state polygons.
  - Use forecast weather instead of observed weather.
  - Extend to 2021–2025 with newer CPCB data.
  - Predict 2–3 days ahead.
  - Model the trajectory of air arriving at Delhi (HYSPLIT).

### Phase 7: Reproducibility and code quality
- [ ] `run_all.sh` runs the full pipeline from a fresh clone (after the Kaggle file is in place) and regenerates every figure and metric.
- [ ] `nbconvert --execute` on both notebooks runs without errors.
- [ ] README covers the question, data sources and licences, setup, the one-command run, the expected runtime and a short summary of results.
- [ ] Docstrings on public functions, comments on the *why*, and no dead code.
- [ ] Run a final clean-environment test: new venv → `pip install -r requirements.txt` → `run_all.sh`.

### Phase 8: Report and viva
- [ ] `reports/report.md` exported to PDF, about 8–10 pages:
  1. Problem and motivation
  2. Data and justification
  3. Cleaning decisions (as a table: issue → action → reason)
  4. EDA insights
  5. Method, with the split diagram
  6. Results table
  7. Error analysis
  8. Limitations
  9. Conclusion
- [ ] Every conclusion cites a figure or table number.
- [ ] Demo: run `run_all.sh` live (or show cached outputs), then walk through 4–5 key figures.
- [ ] Viva prep. Have answers ready for:
  - Why not split randomly?
  - Why use MAE?
  - Why log-transform the target?
  - How do you know there's no leakage?
  - Does this prove the fires cause the smog?
  - Why keep the outliers?
  - What if the bootstrap CI includes 0?

---

## 5. Rubric coverage

| Criterion (marks) | Where it is covered |
|-------------------|---------------------|
| Problem and data (1) | §1 question and measurable goals G1–G3; §2 source, size and relevance |
| Data prep and EDA (2) | Phase 2 cleaning with documented reasons; Phase 3 nine plots with insights; Phase 4 feature engineering |
| Modelling (2.5) | Two baselines and three models, the fire ablation, train/val/test split by time, TimeSeriesSplit CV with GridSearch, leakage controls |
| Evaluation (2) | MAE/RMSE/R², seasonal breakdown, bootstrap CI, alert precision/recall, error analysis, limitations and next steps |
| Reproducibility (1) | Repo structure, pinned requirements, SEED=42, `run_all.sh`, README |
| Report (1.5) | Phase 8 report, figures and viva prep |

## 6. Risks and fallbacks

| Risk | Fallback |
|------|----------|
| Kaggle download blocked | Manual download from the dataset page, or pull the same CPCB data from the CPCB CCR portal |
| Fire features give no improvement (G2 fails) | Report it honestly as a finding. Discuss whether persistence already captures the smoke signal, then try VIIRS and a longer lag window |
| Too many Delhi days missing in 2015 | Start training from the first month with less than 20% missing PM2.5 and record it |
| Too few Severe days in 2019 for stable recall | Also report "Very Poor or worse" (> 120) alerts |

## 7. Order of work
1. Phases 0–1: environment and data downloaded
2. Phase 2: clean tables saved to `data/interim/`
3. Phase 3: EDA notebook with insights
4. Phase 4: modelling table plus the leakage unit test
5. Phase 5: baselines first, then models
6. Phase 6: evaluation, bootstrap and error analysis
7. Phase 7: clean-environment run-through
8. Phase 8: report, rehearse demo and viva
