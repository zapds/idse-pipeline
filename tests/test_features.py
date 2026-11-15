"""Leakage test: features for day t must not change when anything after day t changes."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from config import AQI_CLEAN, FIRES_DAILY, WEATHER_CLEAN  # noqa: E402
from features import ALL_FEATURES, TARGET, build_features  # noqa: E402

CALENDAR_OF_TARGET = {"month_sin", "month_cos", "dow", "is_diwali_window", "is_fire_season"}


@pytest.fixture(scope="module")
def tables():
    if not AQI_CLEAN.exists():
        pytest.skip("run the cleaning steps first")
    return pd.read_parquet(AQI_CLEAN), pd.read_parquet(FIRES_DAILY), pd.read_parquet(WEATHER_CLEAN)


@pytest.mark.parametrize("horizon", [1, 3])
@pytest.mark.parametrize("cut", ["2016-11-05", "2018-03-01", "2019-10-30"])
def test_no_future_information(tables, cut, horizon):
    aqi, fires, weather = tables
    cut = pd.Timestamp(cut)
    base = build_features(aqi, fires, weather, horizon)

    # Scramble everything strictly after the cut-off day.
    rng = np.random.default_rng(0)
    a2, f2, w2 = aqi.copy(), fires.copy(), weather.copy()
    for df in (a2, f2, w2):
        num = df.select_dtypes("number").columns
        df[num] = df[num].astype("float64")
        after = df.index > cut
        df.loc[after, num] = rng.uniform(0, 999, size=(after.sum(), len(num)))
    pert = build_features(a2, f2, w2, horizon)

    # Calendar features describe t+1 but come from the calendar, not from data.
    cols = [c for c in ALL_FEATURES if c not in CALENDAR_OF_TARGET]
    pd.testing.assert_frame_equal(base.loc[:cut, cols], pert.loc[:cut, cols], check_dtype=False)
    # The target at day `cut` IS tomorrow's value, so it must change: proves the test bites.
    assert base.loc[cut, TARGET] != pert.loc[cut, TARGET]


@pytest.mark.parametrize("horizon", [1, 2, 3])
def test_target_is_h_days_ahead(tables, horizon):
    aqi, fires, weather = tables
    f = build_features(aqi, fires, weather, horizon)
    shifted = aqi.pm25.reindex(f.index).shift(-horizon)
    pd.testing.assert_series_equal(f[TARGET], shifted, check_names=False)
