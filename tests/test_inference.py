"""The website's feature path must reproduce the training features exactly."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from config import DAILY_INPUTS, MODEL_TABLE, MODELS_DIR  # noqa: E402
from features import ALL_FEATURES  # noqa: E402


@pytest.fixture(scope="module")
def inference():
    if not (DAILY_INPUTS.exists() and (MODELS_DIR / "RandomForest_with_fires.joblib").exists()):
        pytest.skip("run the pipeline first")
    import inference as inf
    return inf


@pytest.mark.parametrize("date", ["2016-11-05", "2018-07-14", "2019-10-30", "2019-12-25"])
def test_features_match_training_table(inference, date):
    table = pd.read_parquet(MODEL_TABLE)
    got = inference.features_from_inputs(inference.inputs_for_date(date)["inputs"])
    want = table.loc[[pd.Timestamp(date)], ALL_FEATURES]
    np.testing.assert_allclose(got[ALL_FEATURES].to_numpy(float), want.to_numpy(float), rtol=1e-4)  # inputs are rounded to 2 dp for display


def test_predict_returns_all_horizons(inference):
    out = inference.predict(inference.inputs_for_date("2019-11-01")["inputs"])
    assert set(out["horizons"]) == {"1", "2", "3"}
    assert out["horizons"]["1"]["main"] > 0
    assert len(out["fire_what_if"]) == len(inference.FIRE_MULTIPLIERS)


def test_rejects_bad_input(inference):
    inp = inference.inputs_for_date("2019-11-01")["inputs"]
    with pytest.raises(ValueError):
        inference.predict({**inp, "pm25": inp["pm25"][:3]})
