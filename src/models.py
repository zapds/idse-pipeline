"""Phase 5: baselines and model definitions.

Each learned model is an sklearn Pipeline wrapped in a TransformedTargetRegressor, so that:
- imputation and scaling are fitted only on the training folds (no leakage);
- the model is fitted on log1p(PM2.5), because PM2.5 is right-skewed and its errors grow with
  its level, while predictions come back in ug/m3 for the metrics.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import SEED


class Persistence(BaseEstimator, RegressorMixin):
    """B0: tomorrow's PM2.5 = today's PM2.5."""

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return np.asarray(X["pm25_t"], dtype=float)


class SeasonalMean(BaseEstimator, RegressorMixin):
    """B1: mean training PM2.5 for the calendar month of the target day."""

    def fit(self, X, y):
        month = self._month(X)
        self.means_ = pd.Series(np.asarray(y)).groupby(month).mean()
        self.overall_ = float(np.mean(y))
        return self

    def predict(self, X):
        return self._month(X).map(self.means_).fillna(self.overall_).to_numpy()

    @staticmethod
    def _month(X):
        # Recover the target month from its sin/cos encoding (1..12).
        ang = np.arctan2(X["month_sin"].to_numpy(), X["month_cos"].to_numpy())
        m = np.round(ang / (2 * np.pi) * 12).astype(int) % 12
        return pd.Series(np.where(m == 0, 12, m))


def _log_target(reg) -> TransformedTargetRegressor:
    return TransformedTargetRegressor(regressor=reg, func=np.log1p, inverse_func=np.expm1,
                                      check_inverse=False)


def make_ridge():
    return _log_target(Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", Ridge()),
    ]))


def make_rf():
    return _log_target(Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("model", RandomForestRegressor(random_state=SEED, n_jobs=-1)),
    ]))


def make_hgb():
    # HGB handles NaN natively, so no imputer.
    return _log_target(Pipeline([
        ("model", HistGradientBoostingRegressor(random_state=SEED)),
    ]))


# Small grids: ~1,100 training days do not justify large searches.
MODELS = {
    "Ridge": (make_ridge, {"regressor__model__alpha": [0.1, 1, 10, 100]}),
    "RandomForest": (make_rf, {
        "regressor__model__n_estimators": [300],
        "regressor__model__max_depth": [6, 10, None],
        "regressor__model__min_samples_leaf": [3, 10],
        "regressor__model__max_features": [0.5, 1.0],
    }),
    "HistGB": (make_hgb, {
        "regressor__model__learning_rate": [0.03, 0.1],
        "regressor__model__max_depth": [3, 6],
        "regressor__model__max_iter": [200, 500],
        "regressor__model__min_samples_leaf": [20],
        "regressor__model__l2_regularization": [1.0],
    }),
}

BASELINES = {"Persistence": Persistence, "SeasonalMean": SeasonalMean}
