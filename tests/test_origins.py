"""Retrain dates vs forecast dates (origin_step_days)."""

import numpy as np
import pandas as pd
import pytest

from forecast_fm.backtest import backtest
from forecast_fm.data import TS, Y
from forecast_fm.folds import fold_cutoffs, fold_origins, origin_limit
from forecast_fm.metrics import score
from forecast_fm.models import REGISTRY
from forecast_fm.models.base import Forecaster

from .conftest import make_exp, make_project, make_raw, panel_from


class Spy(Forecaster):
    log: list = []

    def fit(self, history, project, cutoff):
        Spy.log.append(("fit", cutoff, history[TS].max()))

    def predict(self, history, future, project):
        Spy.log.append(("predict", future[TS].min() - pd.Timedelta(days=1), history[TS].max()))
        return np.ones(len(future)), None


@pytest.fixture
def spy(monkeypatch):
    Spy.log = []
    monkeypatch.setitem(REGISTRY, "spy", Spy)
    return Spy


def _project(**kw):
    base = dict(horizon=14, horizon_buckets=[7, 14], n_folds=2, fold_step=28, min_train_periods=60,
                origin_step_days=7)
    base.update(kw)
    return make_project(**base)


def test_origins_stay_inside_their_fold_and_before_the_holdout():
    p = _project()
    first, last = pd.Timestamp("2024-01-01"), pd.Timestamp("2024-07-18")
    cut = fold_cutoffs(first, last, p)
    limit = origin_limit(first, last, p)
    origins = fold_origins(cut, p, limit)
    assert origins[0][0] == cut[0] and all(o < cut[1] for o in origins[0])
    assert len(origins[0]) == 4  # 28-day fold, weekly origins
    flat = [o for f in origins for o in f]
    assert all(o + pd.Timedelta(days=14) <= limit for o in flat)
    assert limit == fold_cutoffs(first, last, p, holdout=True)[0]


def test_fit_once_per_fold_and_context_never_passes_the_origin(spy):
    p = _project()
    panel = panel_from(make_raw(n_days=200), p)
    preds, _ = backtest(panel, p, make_exp("spy"))
    fits = [e for e in spy.log if e[0] == "fit"]
    predicts = [e for e in spy.log if e[0] == "predict"]
    assert len(fits) == 2  # one (re)training per fold, not per origin
    assert all(hmax <= cut for _, cut, hmax in fits)
    assert len(predicts) == preds["origin"].nunique() > 2
    assert all(hmax <= origin for _, origin, hmax in predicts)
    assert set(preds["model_age_days"]) >= {0, 7, 14, 21}


def test_each_origin_blind_to_its_future():
    p = _project()
    panel = panel_from(make_raw(n_days=200), p)
    base, _ = backtest(panel, p, make_exp("seasonal_naive"))
    o = sorted(base["origin"].unique())[1]  # a later origin of fold 0
    tampered = panel.copy()
    tampered.loc[tampered[TS] > o, Y] = 9999.0
    again, _ = backtest(tampered, p, make_exp("seasonal_naive"))
    m = base["origin"] == o
    assert np.allclose(base.loc[m, "y_pred"], again.loc[m, "y_pred"])


def test_model_age_table():
    p = _project()
    panel = panel_from(make_raw(n_days=200), p)
    preds, _ = backtest(panel, p, make_exp("seasonal_naive"))
    t = score(preds, p)["tables"]["model_age"]
    assert list(t["model_age_weeks"]) == [0, 1, 2, 3]


def test_without_origin_step_one_origin_per_fold(spy):
    p = _project(origin_step_days=None)
    panel = panel_from(make_raw(n_days=200), p)
    preds, _ = backtest(panel, p, make_exp("spy"))
    assert (preds["origin"] == preds["cutoff"]).all()
    assert "model_age" not in score(preds, p)["tables"]
