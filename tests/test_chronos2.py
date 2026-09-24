"""Chronos-2 wrapper. Stub-pipeline tests need no torch; the tiny-model
tests build a random 1-layer Chronos-2 offline and skip without
chronos-forecasting installed."""

import numpy as np
import pandas as pd
import pytest

from forecast_fm.backtest import backtest, qcol
from forecast_fm.data import SERIES, STOCKOUT, TS, Y
from forecast_fm.models import chronos2, create_model
from forecast_fm.models.chronos2 import interp_quantiles
from forecast_fm.plans import HORIZON, future_frame

from .conftest import make_exp, make_project, make_raw, panel_from

LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


class _T:
    def __init__(self, a):
        self.a = a

    def detach(self):
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.a


class StubPipeline:
    """Forecast = last finite context value + 10*level + h."""

    quantiles = LEVELS

    def __init__(self):
        self.calls = []

    def predict(self, inputs, prediction_length, cross_learning, batch_size):
        self.calls.append(dict(inputs=inputs, cross=cross_learning, batch=batch_size))
        out = []
        for d in inputs:
            t = d["target"][np.isfinite(d["target"])]
            last = float(t[-1]) if len(t) else 0.0
            q = np.array([[last + 10 * lv + h for h in range(1, prediction_length + 1)] for lv in LEVELS])
            out.append(_T(q[None]))
        return out


@pytest.fixture
def stub(monkeypatch):
    monkeypatch.setattr(chronos2, "resolve_device", lambda d: "cpu")
    pipe = StubPipeline()
    monkeypatch.setitem(chronos2._PIPELINES, ("stub", "cpu", "float32"), pipe)
    return pipe


def _fit(project, panel, cutoff, **params):
    m = create_model("chronos2", {"model_id": "stub", **params})
    hist = panel[panel[TS] <= cutoff]
    m.fit(hist, project, cutoff)
    fut = future_frame(hist, cutoff, project, None, actuals=panel[panel[TS] > cutoff], verbose=False)
    return m, hist, fut


def test_covariate_channels(stub):
    p = make_project()
    panel = panel_from(make_raw(n_series=2), p)
    cutoff = panel[TS].max() - pd.Timedelta(days=20)
    m, hist, fut = _fit(p, panel, cutoff, context_length=50)
    m.predict(hist, fut, p)
    d = stub.calls[0]["inputs"][0]
    assert len(d["target"]) == 50
    assert set(d["past_covariates"]) == {"price", "promo_flag", "promo_type", "sessions", "in_stock"}
    assert set(d["future_covariates"]) == {"price", "promo_flag", "promo_type"}  # never past cols
    assert "category" not in d["past_covariates"]  # statics are not covariates
    assert len(d["future_covariates"]["price"]) == p.horizon
    assert d["future_covariates"]["promo_type"].dtype.kind == "U"


def test_stockouts_become_missing(stub):
    p = make_project()
    panel = panel_from(make_raw(n_series=1), p)
    cutoff = panel[TS].max() - pd.Timedelta(days=20)
    m, hist, fut = _fit(p, panel, cutoff)
    m.predict(hist, fut, p)
    t = stub.calls[0]["inputs"][0]["target"]
    assert np.array_equal(np.isnan(t), hist[STOCKOUT].to_numpy())


def test_quantiles_align_and_interpolate(stub):
    p = make_project(quantiles=[0.1, 0.25, 0.5, 0.95])
    panel = panel_from(make_raw(n_series=3), p)
    cutoff = panel[TS].max() - pd.Timedelta(days=20)
    m, hist, fut = _fit(p, panel, cutoff)
    point, qd = m.predict(hist, fut, p)
    h = hist[~hist[STOCKOUT]].sort_values(TS)
    last = fut[SERIES].map(h.groupby(SERIES, observed=True)[Y].last()).astype(float).to_numpy()
    assert np.allclose(point, last + 5 + fut[HORIZON])
    assert np.allclose(qd[0.25], last + 2.5 + fut[HORIZON])
    assert np.allclose(qd[0.95], last + 9 + fut[HORIZON])  # clamped at the top level


def test_streaming_chunks_equal_single_pass(stub):
    p = make_project()
    panel = panel_from(make_raw(n_series=5), p)
    cutoff = panel[TS].max() - pd.Timedelta(days=20)
    m1, hist, fut = _fit(p, panel, cutoff, chunk_series=2)
    a, _ = m1.predict(hist, fut, p)
    assert [len(c["inputs"]) for c in stub.calls] == [2, 2, 1]
    m2, _, _ = _fit(p, panel, cutoff, chunk_series=100)
    b, _ = m2.predict(hist, fut, p)
    assert np.array_equal(a, b)


def test_group_by_cross_learning_groups(stub):
    p = make_project()
    panel = panel_from(make_raw(n_series=6), p)
    cutoff = panel[TS].max() - pd.Timedelta(days=20)
    m, hist, fut = _fit(p, panel, cutoff, group_by=["category"], group_size=2, batch_size=4)
    m.predict(hist, fut, p)
    assert all(c["cross"] for c in stub.calls)
    assert sorted(len(c["inputs"]) for c in stub.calls) == [1, 1, 2, 2]  # 3 A + 3 B, size <= 2
    n_var = 6  # target + 3 known + 2 past
    assert all(c["batch"] >= len(c["inputs"]) * n_var for c in stub.calls)


def test_missing_future_values_are_nan(stub):
    p = make_project()
    panel = panel_from(make_raw(n_series=1), p)
    cutoff = panel[TS].max() - pd.Timedelta(days=20)
    m, hist, fut = _fit(p, panel, cutoff)
    fut.loc[fut[HORIZON] == 3, "price"] = np.nan
    m.predict(hist, fut, p)
    price = stub.calls[0]["inputs"][0]["future_covariates"]["price"]
    assert np.isnan(price[2]) and not np.isnan(price[[0, 1, 3]]).any()


def test_backtest_blind_to_future(stub):
    p = make_project()
    panel = panel_from(make_raw(), p)
    exp = make_exp("chronos2", model_id="stub")
    base, _ = backtest(panel, p, exp)
    tampered = panel.copy()
    late = tampered[TS] > base["cutoff"].min()
    tampered.loc[late, Y] = 9999.0
    tampered.loc[late, "sessions"] = 1e6
    again, _ = backtest(tampered, p, exp)
    first = base["cutoff"] == base["cutoff"].min()
    assert np.allclose(base.loc[first, qcol(0.9)], again.loc[first, qcol(0.9)])


def test_interp_quantiles():
    q = np.array([[1.0, 2.0, 3.0]])
    out = interp_quantiles([0.1, 0.5, 0.9], q, [0.05, 0.3, 0.5, 0.95])
    assert np.allclose(out, [[1.0, 1.5, 2.0, 3.0]])


def test_param_validation():
    with pytest.raises(ValueError, match="unknown params"):
        create_model("chronos2", {"bogus": 1})
    with pytest.raises(ValueError, match="mode"):
        create_model("chronos2", {"fine_tune": {"mode": "qlora"}})


# --- real library: tiny random Chronos-2 --------------------------------------


@pytest.mark.parametrize("params", [{}, {"group_by": ["category"], "group_size": 3},
                                    {"dtype": "bfloat16"}])
def test_real_backtest_all_covariate_kinds(tiny, params):
    p = make_project()
    preds, stats = backtest(panel_from(make_raw(), p), p,
                            make_exp("chronos2", model_id=tiny, device="cpu", **params))
    assert preds["y_pred"].notna().all() and (preds["y_pred"] >= 0).all()
    qs = preds[[qcol(q) for q in p.quantiles]].to_numpy()
    assert (np.diff(qs, axis=1) >= 0).all()
    assert stats[0]["series_per_second"] > 0


def test_real_fine_tune_cache_is_keyed_by_cutoff(tiny, tmp_path):
    p = make_project()
    panel = panel_from(make_raw(n_series=3), p)
    params = {"model_id": tiny, "device": "cpu", "cache_dir": str(tmp_path / "ft"),
              "fine_tune": {"num_steps": 2, "batch_size": 4, "learning_rate": 1e-4}}
    c1 = panel[TS].max() - pd.Timedelta(days=30)
    m = create_model("chronos2", params)
    m.fit(panel[panel[TS] <= c1], p, c1)
    assert m.stats["fine_tune_cache"] == "miss"
    m = create_model("chronos2", params)
    m.fit(panel[panel[TS] <= c1], p, c1)
    assert m.stats["fine_tune_cache"] == "hit"
    c2 = c1 - pd.Timedelta(days=7)
    m = create_model("chronos2", params)
    m.fit(panel[panel[TS] <= c2], p, c2)
    assert m.stats["fine_tune_cache"] == "miss"  # another cutoff never reuses a checkpoint
    assert len(list((tmp_path / "ft").iterdir())) == 2
    hist = panel[panel[TS] <= c2]
    point, _ = m.predict(hist, future_frame(hist, c2, p, None, actuals=panel[panel[TS] > c2],
                                            verbose=False), p)
    assert np.isfinite(point).all()
