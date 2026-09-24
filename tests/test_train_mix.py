"""fine_tune.train_mix: which series a fine-tune trains on."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from forecast_fm.backtest import backtest
from forecast_fm.data import SERIES, TS, Y
from forecast_fm.models import chronos2, create_model
from forecast_fm.train_mix import _capped_counts, select

from .conftest import make_exp, make_project, make_raw, panel_from


def _panel(n_bau=6, n_int=18, n_days=200):
    """Mostly intermittent catalog, labeled by demand_label."""
    raw = make_raw(n_series=n_bau + n_int, n_days=n_days)
    ids = [f"S{i}" for i in range(n_bau + n_int)]
    raw["demand_label"] = raw["sku"].map({s: ("BAU" if i < n_bau else "intermittent")
                                          for i, s in enumerate(ids)})
    p = make_project(horizon=14, horizon_buckets=[7, 14])
    return panel_from(raw, p), p


def test_capped_counts():
    avail = pd.Series({"BAU": 60, "intermittent": 240})
    n = _capped_counts(avail, {"intermittent": 0.25})
    assert n["BAU"] == 60 and n["intermittent"] == 20  # 20 / 80 = 25%
    assert _capped_counts(avail, {"intermittent": 0.9})["intermittent"] == 240  # not binding
    n = _capped_counts(pd.Series({"a": 10, "b": 100, "c": 100}), {"b": 0.2, "c": 0.3})
    assert n["a"] == 10 and n["b"] <= 0.2 * n.sum() + 1 and n["c"] <= 0.3 * n.sum() + 1


def test_select_caps_share_and_is_deterministic():
    panel, p = _panel()
    mix = {"max_share": {"intermittent": 0.25}, "seed": 1}
    ids, rep = select(panel, p, mix)
    assert rep["selected"] == {"BAU": 6, "intermittent": 2}
    assert rep["share"]["intermittent"] == 0.25 and len(ids) == 8
    again, _ = select(panel, p, mix)
    assert list(ids) == list(again)
    other, _ = select(panel, p, {**mix, "seed": 2})
    assert len(other) == 8 and set(other) >= {f"S{i}" for i in range(6)}  # all BAU kept


def test_select_classes_activity_and_max_series():
    panel, p = _panel()
    ids, rep = select(panel, p, {"classes": ["BAU"]})
    assert rep["selected"] == {"BAU": 6} and rep["excluded_class"] == 18
    # series with no sale in the lookback window are dropped
    dead = panel.copy()
    dead.loc[dead[SERIES].isin(["S0", "S1"]) & (dead[TS] > dead[TS].max() - pd.Timedelta(days=90)), Y] = 0
    ids, rep = select(dead, p, {"min_nonzero_days": 1, "lookback_days": 90, "classes": ["BAU"]})
    assert "S0" not in ids and "S1" not in ids and rep["excluded_low_activity"] == 2
    ids, rep = select(panel, p, {"max_share": {"intermittent": 0.5}, "max_series": 6})
    assert len(ids) == 6 and rep["selected"] == {"BAU": 3, "intermittent": 3}


def test_select_uses_only_history():
    """Changing anything after the cutoff cannot change the training set."""
    panel, p = _panel()
    cutoff = panel[TS].max() - pd.Timedelta(days=30)
    mix = {"max_share": {"intermittent": 0.3}, "min_nonzero_days": 5, "lookback_days": 60, "seed": 3}
    base, _ = select(panel[panel[TS] <= cutoff], p, mix)
    tampered = panel.copy()
    tampered.loc[tampered[TS] > cutoff, Y] = 0.0
    again, _ = select(tampered[tampered[TS] <= cutoff], p, mix)
    assert list(base) == list(again)


def test_invalid_mix_rejected():
    with pytest.raises(ValueError, match="unknown keys"):
        create_model("chronos2", {"fine_tune": {"train_mix": {"share": 1}}})
    with pytest.raises(ValueError, match="sum to < 1"):
        create_model("chronos2", {"fine_tune": {"train_mix": {"max_share": {"a": 0.6, "b": 0.5}}}})


class _Model:
    def to(self, **kw):
        return self

    def eval(self):
        return self


class FitStub:
    """Pipeline stub whose fit records the training inputs."""

    quantiles = [0.1, 0.5, 0.9]

    def __init__(self):
        self.model = _Model()
        self.trained = []

    def fit(self, inputs, output_dir, **kw):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self.trained.append([np.asarray(d["target"]).copy() for d in inputs])
        return self

    def predict(self, inputs, prediction_length, cross_learning, batch_size):
        from .test_chronos2 import _T

        return [_T(np.zeros((1, 3, prediction_length))) for _ in inputs]


def test_backtest_training_set_blind_to_future(monkeypatch, tmp_path):
    stub = FitStub()
    monkeypatch.setattr(chronos2, "resolve_device", lambda d: "cpu")
    monkeypatch.setattr(chronos2, "resolve_dtype", lambda *a: None)
    monkeypatch.setitem(chronos2._PIPELINES, ("stub", "cpu", "float32"), stub)
    panel, p = _panel()
    mix = {"max_share": {"intermittent": 0.25}, "seed": 0}
    exp = make_exp("chronos2", model_id="stub", fine_tune={"num_steps": 1, "train_mix": mix},
                   cache_dir=str(tmp_path / "a"))
    _, stats = backtest(panel, p, exp)
    first = stub.trained[0]
    assert len(first) == 8  # 6 BAU + 2 intermittent, not all 24
    assert stats[0]["train_mix"]["share"]["intermittent"] == 0.25
    assert json.loads(next((tmp_path / "a").glob("*/key.json")).read_text())["train_mix"]

    tampered = panel.copy()
    tampered.loc[tampered[TS] > stats[0]["cutoff"], Y] = 9999.0
    stub.trained.clear()
    exp.model_params["cache_dir"] = str(tmp_path / "b")
    backtest(tampered, p, exp)
    assert len(stub.trained[0]) == len(first)
    assert all(np.array_equal(a, b, equal_nan=True) for a, b in zip(first, stub.trained[0], strict=True))


def test_real_fine_tune_with_mix_and_manifest(tiny, tmp_path):
    from forecast_fm.config import ExperimentConfig
    from forecast_fm.finetune import finetune

    panel, p = _panel(n_days=120)
    exp = ExperimentConfig(name="mix", hypothesis="h", model="chronos2", model_params={
        "model_id": tiny, "device": "cpu",
        "fine_tune": {"num_steps": 2, "batch_size": 4, "train_mix": {"max_share": {"intermittent": 0.25}}}})
    out = finetune(p, exp, panel, out=tmp_path / "m")
    man = json.loads((out / "forecast_fm_model.json").read_text())
    assert man["training"]["train_mix"]["selected"] == {"BAU": 6, "intermittent": 2}
    assert man["training"]["n_series"] == 24
