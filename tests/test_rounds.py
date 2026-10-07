"""Multi-round fine-tuning, the streaming catalog profile, and the memory guard."""

import json

import numpy as np
import pandas as pd
import pytest

from forecast_fm.config import ExperimentConfig
from forecast_fm.data import SERIES, TS
from forecast_fm.models import chronos2, create_model
from forecast_fm.train_mix import profile, select_from_profile, stream_profile

from .conftest import make_project, make_raw, panel_from


def _catalog(n=24, n_days=160):
    raw = make_raw(n_series=n, n_days=n_days)
    raw["demand_label"] = raw["sku"].map(lambda s: "BAU" if int(s[1:]) < n // 4 else "intermittent")
    return raw


def _project(**kw):
    return make_project(horizon=14, horizon_buckets=[7, 14], **kw)


@pytest.mark.parametrize("label", ["demand_label", None])
def test_stream_profile_equals_in_memory_profile(tmp_path, label):
    raw = _catalog()
    raw.to_parquet(tmp_path / "s.parquet")
    p = _project(demand_label_col=label)
    as_of = pd.Timestamp("2024-05-01")
    panel = panel_from(raw, p)
    mem = profile(panel[panel[TS] <= as_of], p, lookback_days=60)
    st = stream_profile(p, tmp_path / "s.parquet", as_of, lookback_days=60, batch_rows=500)
    st = st.reindex(mem.index)
    assert (mem["demand_class"] == st["demand_class"]).all()
    assert (mem["length"] == st["length"]).all()
    assert (mem["recent_nonzero"] == st["recent_nonzero"]).all()


def test_rounds_are_disjoint_stratified_and_stop_when_exhausted():
    p = _project()
    panel = panel_from(_catalog(), p)
    prof = profile(panel, p)
    mix = {"max_series": 8, "max_share": {"intermittent": 0.5}, "seed": 1}
    used, rounds = set(), []
    for k in range(10):
        ids, rep = select_from_profile(prof, p, mix, exclude=used, round_index=k)
        if not len(ids):
            break
        assert not used & set(ids)
        used |= set(ids)
        rounds.append(rep)
    assert rounds[0]["selected"] == {"BAU": 4, "intermittent": 4}  # stratified within the round
    # round 2 has 2 BAU left, so only 2 intermittent; then BAU is exhausted and the
    # 50% cap allows no intermittent-only round: the cap holds, rounds stop
    assert [r["selected"] for r in rounds[1:]] == [{"BAU": 2, "intermittent": 2}]
    assert len(used) == 12


def test_memory_guard():
    p = _project()
    prof = profile(panel_from(_catalog(), p), p)
    with pytest.raises(ValueError, match="max_series"):
        select_from_profile(prof, p, {}, max_pool_series=10)
    with pytest.raises(ValueError, match="rounds > 1 needs train_mix.max_series"):
        create_model("chronos2", {"fine_tune": {"rounds": 3}})


def test_memory_rounds_without_mix_guarded(monkeypatch):
    p = _project()
    panel = panel_from(_catalog(), p)
    m = create_model("chronos2", {"fine_tune": {"num_steps": 1, "max_pool_series": 5}})
    with pytest.raises(ValueError, match="limit 5"):
        m.memory_rounds(panel, p)


def _ft(rounds, mode="full", steps=6):
    return {"num_steps": steps, "batch_size": 4, "learning_rate": 1e-3, "mode": mode, "rounds": rounds,
            "log_every": 1, "train_mix": {"max_series": 4, "seed": 0}}


@pytest.mark.parametrize("mode", ["full", "lora"])
def test_real_multi_round_finetune(tiny, tmp_path, mode):
    if mode == "lora":
        pytest.importorskip("peft")
    from forecast_fm.finetune import finetune

    p = _project()
    panel = panel_from(_catalog(n=12), p)
    exp = ExperimentConfig(name="r", hypothesis="h", model="chronos2",
                           model_params={"model_id": tiny, "device": "cpu", "fine_tune": _ft(3, mode)})
    out = finetune(p, exp, panel, out=tmp_path / "m")
    man = json.loads((out / "forecast_fm_model.json").read_text())
    rounds = man["training"]["rounds"]
    assert len(rounds) == 3 and man["training"]["total_steps"] == 6
    assert [r["learning_rate"] for r in rounds] == pytest.approx([1e-3, 2e-3 / 3, 1e-3 / 3])
    assert sum(r["series"] for r in rounds) == 12  # disjoint draws cover the pool
    assert man["training"]["loss_curve"]  # training loss recorded
    assert not list(out.glob(".round-*"))  # intermediate rounds cleaned up
    # the merged full checkpoint loads without the base model and forecasts
    serve = create_model("chronos2", {"model_id": str(out), "device": "cpu"})
    serve.fit(panel, p, panel[TS].max())
    from forecast_fm.plans import future_frame

    fut = future_frame(panel, panel[TS].max(), p, None, actuals=panel.iloc[0:0], verbose=False)
    point, _ = serve.predict(panel, fut, p)
    assert np.isfinite(point).all()


def test_streaming_finetune_loads_only_round_series(tiny, tmp_path, monkeypatch):
    from forecast_fm import data as data_mod
    from forecast_fm.finetune import finetune

    raw = _catalog(n=12)
    raw.to_parquet(tmp_path / "s.parquet")
    p = _project(data_path=str(tmp_path / "s.parquet"))
    loaded = []
    real = data_mod.load_panel

    def spy(project, path=None, shard=None, end=None, series=None):
        loaded.append(None if series is None else set(series))
        return real(project, path, shard, end, series)

    monkeypatch.setattr(data_mod, "load_panel", spy)
    exp = ExperimentConfig(name="s", hypothesis="h", model="chronos2",
                           model_params={"model_id": tiny, "device": "cpu", "fine_tune": _ft(2)})
    out = finetune(p, exp, out=tmp_path / "m")  # no panel: streams the catalog
    assert loaded and all(s is not None and len(s) <= 4 for s in loaded)  # never the whole catalog
    assert len(set().union(*loaded)) == 8
    man = json.loads((out / "forecast_fm_model.json").read_text())
    assert man["train_end"] == str(raw["date"].max().date())


def test_backtest_finetune_with_rounds_stays_blind_to_future(monkeypatch, tmp_path):
    from forecast_fm.backtest import backtest

    from .conftest import make_exp
    from .test_train_mix import FitStub

    stub = FitStub()
    monkeypatch.setattr(chronos2, "resolve_device", lambda d: "cpu")
    monkeypatch.setattr(chronos2, "resolve_dtype", lambda *a: None)
    monkeypatch.setitem(chronos2._PIPELINES, ("stub", "cpu", "float32"), stub)
    p = _project(n_folds=1)
    panel = panel_from(_catalog(), p)
    exp = make_exp("chronos2", model_id="stub", cache_dir=str(tmp_path / "a"), fine_tune=_ft(3))
    _, stats = backtest(panel, p, exp)
    first = [t for r in stub.trained for t in r]
    cutoff = pd.Timestamp(stats[0]["cutoff"])
    assert len(stub.trained) == 3 and all(len(r) == 4 for r in stub.trained)
    tampered = panel.copy()
    tampered.loc[tampered[TS] > cutoff, "y"] = 9999.0
    stub.trained.clear()
    exp.model_params["cache_dir"] = str(tmp_path / "b")
    backtest(tampered, p, exp)
    again = [t for r in stub.trained for t in r]
    assert all(np.array_equal(a, b, equal_nan=True) for a, b in zip(first, again, strict=True))
    assert SERIES  # imported for parity with other tests


def test_bench_runs_and_extrapolates(tiny, tmp_path):
    from forecast_fm.bench import bench, render

    raw = _catalog(n=8)
    raw.to_parquet(tmp_path / "s.parquet")
    p = _project(data_path=str(tmp_path / "s.parquet"), n_folds=1, fold_step=28, origin_step_days=7)
    exp = ExperimentConfig(name="b", hypothesis="h", model="chronos2", model_params={
        "model_id": tiny, "device": "cpu",
        "fine_tune": {"num_steps": 1000, "batch_size": 4, "rounds": 2, "train_mix": {"max_series": 4}}})
    res = bench(p, exp, n_series=6, train_steps=6, catalog=3_000_000, memory_gb=8)
    assert res["series_measured"] == 6 and res["forecast"]["series_per_second"] > 0
    assert res["training"]["seconds_per_step"] > 0
    plan = res["plan"]
    assert plan["finetune_hours"] >= 0 and plan["backtest"]["fine_tunes"] == 1
    assert plan["backtest"]["forecast_origins"] > 1 and plan["shards_for_memory_budget"]["shards"] >= 1
    assert "production forecast of 3,000,000 series" in render(res)


@pytest.mark.parametrize("length", [28, 60, 150, 250, 299, 300, 1000])
def test_fixed_window_targets_stay_in_the_window(length):
    """Simulate the trainer's cut rule (uniform in [min_past, len - H], context
    = the C days before): every target starts inside the last W real days, and
    every real day from max(H, len - W) is a possible cut."""
    from forecast_fm.models.chronos2 import fixed_window

    W, C, H = 200, 100, 14
    y = np.arange(length, dtype=np.float32)  # value = real day index
    d = fixed_window({"target": y, "past_covariates": {"t": np.array(["x"] * length)}}, W, C, H)
    t = d["target"]
    assert len(t) >= C + H  # never filtered out by the trainer
    cuts = range(C, len(t) - H + 1)  # min_past = C
    first_real = [t[c] for c in cuts]  # the target's first day, in real days
    assert min(first_real) == max(H, length - W) and max(first_real) == length - H
    for c in (cuts[0], cuts[-1]):
        ctx = t[max(0, c - C):c]
        real = ctx[~np.isnan(ctx)]
        assert len(real) == min(C, int(t[c]))  # real context: all available, up to C
    assert (d["past_covariates"]["t"][np.isnan(t)] == "").all()  # categorical pad token


def test_fixed_window_passes_min_past_to_trainer(monkeypatch, tmp_path):
    from .test_train_mix import FitStub

    stub, seen = FitStub(), {}
    orig = stub.fit

    def fit(inputs, output_dir, **kw):
        seen.update(kw)
        return orig(inputs, output_dir, **kw)

    stub.fit = fit
    monkeypatch.setattr(chronos2, "resolve_device", lambda d: "cpu")
    monkeypatch.setattr(chronos2, "resolve_dtype", lambda *a: None)
    monkeypatch.setitem(chronos2._PIPELINES, ("stub", "cpu", "float32"), stub)
    p = _project()
    panel = panel_from(_catalog(n=4, n_days=400), p)
    m = create_model("chronos2", {"model_id": "stub", "cache_dir": str(tmp_path), "context_length": 60,
                                  "fine_tune": {"num_steps": 1, "train_window_days": 90}})
    m.fit(panel, p, panel[TS].max())
    assert seen["min_past"] == 60
    assert all(len(t) == 150 for t in stub.trained[0])  # last window + context days, not 400
    with pytest.raises(ValueError, match="horizon"):
        create_model("chronos2", {"model_id": "stub", "cache_dir": str(tmp_path),
                                  "fine_tune": {"num_steps": 1, "train_window_days": 7}}).fit(
            panel, p, panel[TS].max())


def test_real_fixed_window_finetune(tiny, tmp_path):
    from forecast_fm.finetune import finetune

    p = _project()
    panel = panel_from(_catalog(n=6, n_days=200), p)
    exp = ExperimentConfig(name="w", hypothesis="h", model="chronos2", model_params={
        "model_id": tiny, "device": "cpu", "context_length": 32,
        "fine_tune": {"num_steps": 3, "batch_size": 4, "train_window_days": 60}})
    out = finetune(p, exp, panel, out=tmp_path / "m")
    man = json.loads((out / "forecast_fm_model.json").read_text())
    assert man["training"]["train_window_days"] == 60
