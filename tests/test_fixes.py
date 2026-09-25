"""Regression tests for the review findings: each test fails on the old code."""

import json
import os
import subprocess

import numpy as np
import pandas as pd
import pytest

from forecast_fm.config import ExperimentConfig
from forecast_fm.data import SERIES, TS, Y, build_panel, prepare_raw
from forecast_fm.metrics import score
from forecast_fm.models import chronos2, create_model
from forecast_fm.models.chronos2 import MANIFEST, TrainedOnFutureError
from forecast_fm.plans import HORIZON, future_frame, load_plans

from .conftest import make_project, make_raw, panel_from

# 1. the saved-checkpoint guard ------------------------------------------------------

def test_guard_reads_manifest_one_level_up(tmp_path):
    out = tmp_path / "m"
    (out / "finetuned-ckpt").mkdir(parents=True)
    (out / MANIFEST).write_text(json.dumps({"train_end": "2024-06-01"}))
    p = make_project()
    panel = panel_from(make_raw(), p)
    for model_id in (out, out / "finetuned-ckpt"):  # pointing at the weights no longer bypasses it
        with pytest.raises(TrainedOnFutureError, match="2024-06-01"):
            create_model("chronos2", {"model_id": str(model_id)}).fit(panel, p, pd.Timestamp("2024-05-01"))


def test_unverified_finetuned_weights_refused(tmp_path):
    ckpt = tmp_path / "somewhere" / "finetuned-ckpt"
    ckpt.mkdir(parents=True)
    p = make_project()
    panel = panel_from(make_raw(), p)
    with pytest.raises(TrainedOnFutureError, match="no forecast_fm_model.json"):
        create_model("chronos2", {"model_id": str(ckpt)}).fit(panel, p, pd.Timestamp("2024-05-01"))
    adapter = tmp_path / "lora"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}")
    with pytest.raises(TrainedOnFutureError):
        create_model("chronos2", {"model_id": str(adapter)}).fit(panel, p, pd.Timestamp("2024-05-01"))


def test_backtest_cache_checkpoint_is_guarded(tiny, tmp_path):
    p = make_project()
    panel = panel_from(make_raw(n_series=3), p)
    c1 = panel[TS].max() - pd.Timedelta(days=30)
    params = {"model_id": tiny, "device": "cpu", "cache_dir": str(tmp_path / "ft"),
              "fine_tune": {"num_steps": 2, "batch_size": 4}}
    create_model("chronos2", params).fit(panel[panel[TS] <= c1], p, c1)
    cache = next((tmp_path / "ft").iterdir())
    assert json.loads((cache / MANIFEST).read_text())["train_end"] == str(c1.date())
    with pytest.raises(TrainedOnFutureError):
        create_model("chronos2", {"model_id": str(cache / "finetuned-ckpt"), "device": "cpu"}).fit(
            panel, p, c1 - pd.Timedelta(days=7))


# 2. Croston keeps stockouts out ---------------------------------------------------------

def test_croston_ignores_stockout_days():
    days = pd.date_range("2024-01-01", periods=40)
    y = np.zeros(40)
    y[[0, 20, 39]] = 5
    raw = pd.DataFrame({"date": days, "sku": "A", "units": y, "in_stock": 1, "sessions": 1.0,
                        "price": 1.0, "promo_flag": 0, "promo_type": "none", "category": "A",
                        "demand_label": "x"})
    raw.loc[1:19, "in_stock"] = 0
    p = make_project(horizon=7, horizon_buckets=[7])
    panel = panel_from(raw, p)
    fut = future_frame(panel, panel[TS].max(), p, None, actuals=panel.iloc[0:0], verbose=False)
    rate = create_model("croston", {"alpha": 0.5, "window": 40}).predict(panel, fut, p)[0][0]
    # 3 sales in the 21 in-stock days: ~5/10 per day; counting the stockout
    # gap as zero demand would give ~5/19
    assert rate > 0.3


# 4. no silent `actual` default ---------------------------------------------------------------

def test_known_covariate_needs_a_policy():
    from forecast_fm.config import ProjectConfig

    with pytest.raises(ValueError, match="need a covariate_eval_policy"):
        ProjectConfig(known_covariate_cols=["price"], covariate_eval_policy={})


# 5. duplicate plan rows ---------------------------------------------------------------------------

def test_duplicate_plan_rows(tmp_path):
    rows = pd.DataFrame({"as_of": "2024-03-31", "date": ["2024-04-01", "2024-04-01"], "sku": "S0",
                         "price": [1.0, 2.0]})
    rows.to_csv(tmp_path / "p.csv", index=False)
    p = make_project(covariate_eval_policy={"price": "plan"}, planned_covariates_path=str(tmp_path / "p.csv"))
    with pytest.raises(ValueError, match="duplicate"):
        load_plans(p)
    p2 = make_project(covariate_eval_policy={"price": "plan"}, duplicates="last",
                      planned_covariates_path=str(tmp_path / "p.csv"))
    assert load_plans(p2)["price"].tolist() == [2.0]


# 6. ledger ---------------------------------------------------------------------------------------------

def test_verdict_refuses_incomparable_runs():
    from forecast_fm.ledger import eval_signature, verdict

    p = make_project()
    cut = [pd.Timestamp("2024-05-01")]
    ref = {"overall": {"wape": 1.0}, "tables": {"fold": [{"fold": 0, "wape": 1.0}]},
           "eval_signature": eval_signature(p, cut)}
    new = {"overall": {"wape": 0.5}, "tables": {"fold": [{"fold": 0, "wape": 0.5}]},
           "eval_signature": eval_signature(make_project(horizon=7, horizon_buckets=[7]), cut)}
    assert verdict(new, ref, p)[0] == "incomparable"
    new["eval_signature"] = eval_signature(make_project(model_defaults={"x": {}}), cut)
    assert verdict(new, ref, p)[0] == "improved"  # model settings are what experiments change


def test_run_checks_git_before_backtest_and_records_invocation(tmp_path, monkeypatch):
    from forecast_fm import runner
    from forecast_fm.cli import main

    monkeypatch.chdir(tmp_path)
    for k, v in {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
                 "GIT_COMMITTER_EMAIL": "t@t"}.items():
        monkeypatch.setenv(k, v)
    subprocess.run(["git", "init", "-q"], check=True)
    make_raw(n_series=2).to_parquet("sales.parquet")
    d = dict(vars(make_project()))
    d["data_path"] = "sales.parquet"
    import yaml

    (tmp_path / "proj.yaml").write_text(yaml.safe_dump(d))
    (tmp_path / "e.yaml").write_text(yaml.safe_dump({"name": "sn", "hypothesis": "h",
                                                     "model": "seasonal_naive"}))
    called = []
    monkeypatch.setattr(runner, "backtest", lambda *a, **k: called.append(1))
    with pytest.raises(RuntimeError, match="uncommitted"):  # proj.yaml untracked = dirty
        main(["-p", "proj.yaml", "run", "e.yaml"])
    assert not called  # refused before any compute
    monkeypatch.undo()
    monkeypatch.chdir(tmp_path)
    subprocess.run(["git", "add", "."], check=True)
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "x"], check=True)
    main(["-p", "proj.yaml", "--set", "season_length=14", "run", "e.yaml"])
    m = json.loads(next((tmp_path / "experiments").glob("exp001*/metrics.json")).read_text())
    assert m["invocation"]["overrides"] == ["season_length=14"]
    assert m["project"]["season_length"] == 14 and m["cutoffs"] and m["eval_signature"]


def test_record_survives_missing_primary_metric(tmp_path, monkeypatch):
    from forecast_fm.ledger import record

    monkeypatch.chdir(tmp_path)
    res = {"overall": {"wape": None}, "tables": {"fold": pd.DataFrame({"fold": [0], "wape": [None]})}}
    record(ExperimentConfig(name="x", hypothesis="h", model="naive"), make_project(), res, commit=False)


# 7. fine-tune sees full history -------------------------------------------------------------------

def test_finetune_inputs_are_full_series_and_context_length_goes_to_trainer(monkeypatch, tmp_path):
    from .test_train_mix import FitStub

    stub = FitStub()
    seen = {}
    orig = stub.fit

    def fit(inputs, output_dir, **kw):
        seen.update(kw)
        return orig(inputs, output_dir, **kw)

    stub.fit = fit
    monkeypatch.setattr(chronos2, "resolve_device", lambda d: "cpu")
    monkeypatch.setattr(chronos2, "resolve_dtype", lambda *a: None)
    monkeypatch.setitem(chronos2._PIPELINES, ("stub", "cpu", "float32"), stub)
    p = make_project()
    panel = panel_from(make_raw(n_series=2), p)
    m = create_model("chronos2", {"model_id": "stub", "cache_dir": str(tmp_path),
                                  "fine_tune": {"num_steps": 1, "context_length": 50}})
    m.fit(panel, p, panel[TS].max())
    assert [len(t) for t in stub.trained[0]] == [200, 200]  # not cut to 50
    assert seen["context_length"] == 50


# 8. MPS fallback is set before torch loads ---------------------------------------------------------

def test_mps_fallback_env_set_on_import():
    import forecast_fm  # noqa: F401

    assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1"


# 9. categoricals as codes ------------------------------------------------------------------------------

def test_categorical_covariates_stored_as_codes():
    p = make_project()
    panel = panel_from(make_raw(n_series=2, drop_frac=0.2), p)
    assert isinstance(panel["promo_type"].dtype, pd.CategoricalDtype)
    assert set(panel["promo_type"].cat.categories) <= {"none", "pct_off", ""}


# 10. contexts end at the cutoff ------------------------------------------------------------------------

def test_series_ending_early_is_padded_to_the_cutoff(monkeypatch):
    from .test_chronos2 import StubPipeline

    stub = StubPipeline()
    monkeypatch.setattr(chronos2, "resolve_device", lambda d: "cpu")
    monkeypatch.setitem(chronos2._PIPELINES, ("stub", "cpu", "float32"), stub)
    raw = make_raw(n_series=2, n_days=100)
    raw = raw[~((raw["sku"] == "S1") & (raw["date"] > "2024-03-31"))]  # S1 stops 9 days early
    p = make_project(grid_end="series")
    panel = panel_from(raw, p)
    cutoff = panel[TS].max()
    m = create_model("chronos2", {"model_id": "stub"})
    m.fit(panel, p, cutoff)
    fut = future_frame(panel, cutoff, p, None, actuals=panel.iloc[0:0], verbose=False)
    m.predict(panel, fut, p)
    s1 = stub.calls[0]["inputs"][1]
    assert len(s1["target"]) == 100 and np.isnan(s1["target"][-9:]).all()
    assert (s1["past_covariates"]["promo_type"][-9:] == "").all()


# minor ----------------------------------------------------------------------------------------------------

def test_missing_predictions_are_reported_not_flattering():
    p = make_project(horizon=2, horizon_buckets=[2])
    preds = pd.DataFrame({"fold": 0, SERIES: "a", HORIZON: [1, 2], "y_true": [10.0, 10.0],
                          "y_pred": [10.0, np.nan], "stockout": False, "mase_scale": 1.0})
    o = score(preds, p)["overall"]
    assert o["n_missing_pred"] == 1 and o["n"] == 1 and o["wape"] == 0


def test_duplicate_sum_keeps_all_missing_target_missing():
    raw = make_raw(n_series=1, n_days=5)
    raw = pd.concat([raw, raw.iloc[[2]]])
    raw.loc[raw["date"] == raw["date"].iloc[2], "units"] = np.nan
    p = make_project(duplicates="sum", missing_target="nan")
    panel = build_panel(prepare_raw(raw, p), p)
    assert np.isnan(panel[Y].iloc[2])


def test_sampler_volume_until(tmp_path):
    from forecast_fm.sample import series_stats

    raw = make_raw(n_series=2, n_days=20)
    raw.loc[(raw["sku"] == "S0") & (raw["date"] > "2024-01-10"), "units"] = 1000.0
    raw.to_parquet(tmp_path / "r.parquet")
    st = series_stats(tmp_path / "r.parquet", ["sku"], "units", None, ts_col="date", until="2024-01-10")
    assert st.loc["S0", "total"] < 1000 and st.loc["S0", "n"] == 10
