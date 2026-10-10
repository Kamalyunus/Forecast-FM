"""Regression tests for the second fresh-eyes review."""

import json
import subprocess
import warnings

import numpy as np
import pandas as pd
import pytest
import yaml

from forecast_fm import metrics
from forecast_fm.backtest import backtest, forecast_at
from forecast_fm.data import SERIES, TS, Y, build_panel, demand_classes, make_series_id, prepare_raw
from forecast_fm.folds import fold_cutoffs, fold_origins, origin_limit
from forecast_fm.ledger import eval_signature, verdict
from forecast_fm.runner import run_backtest, run_forecast
from forecast_fm.sample import allocate

from .conftest import make_exp, make_project, make_raw, panel_from


def _flat(**kw):
    base = dict(horizon=14, horizon_buckets=[7, 14], n_folds=2, fold_step=14, min_train_periods=60,
                covariate_eval_policy={c: "carry_forward" for c in ("price", "promo_flag", "promo_type")})
    base.update(kw)
    return make_project(**base)


# --- forecast resumption ------------------------------------------------------

def test_forecast_dir_refuses_a_different_shard_count(tmp_path):
    raw = make_raw(n_series=12, n_days=120)
    raw.to_parquet(tmp_path / "s.parquet")
    p = _flat(data_path=str(tmp_path / "s.parquet"), min_history_days=1)
    out = run_forecast(p, make_exp(), shards=4, only=[0], out_dir=tmp_path / "fc")
    with pytest.raises(ValueError, match="started with --shards 4"):
        run_forecast(p, make_exp(), shards=3, out_dir=out)
    with pytest.raises(ValueError, match="out of range"):
        run_forecast(p, make_exp(), shards=1, only=[3], out_dir=tmp_path / "fc2")
    run_forecast(p, make_exp(), shards=3, out_dir=out, force=True)  # a restart wipes the old parts
    man = json.loads((out / "_manifest.json").read_text())
    assert man["shards"] == 3 and len(man["parts"]) == 3
    assert pd.read_parquet(out)[SERIES].nunique() == 12


# --- folds -------------------------------------------------------------------------

def test_explicit_cutoffs_cannot_score_the_rule_based_holdout():
    p = _flat(cutoffs=["2024-06-10"], holdout_folds=1)
    first, last = pd.Timestamp("2024-01-01"), pd.Timestamp("2024-06-30")
    with pytest.raises(ValueError, match="overlaps the holdout"):
        fold_cutoffs(first, last, p)
    ok = _flat(cutoffs=["2024-06-01"], holdout_folds=1)
    assert fold_cutoffs(first, last, ok) == [pd.Timestamp("2024-06-01")]
    assert fold_cutoffs(first, last, _flat(holdout_folds=0), holdout=True) == []
    with pytest.raises(ValueError, match="holdout"):
        fold_origins([pd.Timestamp("2024-06-10")], _flat(origin_step_days=7), pd.Timestamp("2024-06-16"))
    assert origin_limit(first, last, _flat(holdout_folds=0)) == last


# --- scoring --------------------------------------------------------------------------

def test_cold_start_none_counts_missing_predictions():
    raw = make_raw(n_series=6, n_days=200)
    late = raw[raw["sku"] == "S0"].copy()
    late["sku"] = "LATE"
    late = late[late["date"] > "2024-06-20"]  # launches inside the last fold's horizon
    panel = panel_from(pd.concat([raw, late]), _flat())
    for method in ("launch_profile", "none"):
        p = _flat(cold_start={"method": method, "min_analogs": 1}, min_history_days=1)
        preds, _ = backtest(panel, p, make_exp("seasonal_naive"))
        res = metrics.score(preds, p)
        rows_late = preds[preds[SERIES] == "LATE"]
        assert len(rows_late)  # present either way
        if method == "none":
            assert rows_late["y_pred"].isna().all()
            assert res["overall"]["n_missing_pred"] == int(rows_late["y_true"].notna().sum())


def test_quantile_metrics_only_over_rows_that_have_quantiles():
    p = _flat()
    rows = pd.DataFrame({
        SERIES: ["a"] * 4, "fold": [0, 0, 1, 1], "horizon": [1, 2, 1, 2], "y_true": [10.0, 10, 10, 10],
        "y_pred": [10.0, 10, 10, 10], "stockout": False, "mase_scale": 1.0,
        "q_0.1": [8.0, 8, np.nan, np.nan], "q_0.9": [12.0, 12, np.nan, np.nan]})
    t = metrics.score(rows, p)["tables"]["fold"].set_index("fold")
    assert t.loc[0, "wql"] == pytest.approx(0.04) and t.loc[0, "cov0.9"] == 1.0  # 2*(0.2+0.2)/2 / 10
    assert pd.isna(t.loc[1, "wql"]) and pd.isna(t.loc[1, "cov0.9"])  # not 0.0 / 0 %
    assert pd.isna(metrics.score(rows, p)["overall"]["wql"])  # mixed -> unknown, not a blend


def test_sharded_cold_start_equals_unsharded_when_launches_are_shard_local(tmp_path):
    """A category whose SKUs all launch after the data starts must keep its first launch
    as an analog in every shard (the data start is global, not the shard's)."""
    frames = []
    for i in range(12):
        r = make_raw(n_series=1, n_days=150, seed=i)
        r["sku"] = f"S{i:02d}"
        r["category"] = "late" if i >= 6 else "early"
        if i >= 6:
            r = r[r["date"] >= pd.Timestamp("2024-01-01") + pd.Timedelta(days=10 + 5 * (i - 6))]
        frames.append(r)
    raw = pd.concat(frames, ignore_index=True)
    raw.to_parquet(tmp_path / "s.parquet")
    p = _flat(data_path=str(tmp_path / "s.parquet"), static_cols=["category"], shard_by=["category"],
              min_history_days=40, cold_start={"profile_cols": ["category"], "min_analogs": 1},
              n_folds=1, fold_step=14, min_train_periods=30)
    one, _ = run_backtest(p, make_exp(), shards=1)
    two, _ = run_backtest(p, make_exp(), shards=2)
    for k in ("wape", "bias", "mase"):
        assert one["overall"][k] == pytest.approx(two["overall"][k], nan_ok=True)


# --- data ---------------------------------------------------------------------------

def test_timezone_aware_dates_and_integer_ids_and_nat():
    raw = make_raw(n_series=2, n_days=40)
    tz = raw.copy()
    tz["date"] = pd.to_datetime(tz["date"]).dt.tz_localize("UTC")
    p = _flat(start_date="2024-01-05")
    out = prepare_raw(tz, p, "t")
    assert out[TS].dt.tz is None and out[TS].min() == pd.Timestamp("2024-01-05")
    ids = make_series_id(pd.DataFrame({"sku": [123.0, 456.0, np.nan]}), ["sku"])
    assert list(ids[:2]) == ["123", "456"]
    bad = raw.copy()
    bad.loc[3, "date"] = pd.NaT
    with pytest.raises(ValueError, match="have no date"):
        prepare_raw(bad, _flat(), "t")


def test_missing_static_and_label_are_tokens_not_nan():
    raw = make_raw(n_series=3, n_days=40)
    raw["demand_label"] = np.where(raw["sku"] == "S1", None, "BAU")
    raw["category"] = np.where(raw["sku"] == "S2", None, raw["category"])
    p = _flat(static_cols=["category"], demand_label_col="demand_label")
    panel = build_panel(prepare_raw(raw, p, "t"), p)
    assert not panel["category"].isna().any() and "" in panel["category"].cat.categories
    cls = demand_classes(panel, p)
    assert cls["S1"] == "unlabeled" and cls["S0"] == "BAU"


def test_stray_string_in_numeric_covariate_is_coerced_with_a_warning():
    raw = make_raw(n_series=2, n_days=120)
    raw["price"] = raw["price"].astype(object)
    raw.loc[5, "price"] = "N/A"
    with pytest.warns(UserWarning, match="numeric except for 1 value"):
        out = prepare_raw(raw, _flat(), "t")
    assert pd.api.types.is_float_dtype(out["price"]) and out["price"].isna().sum() == 1


def test_sample_floors_never_exceed_n():
    sizes = pd.Series(1000, index=[f"s{i}" for i in range(100)])
    q = allocate(sizes, 500, 20)
    assert int(q.sum()) == 500 and (q >= 0).all()
    assert int(allocate(sizes, 5000, 20).sum()) == 5000


# --- ledger ---------------------------------------------------------------------------

def test_signature_covers_data_override_and_origins_but_not_warning_thresholds(tmp_path):
    data, other = str(tmp_path / "sales.csv"), str(tmp_path / "other.csv")
    make_raw().to_csv(data, index=False)
    make_raw(seed=1).to_csv(other, index=False)
    p = _flat(data_path=data)
    cut = [pd.Timestamp("2024-06-01")]
    base = eval_signature(p, cut)
    assert eval_signature(p, cut, data=other) != base
    assert eval_signature(p, cut, origins=[[cut[0], cut[0] + pd.Timedelta(days=7)]]) != base
    assert eval_signature(_flat(data_path=data, min_plan_coverage=0.5, plan_max_age_days=3), cut) == base


def test_verdict_ignores_folds_without_a_finite_value():
    p = _flat()
    ref = {"overall": {"wape": 1.0}, "tables": {"fold": [{"fold": k, "wape": 1.0} for k in range(3)]}}
    new = {"overall": {"wape": 1.05}, "tables": {"fold": [{"fold": 0, "wape": 0.9},
                                                          {"fold": 1, "wape": float("nan")},
                                                          {"fold": 2, "wape": float("nan")}]}}
    assert verdict(new, ref, p)[0] == "inconclusive"  # one better fold, two unknown: not "regressed"


def test_ledgered_run_reserves_its_id_and_a_failed_commit_raises(tmp_path, monkeypatch, capsys):
    from forecast_fm import ledger
    from forecast_fm.cli import main

    monkeypatch.chdir(tmp_path)
    subprocess.run(["git", "init", "-q"], check=True)
    make_raw(n_series=2).to_parquet("sales.parquet")
    d = dict(vars(make_project()))
    d["data_path"] = "sales.parquet"
    (tmp_path / "proj.yaml").write_text(yaml.safe_dump(d))
    (tmp_path / "e.yaml").write_text(yaml.safe_dump({"name": "sn", "hypothesis": "h",
                                                     "model": "seasonal_naive"}))
    subprocess.run(["git", "add", "."], check=True)
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "x"], check=True)
    # a stale reservation from a run that is still going: the new run takes the next id
    (tmp_path / "experiments" / "exp001-other").mkdir(parents=True)
    (tmp_path / "experiments" / "exp001-other" / ledger.RUNNING).write_text("x")
    for k, v in {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
                 "GIT_COMMITTER_EMAIL": "t@t"}.items():
        monkeypatch.setenv(k, v)
    main(["-p", "proj.yaml", "run", "e.yaml"])
    out = capsys.readouterr().out
    assert "[run] ledger entry exp002-sn reserved" in out and "committed" in out
    assert (tmp_path / "experiments" / "exp002-sn" / "metrics.json").exists()
    assert not (tmp_path / "experiments" / "exp002-sn" / ledger.RUNNING).exists()
    assert ledger.has_experiment("exp002") and not ledger.has_experiment("exp001")
    # git failing (a hook, a lock, no identity) must not report a ledgered run
    real_git = ledger._git

    def failing_git(*a, check=False):
        if a[0] == "commit":
            raise RuntimeError("git commit failed: boom")
        return real_git(*a, check=check)

    monkeypatch.setattr(ledger, "_git", failing_git)
    with pytest.raises(RuntimeError, match="git commit failed"):
        main(["-p", "proj.yaml", "run", "e.yaml"])


def test_debug_run_outside_git_works_and_ledgered_run_refuses(tmp_path, monkeypatch):
    from forecast_fm.cli import main

    monkeypatch.chdir(tmp_path)  # tmp_path is not a git checkout
    make_raw(n_series=2).to_parquet("sales.parquet")
    d = dict(vars(make_project()))
    d["data_path"] = "sales.parquet"
    (tmp_path / "proj.yaml").write_text(yaml.safe_dump(d))
    (tmp_path / "e.yaml").write_text(yaml.safe_dump({"name": "sn", "hypothesis": "h",
                                                     "model": "seasonal_naive"}))
    main(["-p", "proj.yaml", "run", "e.yaml", "--no-commit"])
    with pytest.raises(RuntimeError, match="not inside a git checkout"):
        main(["-p", "proj.yaml", "run", "e.yaml"])


# --- config -----------------------------------------------------------------------------

def test_late_failing_settings_are_rejected_up_front():
    with pytest.raises(ValueError, match="horizon_buckets"):
        make_project(horizon=14, horizon_buckets=[0, 7, 14])
    with pytest.raises(ValueError, match="horizon_buckets"):
        make_project(horizon=14, horizon_buckets=[7, 7, 14])
    with pytest.raises(ValueError, match="season_length"):
        make_project(season_length=0)
    with pytest.raises(ValueError, match="profile_cols"):
        make_project(cold_start={"profile_cols": ["categry"]})


# --- cold start with no analogs ------------------------------------------------------------

def test_no_analog_pool_gives_missing_not_zero():
    raw = make_raw(n_series=3, n_days=60)  # every series starts on day 0: no observed launch
    p = _flat(min_history_days=1, n_folds=1, min_train_periods=20)
    panel = panel_from(raw, p)
    new = pd.DataFrame({SERIES: ["NEW"], "launch_date": [pd.Timestamp("2024-03-01")]})
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        pred, st = forecast_at(panel, panel[TS].max(), p, make_exp(), None, production=True, new_series=new)
    assert any("no analog launch profile" in str(x.message) for x in w)
    assert pred.loc[pred[SERIES] == "NEW", "y_pred"].isna().all()
    assert st["cold_start"]["n_no_profile"] == 1
    assert Y  # imported for parity


def test_chronos2_stale_context_is_a_missing_forecast(monkeypatch, tmp_path):
    from forecast_fm.models import chronos2, create_model
    from forecast_fm.plans import future_frame

    from .test_train_mix import FitStub

    stub = FitStub()
    monkeypatch.setattr(chronos2, "resolve_device", lambda d: "cpu")
    monkeypatch.setattr(chronos2, "resolve_dtype", lambda *a: None)
    monkeypatch.setitem(chronos2._PIPELINES, ("stub", "cpu", "float32"), stub)
    raw = make_raw(n_series=3, n_days=200)
    raw = raw[~((raw["sku"] == "S1") & (raw["date"] > "2024-03-01"))]  # S1 discontinued early
    p = _flat(grid_end="series", min_history_days=1)
    panel = panel_from(raw, p)
    cutoff = panel[TS].max()
    m = create_model("chronos2", {"model_id": "stub", "context_length": 30, "cache_dir": str(tmp_path)})
    m.fit(panel, p, cutoff)
    fut = future_frame(panel, cutoff, p, None, verbose=False)
    point, _ = m.predict(panel, fut, p)
    out = fut.assign(y_pred=point)
    assert out.loc[out[SERIES].astype(str) == "S1", "y_pred"].isna().all()
    assert out.loc[out[SERIES].astype(str) != "S1", "y_pred"].notna().all()
    assert m.stats["n_no_context"] == 1
