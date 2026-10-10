"""Regression cases for holdout, scoring, input identity, and safe resume."""

import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from forecast_fm import metrics, runner
from forecast_fm.config import ProjectConfig
from forecast_fm.folds import fold_cutoffs, fold_origins, origin_limit
from forecast_fm.ledger import eval_signature, record, verdict
from forecast_fm.models import REGISTRY
from forecast_fm.models.base import Forecaster
from forecast_fm.provenance import InputSnapshot

from .conftest import make_exp, make_project, make_raw


@pytest.mark.parametrize("step", [None, 7])
@pytest.mark.parametrize("holdout_folds", [0, 1])
def test_automatic_validation_respects_explicit_holdout(step, holdout_folds):
    p = ProjectConfig(horizon=30, horizon_buckets=[30], n_folds=2, fold_step=30,
                      min_train_periods=30, holdout_cutoffs=["2025-10-15"],
                      holdout_folds=holdout_folds, origin_step_days=step)
    first, last = pd.Timestamp("2024-01-01"), pd.Timestamp("2025-12-31")
    cutoffs = fold_cutoffs(first, last, p)
    assert len(cutoffs) == 2
    limit = origin_limit(first, last, p)
    for origins in fold_origins(cutoffs, p, limit):
        assert all(o + pd.Timedelta(days=p.horizon) <= limit for o in origins)
    assert fold_cutoffs(first, last + pd.Timedelta(days=90), p) == cutoffs


@pytest.mark.parametrize("step", [None, 7])
def test_every_origin_checks_boundary_including_single_origin(step):
    p = ProjectConfig(horizon=30, horizon_buckets=[30], origin_step_days=step)
    limit = pd.Timestamp("2025-10-15")
    with pytest.raises(ValueError, match="holdout"):
        fold_origins([limit - pd.Timedelta(days=29)], p, limit)
    assert fold_origins([limit - pd.Timedelta(days=30)], p, limit) == [
        [limit - pd.Timedelta(days=30)]]


def test_invalid_explicit_holdout_does_not_silently_disable_guard():
    p = ProjectConfig(horizon=30, horizon_buckets=[30], cutoffs=["2025-01-01"],
                      holdout_cutoffs=["2025-12-20"])
    first, last = pd.Timestamp("2024-01-01"), pd.Timestamp("2025-12-31")
    for fn in (fold_cutoffs, origin_limit):
        with pytest.raises(ValueError, match="horizon runs past"):
            fn(first, last, p)


def _predictions():
    return pd.DataFrame({"fold": [0, 0, 1, 1, 2, 2], "series_id": ["easy", "hard"] * 3,
                         "horizon": [1, 2] * 3, "y_true": [10., 100.] * 3,
                         "y_pred": [11., 50.] * 3, "stockout": False, "mase_scale": 1.})


def _payload(result):
    return {"overall": result["overall"],
            "tables": {k: v.to_dict("records") for k, v in result["tables"].items()}}


def test_omitting_hard_predictions_cannot_win_or_be_a_reference():
    p = ProjectConfig(horizon=2, horizon_buckets=[1, 2])
    ref = _payload(metrics.score(_predictions(), p))
    candidate = metrics.score(_predictions().assign(y_pred=[10., np.nan] * 3), p)
    assert candidate["overall"]["wape"] == 0
    assert candidate["overall"]["n_missing_pred"] == 3
    assert candidate["tables"]["bucket"].set_index("bucket").loc["h02-2", "n_missing_pred"] == 3
    for new, old in ((_payload(candidate), ref), (ref, _payload(candidate)), (_payload(candidate), None)):
        assert verdict(new, old, p) == ("inconclusive", None)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_all_invalid_predictions_produce_counts_and_null_metrics(bad):
    p = ProjectConfig(horizon=2, horizon_buckets=[2])
    rows = _predictions().assign(y_pred=bad)
    result = metrics.score(rows, p)
    assert result["overall"]["n"] == 0
    assert result["overall"]["n_missing_pred"] == len(rows)
    assert all(result["overall"][m] is None for m in ("wape", "bias", "mase"))
    assert set(result["tables"]["fold"]["fold"]) == {0, 1, 2}
    assert verdict(_payload(result), None, p) == ("inconclusive", None)


@pytest.mark.parametrize("kind", ["empty", "stockout", "no_truth"])
def test_no_eligible_truth_is_a_valid_empty_report(kind):
    p = ProjectConfig(horizon=2, horizon_buckets=[2])
    rows = _predictions()
    if kind == "empty":
        rows = rows.iloc[:0]
    elif kind == "stockout":
        rows = rows.assign(stockout=True)
    else:
        rows = rows.assign(y_true=np.nan)
    result = metrics.score(rows, p)
    assert result["overall"]["n"] == result["overall"]["n_missing_pred"] == 0
    assert result["overall"]["wape"] is None


def test_missing_only_shard_combines_exactly():
    p = ProjectConfig(horizon=2, horizon_buckets=[2])
    rows = _predictions().assign(y_pred=[10., np.nan] * 3)
    one = metrics.score(rows, p)
    combined = metrics.result(metrics.combine([
        metrics.partials(rows[rows.series_id == sid], p) for sid in ("easy", "hard")]), p)
    assert one["overall"] == combined["overall"]
    for name in one["tables"]:
        pd.testing.assert_frame_equal(one["tables"][name], combined["tables"][name])


def test_missing_forecasts_are_tolerated_up_to_max_missing_share():
    """Routine gaps (a new SKU without analogs) must not neuter the ledger; a run
    that skips many rows must not win. The population otherwise matches, since
    the signature already pins data, cutoffs and origins."""
    p = ProjectConfig(horizon=2, horizon_buckets=[2])
    ref = _payload(metrics.score(_predictions(), p))
    rows = pd.concat([_predictions()] * 50, ignore_index=True)  # 300 rows
    rows.loc[0, "y_pred"] = np.nan                                 # 1 of 300 missing: 0.3%
    small_gap = _payload(metrics.score(rows.assign(y_pred=rows["y_pred"] * 0.5), p))
    assert verdict(small_gap, ref, p)[0] in ("improved", "regressed", "inconclusive")
    assert verdict(small_gap, ref, p)[0] != "incomparable"
    strict = ProjectConfig(horizon=2, horizon_buckets=[2], max_missing_share=0.0)
    assert verdict(small_gap, ref, strict) == ("inconclusive", None)
    half = _payload(metrics.score(_predictions().assign(y_pred=[10., np.nan] * 3), p))
    assert verdict(half, ref, p) == ("inconclusive", None)
    assert verdict(ref, half, p) == ("inconclusive", None)


@pytest.fixture
def project(tmp_path):
    path = tmp_path / "sales.csv"
    make_raw(n_series=6, n_days=120).to_csv(path, index=False)
    return make_project(data_path=str(path), min_train_periods=30,
                        covariate_eval_policy={c: "carry_forward"
                                               for c in ("price", "promo_flag", "promo_type")})


@pytest.mark.parametrize("source", ["sales", "plans"])
def test_signature_changes_for_replaced_bytes_even_with_same_mtime(project, tmp_path, source):
    path = tmp_path / "input.csv"
    path.write_text("sku,date,units\nA,2025-01-01,1\n")
    p = replace(project, **({"data_path": str(path)} if source == "sales"
                            else {"planned_covariates_path": str(path)}))
    before = eval_signature(p, [])
    stamp = path.stat()
    path.write_text("sku,date,units\nA,2025-01-01,9\n")
    os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    assert eval_signature(p, []) != before
    path.touch()
    stable = eval_signature(p, [])
    path.touch()
    assert eval_signature(p, []) == stable


@pytest.mark.parametrize("layout", ["glob", "directory", "list"])
def test_fingerprints_cover_all_input_parts_and_membership(project, tmp_path, layout):
    folder = tmp_path / "parts"
    folder.mkdir()
    a, b = folder / "a.parquet", folder / "b.parquet"
    make_raw(n_series=1).to_parquet(a)
    make_raw(n_series=1, seed=1).to_parquet(b)
    source = {"glob": str(folder / "*.parquet"), "directory": str(folder),
              "list": [str(a), str(b)]}[layout]
    p = replace(project, data_path=source)
    before = eval_signature(p, [])
    make_raw(n_series=1, seed=2).to_parquet(b)
    assert eval_signature(p, []) != before
    if layout != "list":
        before = eval_signature(p, [])
        b.unlink()
        assert eval_signature(p, []) != before


def test_data_override_is_the_fingerprinted_input(project, tmp_path):
    override = tmp_path / "override.csv.gz"
    make_raw(n_series=1).to_csv(override, index=False, compression="gzip")
    before = eval_signature(project, [], data=str(override))
    make_raw(seed=9).to_csv(project.data_path, index=False)
    assert eval_signature(project, [], data=str(override)) == before
    make_raw(n_series=1, seed=3).to_csv(override, index=False, compression="gzip")
    assert eval_signature(project, [], data=str(override)) != before


def test_backtest_captures_inputs_before_compute_and_rejects_changes(project, monkeypatch):
    original = runner.backtest

    def changing(*args, **kwargs):
        result = original(*args, **kwargs)
        make_raw(seed=5).to_csv(project.data_path, index=False)
        return result

    monkeypatch.setattr(runner, "backtest", changing)
    with pytest.warns(UserWarning, match="changed during the run"):
        result, _ = runner.run_backtest(project, make_exp())
    assert result["inputs_changed"] == ["sales"]  # the hours of compute are kept ...
    payload = {"overall": result["overall"], "tables": {}, "inputs_changed": result["inputs_changed"]}
    assert verdict(payload, None, project) == ("inconclusive", None)  # ... but never become a verdict


def test_ledger_retains_the_inputs_used_by_backtest(project, tmp_path):
    p = replace(project, reports_dir=str(tmp_path / "reports"))
    result, _ = runner.run_backtest(p, make_exp())
    out = record(make_exp(), p, result, commit=False)
    saved = json.loads((out / "metrics.json").read_text())
    assert saved["input_fingerprints"] == result["input_fingerprints"]
    assert saved["eval_signature"] == eval_signature(p, result["cutoffs"], result["origins"])


@pytest.mark.parametrize("change", ["model", "params", "horizon", "quantiles", "sales", "plans", "new"])
def test_forecast_resume_rejects_changed_recipe_or_inputs(project, tmp_path, change):
    p, exp = project, make_exp()
    plans, new = tmp_path / "plans.csv", tmp_path / "new.csv"
    if change == "plans":
        plans.write_text("as_of,date,sku,price\n2024-04-29,2024-04-30,S0,10\n")
        p = replace(p, planned_covariates_path=str(plans))
    if change == "new":
        new.write_text("sku,launch_date,category\n")
        p = replace(p, cold_start={"new_series_path": str(new)})
    out = runner.run_forecast(p, exp, shards=2, only=[0], out_dir=tmp_path / "forecast")
    old_run = (out / "_run.json").read_bytes()
    old_part = (out / "part-00000.parquet").read_bytes()
    if change == "model":
        exp = make_exp("naive")
    elif change == "params":
        exp = make_exp("croston", alpha=0.25)
    elif change == "horizon":
        p = replace(p, horizon=7, horizon_buckets=[7])
    elif change == "quantiles":
        p = replace(p, quantiles=[0.2, 0.5, 0.8])
    elif change == "sales":
        raw = pd.read_csv(p.data_path)
        raw["units"] += 1
        raw.to_csv(p.data_path, index=False)
    elif change == "plans":
        plans.write_text(plans.read_text().replace(",10\n", ",20\n"))
    else:
        new.write_text("sku,launch_date,category\nNEW,2024-05-01,A\n")
    with pytest.raises(ValueError, match="different or unverified"):
        runner.run_forecast(p, exp, shards=2, only=[1], out_dir=out)
    assert (out / "_run.json").read_bytes() == old_run
    assert (out / "part-00000.parquet").read_bytes() == old_part
    assert not (out / "_manifest.json").exists()


class CheckpointSpy(Forecaster):
    PARAM_KEYS = frozenset({"model_id"})

    def predict(self, history, future, project):
        return np.ones(len(future)), None


def test_resume_hashes_local_checkpoint_contents(project, tmp_path, monkeypatch):
    monkeypatch.setitem(REGISTRY, "checkpoint_spy", CheckpointSpy)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    weights = checkpoint / "weights.bin"
    weights.write_bytes(b"version1")
    exp = make_exp("checkpoint_spy", model_id=str(checkpoint))
    out = runner.run_forecast(project, exp, out_dir=tmp_path / "forecast")
    weights.write_bytes(b"version2")
    with pytest.raises(ValueError, match="different or unverified"):
        runner.run_forecast(project, exp, out_dir=out)


def test_force_changed_recipe_requires_full_restart(project, tmp_path):
    out = runner.run_forecast(project, make_exp(), shards=2, out_dir=tmp_path / "forecast")
    new = replace(project, horizon=7, horizon_buckets=[7])
    with pytest.raises(ValueError, match="without --shard"):
        runner.run_forecast(new, make_exp(), shards=2, only=[0], out_dir=out, force=True)
    runner.run_forecast(new, make_exp(), shards=2, out_dir=out, force=True)
    forecasts = pd.read_parquet(out)
    assert len(forecasts) == 6 * 7
    assert forecasts.horizon.max() == 7
    manifest = json.loads((out / "_manifest.json").read_text())
    assert manifest["horizon"] == 7 and manifest["run_identity"] and manifest["input_fingerprints"]


@pytest.mark.parametrize("legacy", [False, True])
def test_unverified_existing_parts_require_force(project, tmp_path, legacy):
    out = runner.run_forecast(project, make_exp(), out_dir=tmp_path / "forecast")
    if legacy:
        (out / "_run.json").write_text(json.dumps({"shards": 1, "origin": "2024-04-29"}))
    else:
        (out / "_run.json").unlink()
    with pytest.raises(ValueError, match="unverified|without a verified"):
        runner.run_forecast(project, make_exp(), out_dir=out)
    runner.run_forecast(project, make_exp(), out_dir=out, force=True)
    assert json.loads((out / "_run.json").read_text())["version"] == 2


def test_force_refuses_active_shard_workers(tmp_path):
    with runner._run_lock(tmp_path, force=False), runner._run_lock(tmp_path, force=False):
        with pytest.raises(ValueError, match="in use"), runner._run_lock(tmp_path, force=True):
            pytest.fail("forced restart acquired an active directory")
    with runner._run_lock(tmp_path, force=True):
        pass


def test_mutating_inputs_never_publishes_a_completed_forecast(project, tmp_path, monkeypatch):
    original = runner.forecast_at

    def changing(*args, **kwargs):
        result = original(*args, **kwargs)
        make_raw(seed=7).to_csv(project.data_path, index=False)
        return result

    monkeypatch.setattr(runner, "forecast_at", changing)
    out = tmp_path / "forecast"
    with pytest.raises(ValueError, match="input changed during run"):
        runner.run_forecast(project, make_exp(), out_dir=out)
    assert not list(out.glob("part-*.parquet"))
    assert not (out / "_manifest.json").exists()


def test_input_snapshot_detects_added_files(tmp_path):
    (tmp_path / "a.csv").write_text("x\n1\n")
    snapshot = InputSnapshot({"sales": str(tmp_path / "*.csv")})
    (tmp_path / "b.csv").write_text("x\n2\n")
    with pytest.raises(ValueError, match="input changed"):
        snapshot.check()


def test_parallel_shards_and_duplicate_workers_publish_one_consistent_run(project, tmp_path):
    out = tmp_path / "parallel"

    def worker(shards):
        return runner.run_forecast(project, make_exp(), shards=2, only=shards, out_dir=out)

    with ThreadPoolExecutor(max_workers=3) as pool:
        assert list(pool.map(worker, ([0], [1], [0, 1]))) == [out] * 3
    frame = pd.read_parquet(out)
    assert len(frame) == 6 * project.horizon
    assert not frame.duplicated(["series_id", "ts"]).any()
    assert len(json.loads((out / "_manifest.json").read_text())["parts"]) == 2


def test_custom_holdout_targets_and_past_covariates_never_affect_validation(project):
    p = replace(project, holdout_cutoffs=["2024-03-25"])
    before, predictions = runner.run_backtest(p, make_exp())
    raw = pd.read_csv(p.data_path)
    future = pd.to_datetime(raw["date"]) > pd.Timestamp(p.holdout_cutoffs[0])
    raw.loc[future, ["units", "sessions"]] = 99999
    raw.to_csv(p.data_path, index=False)
    after, again = runner.run_backtest(p, make_exp())
    assert predictions["ts"].max() <= pd.Timestamp(p.holdout_cutoffs[0])
    pd.testing.assert_frame_equal(predictions, again)
    assert before["input_fingerprints"] != after["input_fingerprints"]


def test_unversioned_reference_is_incomparable():
    p = ProjectConfig(horizon=2, horizon_buckets=[2])
    ref = _payload(metrics.score(_predictions(), p))
    new = {**ref, "eval_signature": "content-versioned"}
    assert verdict(new, ref, p) == ("incomparable", None)
