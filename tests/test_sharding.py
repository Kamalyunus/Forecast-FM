"""Sharded backtests and forecasts equal unsharded ones."""

import json

import numpy as np
import pandas as pd
import pytest

from forecast_fm.data import SERIES, date_span, load_raw, shard_of
from forecast_fm.runner import run_backtest, run_forecast

from .conftest import make_exp, make_project, make_raw

# no plan file in these tests: production holds known covariates flat
CF = {"price": "carry_forward", "promo_flag": "carry_forward", "promo_type": "carry_forward"}


@pytest.fixture
def data(tmp_path):
    raw = make_raw(n_series=12, n_days=200)
    raw.loc[raw["sku"] == "S11", "date"] += pd.Timedelta(days=150)  # launches late: a new series
    raw = raw[raw["date"] <= raw["date"].min() + pd.Timedelta(days=199)]
    path = tmp_path / "sales.parquet"
    raw.to_parquet(path, index=False)
    return str(path)


def test_shards_partition_the_series(data):
    p = make_project(data_path=data)
    ids = [set(load_raw(p, shard=(i, 3))[SERIES]) for i in range(3)]
    assert set().union(*ids) == set(load_raw(p)[SERIES])
    assert sum(len(s) for s in ids) == len(set().union(*ids))  # disjoint


def test_shard_by_keeps_groups_together(data):
    p = make_project(data_path=data, shard_by=["category"])
    for i in range(3):
        raw = load_raw(p, shard=(i, 3))
        if len(raw):
            assert raw["category"].nunique() == 1


@pytest.mark.parametrize("model", ["seasonal_naive", "croston"])
def test_sharded_backtest_metrics_equal_unsharded(data, model):
    p = make_project(data_path=data, slice_cols=["category"], cold_start={"min_analogs": 1})
    one, _ = run_backtest(p, make_exp(model), shards=1)
    three, _ = run_backtest(p, make_exp(model), shards=3)
    assert one["cutoffs"] == three["cutoffs"]
    for k, v in one["overall"].items():
        assert three["overall"][k] == pytest.approx(v, nan_ok=True), k
    for name, t in one["tables"].items():
        pd.testing.assert_frame_equal(t.sort_values(list(t.columns[:1])).reset_index(drop=True),
                                      three["tables"][name].sort_values(list(t.columns[:1]))
                                      .reset_index(drop=True), check_dtype=False, rtol=1e-6)


def test_sharded_forecast_equals_unsharded_and_resumes(data, tmp_path, capsys):
    p = make_project(data_path=data, cold_start={"min_analogs": 1}, covariate_eval_policy=CF)
    a = pd.read_parquet(run_forecast(p, make_exp(), shards=1, out_dir=tmp_path / "one"))
    out = run_forecast(p, make_exp(), shards=3, out_dir=tmp_path / "three")
    b = pd.read_parquet(out)
    key = [SERIES, "ts"]
    a, b = a.sort_values(key).reset_index(drop=True), b.sort_values(key).reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b, check_dtype=False, check_categorical=False)
    assert {"origin", "sku", "horizon", "y_pred", "lifecycle", "model"} <= set(b.columns)
    assert len(b) == b[SERIES].nunique() * p.horizon  # the long daily file: series x horizon
    man = json.loads((out / "_manifest.json").read_text())
    assert man["shards"] == 3 and len(man["parts"]) == 3
    run_forecast(p, make_exp(), shards=3, out_dir=out)
    assert "exists, skipped" in capsys.readouterr().out


def test_shards_can_run_as_separate_processes(data, tmp_path):
    p = make_project(data_path=data, covariate_eval_policy=CF)
    out = tmp_path / "par"
    run_forecast(p, make_exp(), shards=2, only=[1], out_dir=out)
    assert not (out / "_manifest.json").exists()
    run_forecast(p, make_exp(), shards=2, only=[0], out_dir=out)
    assert (out / "_manifest.json").exists()


def test_sharded_grid_ends_on_the_global_last_date(data):
    p = make_project(data_path=data)
    first, last = date_span(p)
    from forecast_fm.data import load_panel

    for i in range(4):
        panel = load_panel(p, shard=(i, 4), end=last)
        if len(panel):
            assert panel["ts"].max() == last


def test_sharded_fine_tune_backtest_refused(data):
    p = make_project(data_path=data)
    with pytest.raises(ValueError, match="one model per shard"):
        run_backtest(p, make_exp("chronos2", fine_tune={"num_steps": 1}), shards=2)


def test_shard_of_is_stable():
    ids = pd.Series(["a", "b", "c", "SKU1|WH2"])
    assert np.array_equal(shard_of(ids, 7), shard_of(ids.copy(), 7))
