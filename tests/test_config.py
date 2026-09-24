"""project.yaml flexibility: loading layers, data options, plan options,
explicit cutoffs, slices, model defaults."""

import numpy as np
import pandas as pd
import pytest
import yaml

from forecast_fm.backtest import backtest
from forecast_fm.config import load_project
from forecast_fm.data import SERIES, STOCKOUT, TS, Y, build_panel, prepare_raw, read_table
from forecast_fm.folds import fold_cutoffs
from forecast_fm.metrics import score
from forecast_fm.plans import AS_OF, future_frame, load_plans

from .conftest import make_exp, make_project, make_raw, panel_from


def _write(path, d):
    path.write_text(yaml.safe_dump(d))
    return path


# --- loading ------------------------------------------------------------------


def test_sections_extends_profiles_overrides_env(tmp_path, monkeypatch):
    _write(tmp_path / "base.yaml", {
        "data": {"target_col": "units", "series_id_cols": ["sku"]},
        "task": {"horizon": 28, "horizon_buckets": [7, 28]},
        "model_defaults": {"chronos2": {"device": "auto", "batch_size": 256}},
        "profiles": {"full": {"data_path": "full.parquet"}},
    })
    _write(tmp_path / "project.yaml", {
        "extends": "base.yaml",
        "data": {"data_path": "${DATA_DIR:-data}/sample.parquet"},
        "backtest": {"n_folds": 2},
        "model_defaults": {"chronos2": {"batch_size": 512}},
        "profiles": {"mac": {"model_defaults": {"chronos2": {"device": "mps"}}}},
    })
    p = load_project(tmp_path / "project.yaml")
    assert p.data_path == "data/sample.parquet" and p.horizon == 28 and p.n_folds == 2
    assert p.model_defaults["chronos2"] == {"device": "auto", "batch_size": 512}  # deep merge

    monkeypatch.setenv("DATA_DIR", "/mnt/x")
    p = load_project(tmp_path / "project.yaml", profile="mac",
                     overrides=["n_folds=3", "covariate_types.promo_type=categorical"])
    assert p.data_path == "/mnt/x/sample.parquet" and p.n_folds == 3
    assert p.model_defaults["chronos2"]["device"] == "mps"
    assert p.covariate_types == {"promo_type": "categorical"}
    assert load_project(tmp_path / "project.yaml", profile="full").data_path == "full.parquet"

    monkeypatch.setenv("FORECAST_FM_PROFILE", "full")
    assert load_project(tmp_path / "project.yaml").data_path == "full.parquet"


def test_helpful_errors(tmp_path):
    _write(tmp_path / "p.yaml", {"horizn": 35})
    with pytest.raises(ValueError, match="did you mean 'horizon'"):
        load_project(tmp_path / "p.yaml")
    _write(tmp_path / "p.yaml", {"profiles": {"a": {}}})
    with pytest.raises(ValueError, match="profile 'b'"):
        load_project(tmp_path / "p.yaml", profile="b")
    _write(tmp_path / "p.yaml", {"data_path": "${NOPE_NOT_SET}"})
    with pytest.raises(ValueError, match="NOPE_NOT_SET"):
        load_project(tmp_path / "p.yaml")
    _write(tmp_path / "p.yaml", {"extends": "p.yaml"})
    with pytest.raises(ValueError, match="cycle"):
        load_project(tmp_path / "p.yaml")
    with pytest.raises(ValueError) as e:  # all problems reported at once
        make_project(duplicates="add", primary_metric="rmse")
    assert "duplicates" in str(e.value) and "primary_metric" in str(e.value)


def test_repo_project_yaml_and_profiles_load():
    for profile in (None, "full", "smoke", "cuda"):
        p = load_project("project.yaml", profile=profile)
        assert 0.95 in p.quantiles  # service_level joins the quantiles


def test_model_defaults_reach_the_model(monkeypatch):
    from forecast_fm.models import REGISTRY
    from forecast_fm.models.baselines import SeasonalNaive

    seen = {}

    class Rec(SeasonalNaive):
        def __init__(self, params=None):
            seen.update(params or {})
            super().__init__(params)

    monkeypatch.setitem(REGISTRY, "rec", Rec)
    p = make_project(model_defaults={"rec": {"season_length": 14}})
    backtest(panel_from(make_raw(n_series=2), p), p, make_exp("rec"))
    assert seen == {"season_length": 14}
    backtest(panel_from(make_raw(n_series=2), p), p, make_exp("rec", season_length=7))
    assert seen == {"season_length": 7}  # the experiment wins
    assert make_project().model_params("chronos2", {})["cache_dir"] == "reports/chronos2_ft"


# --- data options ---------------------------------------------------------------


def test_derived_filter_and_date_window():
    raw = make_raw(n_series=4, n_days=60)
    raw["regular_price"] = 10.0
    p = make_project(derived_columns={"disc": "1 - price / regular_price", "disc2": "disc * 2"},
                     known_covariate_cols=["price", "disc2"], covariate_eval_policy={},
                     row_filter="category == 'A'", start_date="2024-01-10", end_date="2024-02-15")
    panel = panel_from(raw, p)
    assert set(panel[SERIES].cat.categories) == {"S0", "S1"}
    assert panel[TS].min() == pd.Timestamp("2024-01-10") and panel[TS].max() == pd.Timestamp("2024-02-15")
    promo = panel["price"] < 10
    assert np.allclose(panel.loc[promo, "disc2"], 0.4) and np.allclose(panel.loc[~promo, "disc2"], 0)


def test_duplicates_aggregated():
    raw = make_raw(n_series=1, n_days=10)
    extra = raw.iloc[[3]].copy()
    extra["units"] = 5.0
    both = pd.concat([raw, extra])
    for rule, want in (("sum", raw["units"].iloc[3] + 5), ("last", 5.0),
                       ("max", max(raw["units"].iloc[3], 5.0))):
        p = make_project(duplicates=rule)
        panel = build_panel(prepare_raw(both, p), p)
        assert panel.loc[panel[TS] == raw["date"].iloc[3], Y].iloc[0] == want


def test_negative_and_missing_target_rules():
    raw = make_raw(n_series=1, n_days=20)
    raw.loc[2, "units"] = -3.0
    raw = raw.drop(index=5)
    clip = panel_from(raw, make_project(negative_target="clip"))
    assert clip[Y].iloc[2] == 0
    nan = panel_from(raw, make_project(negative_target="nan", missing_target="nan"))
    assert np.isnan(nan[Y].iloc[2]) and np.isnan(nan[Y].iloc[5])
    with pytest.raises(ValueError, match="negative"):
        panel_from(raw, make_project(negative_target="error"))


def test_stockout_expr():
    raw = make_raw(n_series=1, n_days=30)
    raw["stock_on_hand"] = np.where(np.arange(30) % 10 == 0, 0, 5)
    p = make_project(in_stock_col=None, past_covariate_cols=["sessions"],
                     stockout_expr="stock_on_hand <= 0")
    panel = panel_from(raw, p)
    assert panel[STOCKOUT].to_numpy().nonzero()[0].tolist() == [0, 10, 20]


def test_min_history_days():
    raw = make_raw(n_series=3, n_days=60)
    raw = raw[~((raw["sku"] == "S2") & (raw["date"] < "2024-02-20"))]
    panel = panel_from(raw, make_project(min_history_days=30))
    assert set(panel[SERIES].cat.categories) == {"S0", "S1"}


def test_glob_and_list_inputs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    raw = make_raw(n_series=2, n_days=10)
    raw.iloc[:10].to_parquet("part-1.parquet")
    raw.iloc[10:].to_csv("part-2.csv", index=False)
    assert len(read_table("part-*.parquet")) == 10
    assert len(read_table(["part-1.parquet", "part-2.csv"])) == 20


# --- plans, folds, slices ---------------------------------------------------------


def test_plan_column_mapping_age_and_coverage(tmp_path):
    raw = make_raw(n_series=2, n_days=120)
    cutoff = pd.Timestamp("2024-03-31")
    issued = cutoff - pd.Timedelta(days=10)
    rows = [{"issued": issued, "day": d, "sku": s, "planned_price": 7.0, "promo_flag": 0,
             "promo_type": "none"}
            for d in pd.date_range(issued + pd.Timedelta(days=1), periods=30) for s in ("S0", "S1")]
    pd.DataFrame(rows).to_parquet(tmp_path / "plans.parquet")
    p = make_project(covariate_eval_policy={"price": "plan"},
                     planned_covariates_path=str(tmp_path / "plans.parquet"),
                     plan_as_of_col="issued", plan_timestamp_col="day",
                     plan_columns={"planned_price": "price"}, plan_max_age_days=7,
                     min_plan_coverage=0.5)
    panel = panel_from(raw, p)
    plans = load_plans(p)
    assert {AS_OF, "price"} <= set(plans.columns)
    with pytest.warns(UserWarning, match="10 days old"):
        fut = future_frame(panel[panel[TS] <= cutoff], cutoff, p, plans,
                           actuals=panel[panel[TS] > cutoff], verbose=False)
    assert set(fut["price"].dropna()) == {7.0}  # the 10-day-old snapshot still supplies values


def test_explicit_cutoffs():
    p = make_project(cutoffs=["2024-04-01", "2024-03-01"], holdout_cutoffs=["2024-05-01"])
    first, last = pd.Timestamp("2024-01-01"), pd.Timestamp("2024-06-30")
    assert fold_cutoffs(first, last, p) == [pd.Timestamp("2024-03-01"), pd.Timestamp("2024-04-01")]
    assert fold_cutoffs(first, last, p, holdout=True) == [pd.Timestamp("2024-05-01")]
    with pytest.raises(ValueError, match="overlap"):
        fold_cutoffs(first, last, make_project(cutoffs=["2024-04-25"], holdout_cutoffs=["2024-05-01"]))
    with pytest.raises(ValueError, match="past the last date"):
        fold_cutoffs(first, last, make_project(cutoffs=["2024-06-25"]))


def test_slice_cols_tables():
    p = make_project(slice_cols=["category"])
    panel = panel_from(make_raw(), p)
    preds, _ = backtest(panel, p, make_exp())
    statics = panel.drop_duplicates(SERIES).set_index(SERIES)[["category"]]
    t = score(preds, p, None, statics)["tables"]["category"]
    assert set(t["category"]) == {"A", "B"}
    with pytest.raises(ValueError, match="slice_cols"):
        make_project(slice_cols=["price"])


def test_cli_config_command(tmp_path, monkeypatch, capsys):
    from forecast_fm.cli import main

    monkeypatch.chdir(tmp_path)
    make_raw(n_series=2, n_days=10).to_parquet("sales.parquet")
    d = dict(vars(make_project()))
    d["data_path"] = "sales.parquet"
    _write(tmp_path / "project.yaml", d)
    main(["--set", "horizon=7", "--set", "horizon_buckets=[7]", "config", "--check-data"])
    out = capsys.readouterr().out
    assert "horizon: 7" in out and "all declared columns present" in out
