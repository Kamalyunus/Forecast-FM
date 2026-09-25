"""New-series forecasts (cold start): launch profiles from analogs."""

import numpy as np
import pandas as pd
import pytest

from forecast_fm.backtest import backtest, forecast_at
from forecast_fm.cold_start import LAUNCH, launch_profiles, production_new_series
from forecast_fm.data import SERIES, TS, Y

from .conftest import make_exp, make_project, panel_from

DAYS = pd.date_range("2024-01-01", periods=200)


def _launches(n_per_cat=6):
    """Series launching on staggered dates; demand at age a = slope * min(a, 20)
    with slope 1 in category A and 2 in B. One old series starts on day 0
    (left-censored, never an analog)."""
    rows = []
    for c, slope in (("A", 1.0), ("B", 2.0)):
        for k in range(n_per_cat):
            launch = DAYS[10 + 12 * k]
            for d in DAYS[DAYS >= launch]:
                a = (d - launch).days
                rows.append({"date": d, "sku": f"{c}{k}", "units": slope * min(a, 20), "category": c})
    for d in DAYS:
        rows.append({"date": d, "sku": "OLD", "units": 100.0, "category": "A"})
    raw = pd.DataFrame(rows)
    raw["price"], raw["promo_flag"], raw["promo_type"] = 1.0, 0, "none"
    raw["sessions"], raw["in_stock"], raw["demand_label"] = 1.0, 1, "x"
    return raw


def _project(**kw):
    base = dict(horizon=14, horizon_buckets=[7, 14], n_folds=1, min_train_periods=60,
                cold_start={"profile_cols": ["category"], "min_analogs": 2})
    base.update(kw)
    return make_project(**base)


def test_profiles_by_category_ignore_censored_series():
    p = _project()
    panel = panel_from(_launches(), p)
    prof = launch_profiles(panel, p, max_age=30, quantiles=[0.5])
    by_cat = prof[("category",)]
    assert by_cat.loc[("A", 5), "mean"] == pytest.approx(5.0)
    assert by_cat.loc[("B", 5), "mean"] == pytest.approx(10.0)
    assert by_cat.loc[("A", 25), "mean"] == pytest.approx(20.0)  # OLD (=100) is not an analog


def test_new_series_in_backtest_follow_their_category_profile():
    p = _project()
    panel = panel_from(_launches(), p)
    cutoff = pd.Timestamp("2024-03-08")  # A5/B5 launch 2024-03-11, inside the horizon
    pred, stats = forecast_at(panel, cutoff, p, make_exp(), None)
    b5 = pred[pred[SERIES] == "B5"].set_index(TS)
    assert (b5["lifecycle"] == "new").all()
    assert b5.loc[pd.Timestamp("2024-03-10"), "y_pred"] == 0  # before launch
    assert b5.loc[pd.Timestamp("2024-03-16"), "y_pred"] == pytest.approx(10.0)  # age 5, slope 2
    assert stats["cold_start"]["n_cold"] >= 2


def test_short_history_scaled_by_its_own_sales():
    p = _project(min_history_days=10)
    raw = _launches()
    raw.loc[raw["sku"] == "B4", "units"] *= 3  # sells 3x its category profile
    panel = panel_from(raw, p)
    cutoff = pd.Timestamp("2024-03-03")  # B4 launched 2024-02-28: 5 days of history
    pred, _ = forecast_at(panel, cutoff, p, make_exp(), None)
    b4 = pred[pred[SERIES] == "B4"]
    assert (b4["lifecycle"] == "short_history").all()
    scale = b4["y_pred"].iloc[0] / (2.0 * 5)  # first forecast day = age 5; profile there = 10
    assert 1.5 < scale <= 3.0  # sells 3x: scaled up, shrunk towards 1


def test_profiles_use_history_only():
    p = _project()
    panel = panel_from(_launches(), p)
    cutoff = pd.Timestamp("2024-03-10")
    a, _ = forecast_at(panel, cutoff, p, make_exp(), None)
    tampered = panel.copy()
    tampered.loc[tampered[TS] > cutoff, Y] = 9999.0
    b, _ = forecast_at(tampered, cutoff, p, make_exp(), None)
    new = a["lifecycle"] == "new"
    assert np.allclose(a.loc[new, "y_pred"], b.loc[new, "y_pred"])


def test_backtest_scores_new_series_by_lifecycle():
    from forecast_fm.metrics import score

    p = _project()
    panel = panel_from(_launches(), p)
    preds, _ = backtest(panel, p, make_exp(), cutoffs=[pd.Timestamp("2024-03-10")])
    t = score(preds, p)["tables"]["lifecycle"].set_index("lifecycle")
    assert {"established", "new"} <= set(t.index)
    assert t.loc["new", "wape"] < 0.2  # the profile matches the synthetic launch curve


def test_production_new_series_from_file_and_plans(tmp_path):
    pd.DataFrame({"sku": ["NEW1"], "category": ["B"], "launch_date": ["2024-07-25"]}).to_csv(
        tmp_path / "new.csv", index=False)
    last = DAYS[-1]
    # production reads every known covariate from the plan snapshot
    plan = pd.DataFrame({"as_of": last, "date": pd.date_range(last + pd.Timedelta(days=3), periods=10),
                         "sku": "NEW2", "price": 5.0, "promo_flag": 0, "promo_type": "none"})
    plan.to_csv(tmp_path / "plans.csv", index=False)
    p = _project(covariate_eval_policy={"price": "plan"}, planned_covariates_path=str(tmp_path / "plans.csv"),
                 cold_start={"profile_cols": ["category"], "min_analogs": 2,
                             "new_series_path": str(tmp_path / "new.csv")})
    panel = panel_from(_launches(), p)
    from forecast_fm.plans import load_plans

    plans = load_plans(p)
    new = production_new_series(p, panel, last, plans)
    assert set(new[SERIES]) == {"NEW1", "NEW2"}
    assert new.set_index(SERIES).loc["NEW2", LAUNCH] == last + pd.Timedelta(days=3)
    pred, _ = forecast_at(panel, last, p, make_exp(), plans, production=True, new_series=new)
    n1 = pred[pred[SERIES] == "NEW1"].set_index(TS)["y_pred"]
    assert n1.loc[pd.Timestamp("2024-07-24")] == 0
    assert n1.loc[pd.Timestamp("2024-07-30")] == pytest.approx(10.0)
    assert (pred[pred[SERIES] == "NEW2"]["y_pred"] >= 0).all()  # no statics -> all-series profile


def test_cold_start_methods_zero_and_none():
    panel = panel_from(_launches(), _project())
    cutoff = pd.Timestamp("2024-03-10")
    z, _ = forecast_at(panel, cutoff, _project(cold_start={"method": "zero"}), make_exp(), None)
    assert (z.loc[z["lifecycle"] == "new", "y_pred"] == 0).all()
    n, _ = forecast_at(panel, cutoff, _project(cold_start={"method": "none"}), make_exp(), None)
    assert (n["lifecycle"] == "established").all()
