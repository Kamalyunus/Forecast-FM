import numpy as np
import pandas as pd
import pytest

from forecast_fm.data import SERIES, TS
from forecast_fm.folds import fold_cutoffs
from forecast_fm.plans import AS_OF, HORIZON, future_frame, load_plans

from .conftest import make_project, make_raw, panel_from


def test_cutoffs_fit_horizon_and_holdout_is_disjoint():
    p = make_project(horizon=14, fold_step=7, n_folds=3, holdout_folds=1, min_train_periods=30)
    first, last = pd.Timestamp("2024-01-01"), pd.Timestamp("2024-06-30")
    val = fold_cutoffs(first, last, p)
    hold = fold_cutoffs(first, last, p, holdout=True)
    assert len(val) == 3 and val == sorted(val)
    assert hold == [last - pd.Timedelta(days=14)]
    # no validation horizon window reaches into the holdout window
    assert max(val) + pd.Timedelta(days=14) <= hold[0]


def test_cutoffs_respect_min_train():
    p = make_project(min_train_periods=10_000)
    with pytest.raises(ValueError, match="min_train_periods"):
        fold_cutoffs(pd.Timestamp("2024-01-01"), pd.Timestamp("2024-06-30"), p)


def _plans(tmp_path, cutoff, fmt="parquet"):
    """Two snapshots: one at the cutoff (price 7) and one issued AFTER it
    (price 99): the later one must never be visible."""
    rows = []
    for as_of, price in ((cutoff, 7.0), (cutoff + pd.Timedelta(days=3), 99.0)):
        for d in pd.date_range(as_of + pd.Timedelta(days=1), periods=14):
            for sku in ("S0", "S1"):
                rows.append({AS_OF: as_of, "date": d, "sku": sku, "price": price,
                             "promo_flag": 1, "promo_type": "bogo"})
    df = pd.DataFrame(rows)
    df = df[~((df["sku"] == "S1") & (df["date"] == cutoff + pd.Timedelta(days=5)))]  # a gap
    path = tmp_path / f"plans.{fmt}"
    df.to_parquet(path) if fmt == "parquet" else df.to_csv(path, index=False)
    return path


@pytest.mark.parametrize("fmt", ["parquet", "csv"])
def test_plan_policy_uses_latest_snapshot_not_after_cutoff(tmp_path, fmt):
    raw = make_raw(n_series=2, n_days=120)
    cutoff = pd.Timestamp("2024-03-31")
    p = make_project(covariate_eval_policy={"price": "plan", "promo_flag": "plan", "promo_type": "plan"},
                     planned_covariates_path=str(_plans(tmp_path, cutoff, fmt)))
    panel = panel_from(raw, p)
    hist = panel[panel[TS] <= cutoff]
    with pytest.warns(UserWarning, match="coverage"):
        fut = future_frame(hist, cutoff, p, load_plans(p), actuals=panel[panel[TS] > cutoff])
    assert len(fut) == 2 * 14 and fut[HORIZON].min() == 1 and fut[HORIZON].max() == 14
    assert set(fut["price"].dropna()) == {7.0}
    gap = fut[(fut[SERIES] == "S1") & (fut[TS] == cutoff + pd.Timedelta(days=5))]
    assert np.isnan(gap["price"].iloc[0])  # missing plan -> NaN, never 0
    assert "sessions" not in fut and "in_stock" not in fut  # past covariates never in horizon


def test_actual_and_carry_forward_policies():
    raw = make_raw(n_series=2, n_days=120)
    p = make_project(covariate_eval_policy={"price": "carry_forward"})
    panel = panel_from(raw, p)
    cutoff = pd.Timestamp("2024-03-31")
    hist, after = panel[panel[TS] <= cutoff], panel[panel[TS] > cutoff]
    fut = future_frame(hist, cutoff, p, None, actuals=after)
    last = hist.sort_values(TS).groupby(SERIES, observed=True)["price"].last()
    assert np.allclose(fut["price"], fut[SERIES].map(last).astype(float))
    m = fut.merge(after[[SERIES, TS, "promo_flag"]], on=[SERIES, TS], suffixes=("", "_a"))
    assert np.allclose(m["promo_flag"], m["promo_flag_a"])  # 'actual' = realized


def test_production_reads_actual_covariates_from_plans(tmp_path):
    raw = make_raw(n_series=2, n_days=91)
    cutoff = pd.Timestamp(raw["date"].max())
    p = make_project(covariate_eval_policy={"price": "plan"},
                     planned_covariates_path=str(_plans(tmp_path, cutoff)))
    panel = panel_from(raw, p)
    with pytest.warns(UserWarning):
        fut = future_frame(panel, cutoff, p, load_plans(p), production=True)
    assert set(fut["promo_flag"].dropna()) == {1.0}  # from the plan, not zero-filled
