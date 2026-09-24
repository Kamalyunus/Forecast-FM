import numpy as np
import pandas as pd
import pytest

from forecast_fm.data import SERIES, STOCKOUT, TS, Y, build_panel, demand_classes, prepare_raw

from .conftest import make_project, make_raw, panel_from


def _reference_grid(raw: pd.DataFrame, project) -> pd.DataFrame:
    """Straightforward per-series reindex: the behavior the vectorized grid
    must reproduce."""
    df = prepare_raw(raw, project)
    end = df[TS].max()
    out = []
    for sid, g in df.groupby(SERIES):
        idx = pd.date_range(g[TS].min(), end, freq="D")
        g = g.set_index(TS).reindex(idx)
        g.index.name = TS
        g[SERIES] = sid
        g[Y] = g[Y].fillna(0.0)
        g["price"] = g["price"].ffill()
        g["promo_flag"] = g["promo_flag"].fillna(0.0)
        out.append(g.reset_index())
    return pd.concat(out, ignore_index=True)


def test_grid_matches_reference_and_preserves_mass():
    project = make_project()
    raw = make_raw(n_series=5, n_days=90, drop_frac=0.3)
    panel = panel_from(raw, project)
    ref = _reference_grid(raw, project)

    assert len(panel) == len(ref)
    m = panel.merge(ref, on=[SERIES, TS], suffixes=("", "_ref"))
    assert len(m) == len(panel)
    assert np.allclose(m[Y], m[f"{Y}_ref"])
    assert np.allclose(m["price"], m["price_ref"], equal_nan=True)
    assert np.allclose(m["promo_flag"], m["promo_flag_ref"])
    assert np.isclose(panel[Y].sum(), raw["units"].sum())


def test_grid_fill_rules_and_dtypes():
    project = make_project()
    raw = make_raw(n_series=2, n_days=30)
    raw = raw[~((raw["sku"] == "S0") & (raw["date"] == "2024-01-10"))]
    panel = panel_from(raw, project)
    row = panel[(panel[SERIES] == "S0") & (panel[TS] == "2024-01-10")].iloc[0]
    assert row[Y] == 0 and row["promo_flag"] == 0
    assert np.isnan(row["sessions"])  # past covariate: unobserved, not invented
    assert row["promo_type"] == ""     # categorical nan-fill -> absence token
    assert not row[STOCKOUT]
    assert isinstance(panel[SERIES].dtype, pd.CategoricalDtype)
    assert panel[Y].dtype == np.float32 and panel["price"].dtype == np.float32
    assert isinstance(panel["promo_type"].dtype, pd.CategoricalDtype)
    assert (panel.groupby(SERIES, observed=True)["category"].nunique() == 1).all()


def test_stockout_flag_from_in_stock():
    project = make_project()
    raw = make_raw(n_series=1, n_days=30)
    panel = panel_from(raw, project)
    assert (panel[STOCKOUT].to_numpy() == (raw["in_stock"] == 0).to_numpy()).all()


def test_grid_end_series_stops_at_last_row():
    raw = make_raw(n_series=2, n_days=30)
    raw = raw[~((raw["sku"] == "S1") & (raw["date"] > "2024-01-20"))]
    g = panel_from(raw, make_project())
    s = panel_from(raw, make_project(grid_end="series"))
    assert g[g[SERIES] == "S1"][TS].max() == pd.Timestamp("2024-01-30")
    assert s[s[SERIES] == "S1"][TS].max() == pd.Timestamp("2024-01-20")


def test_duplicate_rows_raise():
    project = make_project()
    raw = make_raw(n_series=1, n_days=10)
    with pytest.raises(ValueError, match="duplicate"):
        build_panel(prepare_raw(pd.concat([raw, raw.iloc[:1]]), project), project)


def test_multi_column_series_id():
    project = make_project(series_id_cols=["sku", "category"])
    panel = panel_from(make_raw(n_series=2, n_days=10), project)
    assert set(panel[SERIES].cat.categories) == {"S0|A", "S1|B"}


def test_demand_classes_label_and_computed():
    raw = make_raw(n_series=3, n_days=120)
    labeled = demand_classes(panel_from(raw, make_project()), make_project())
    assert labeled["S2"] == "intermittent" and labeled["S0"] == "BAU"
    p = make_project(demand_label_col=None)
    computed = demand_classes(panel_from(raw, p), p)
    assert computed["S0"] == "smooth"
    assert computed["S2"] in {"intermittent", "lumpy"}
