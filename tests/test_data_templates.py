"""docs/data_templates/ must load with the repo's project.yaml, so the data
requirements doc can't drift from what the code accepts."""

from pathlib import Path

import pandas as pd
import pytest

from forecast_fm.cold_start import LAUNCH, production_new_series
from forecast_fm.config import load_project
from forecast_fm.data import SERIES, STOCKOUT, TS, build_panel, load_raw
from forecast_fm.plans import future_frame, load_plans

ROOT = Path(__file__).resolve().parents[1]
T = ROOT / "docs" / "data_templates"


def _project():
    return load_project(ROOT / "project.yaml", overrides=[
        f"data_path={T / 'sales_history.csv'}",
        f"planned_covariates_path={T / 'plan_snapshots.csv'}",
        f"cold_start.new_series_path={T / 'new_skus.csv'}",
    ])


@pytest.mark.filterwarnings("ignore:plan coverage")  # the template plan only lists 2 days
def test_templates_load_with_the_repo_project_yaml():
    p = _project()
    panel = build_panel(load_raw(p), p)
    assert set(panel[SERIES].astype(str)) == {"SKU-1001", "SKU-2002"}
    day = lambda sku, d: panel[(panel[SERIES] == sku) & (panel[TS] == d)].iloc[0]  # noqa: E731
    # oos_hours: 18h out of stock -> censored day; 3h -> kept, availability 21/24
    assert day("SKU-1001", "2026-09-30")[STOCKOUT]
    assert not day("SKU-1001", "2026-09-29")[STOCKOUT]
    assert day("SKU-1001", "2026-09-29")["availability"] == pytest.approx(0.875)
    # SKU-2002's missing 09-29 row: zero sales, availability unknown (not "fully on sale")
    assert day("SKU-2002", "2026-09-29")["y"] == 0
    assert pd.isna(day("SKU-2002", "2026-09-29")["availability"])

    origin = panel[TS].max()
    plans = load_plans(p)
    fut = future_frame(panel, origin, p, plans, production=True, verbose=False)
    assert fut.loc[(fut[SERIES] == "SKU-1001") & (fut["horizon"] == 2), "price"].item() == \
        pytest.approx(14.99)

    new = production_new_series(p, panel, origin, plans).set_index(SERIES)
    assert set(new.index) == {"SKU-3003", "SKU-4004"}  # from the file; 3003 also in the plan
    assert new.loc["SKU-4004", LAUNCH] == pd.Timestamp("2026-11-01")
