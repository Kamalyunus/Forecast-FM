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
    # in_stock = 0 marks a stockout; SKU-2002's missing 09-29 row is a zero-sales day
    assert panel.loc[(panel[SERIES] == "SKU-1001") & (panel[TS] == "2026-09-30"), STOCKOUT].item()
    assert panel.loc[(panel[SERIES] == "SKU-2002") & (panel[TS] == "2026-09-29"), "y"].item() == 0

    origin = panel[TS].max()
    plans = load_plans(p)
    fut = future_frame(panel, origin, p, plans, production=True, verbose=False)
    assert fut.loc[(fut[SERIES] == "SKU-1001") & (fut["horizon"] == 2), "price"].item() == \
        pytest.approx(14.99)

    new = production_new_series(p, panel, origin, plans).set_index(SERIES)
    assert set(new.index) == {"SKU-3003", "SKU-4004"}  # from the file; 3003 also in the plan
    assert new.loc["SKU-4004", LAUNCH] == pd.Timestamp("2026-11-01")
