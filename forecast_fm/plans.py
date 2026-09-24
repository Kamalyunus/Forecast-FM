"""Horizon values of known covariates: what the model is GIVEN for the
forecast window.

Policies per known covariate (`covariate_eval_policy`):

    actual         the realized value. Honest only if it never changes after
                   the cutoff (a fixed calendar).
    plan           the latest plan snapshot with as_of <= cutoff. Snapshots
                   issued after the cutoff are never visible.
    carry_forward  the last value at or before the cutoff, held flat.

Past covariates never get horizon values: they are not in the future frame.
Missing horizon values stay NaN (never 0: a price of 0 is poison for a
covariate-aware model).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .config import ProjectConfig
from .data import SERIES, TS, categorical_cols, make_series_id, read_table

AS_OF = "as_of"
HORIZON = "horizon"
MIN_PLAN_COVERAGE = 0.99


def load_plans(project: ProjectConfig) -> pd.DataFrame | None:
    """Plan snapshots: `as_of, <timestamp_col>, <series_id_cols...>, <known
    covariates...>` as csv or parquet; each snapshot covers as_of+1 ..
    as_of+horizon."""
    if not project.planned_covariates_path:
        return None
    df = read_table(project.planned_covariates_path)
    need = [AS_OF, project.timestamp_col, *project.series_id_cols]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"plan snapshots missing columns {missing}")
    df[SERIES] = make_series_id(df, project.series_id_cols)
    df = df.rename(columns={project.timestamp_col: TS})
    df[TS] = pd.to_datetime(df[TS]).dt.normalize()
    df[AS_OF] = pd.to_datetime(df[AS_OF]).dt.normalize()
    keep = [c for c in project.known_covariate_cols if c in df.columns]
    return df[[AS_OF, SERIES, TS, *keep]]


def snapshot_as_of(plans: pd.DataFrame, cutoff: pd.Timestamp) -> tuple[pd.Timestamp | None, pd.DataFrame]:
    """The latest snapshot issued at or before `cutoff`."""
    dates = plans[AS_OF].unique()
    dates = dates[dates <= np.datetime64(cutoff)]
    if len(dates) == 0:
        return None, plans.iloc[0:0]
    as_of = pd.Timestamp(dates.max())
    return as_of, plans[plans[AS_OF] == as_of]


def future_frame(history: pd.DataFrame, cutoff: pd.Timestamp, project: ProjectConfig,
                 plans: pd.DataFrame | None, actuals: pd.DataFrame | None = None,
                 production: bool = False, verbose: bool = True) -> pd.DataFrame:
    """[series_id, ts, horizon, <known covariates>] for every series with
    history, horizon 1..H after `cutoff`.

    `history` holds data <= cutoff only. `actuals` holds the rows after the
    cutoff (backtest); only declared known covariates are read from it. In
    production there are no actuals, so `actual` covariates come from the
    latest plan snapshot as well.
    """
    H = project.horizon
    sids = history[SERIES].unique()
    if hasattr(sids, "categories"):
        sids = sids.astype(history[SERIES].dtype)
    fut = pd.DataFrame({
        SERIES: np.repeat(np.asarray(sids), H),
        TS: np.tile(pd.date_range(cutoff + pd.Timedelta(days=1), periods=H, freq="D").to_numpy(), len(sids)),
        HORIZON: np.tile(np.arange(1, H + 1, dtype=np.int16), len(sids)),
    })
    fut[SERIES] = fut[SERIES].astype(history[SERIES].dtype)

    known = [c for c in project.known_covariate_cols if c in history.columns]
    if not known:
        return fut
    by_policy: dict[str, list[str]] = {}
    for c in known:
        p = project.policy(c)
        if production and p == "actual":
            p = "plan"
        by_policy.setdefault(p, []).append(c)

    if "actual" in by_policy:
        if actuals is None:
            raise ValueError("policy 'actual' needs the realized future rows")
        cols = by_policy["actual"]
        a = actuals[[SERIES, TS, *cols]]
        fut = fut.merge(a, on=[SERIES, TS], how="left")
    if "carry_forward" in by_policy:
        cols = by_policy["carry_forward"]
        last = history.sort_values(TS).groupby(SERIES, observed=True)[cols].last()
        fut = fut.merge(last, left_on=SERIES, right_index=True, how="left")
    if "plan" in by_policy:
        cols = by_policy["plan"]
        if plans is None:
            if production:
                raise ValueError(f"a production forecast reads known covariates {cols} from the "
                                 "latest plan snapshot: set planned_covariates_path to a file "
                                 "with today's as_of (or use carry_forward for them)")
            raise ValueError(f"policy 'plan' for {cols} but no plan snapshots loaded")
        as_of, snap = snapshot_as_of(plans, cutoff)
        missing_cols = [c for c in cols if c not in snap.columns]
        if missing_cols:
            raise ValueError(f"plan snapshots have no column(s) {missing_cols}")
        snap = snap[[SERIES, TS, *cols]].copy()
        snap[SERIES] = snap[SERIES].astype(fut[SERIES].dtype)
        fut = fut.merge(snap, on=[SERIES, TS], how="left")
        coverage = {c: float(fut[c].notna().mean()) for c in cols}
        if verbose:
            print(f"[plan] cutoff {cutoff.date()}: snapshot as_of "
                  f"{as_of.date() if as_of is not None else 'NONE'}; coverage "
                  + ", ".join(f"{c}={v:.1%}" for c, v in coverage.items()))
        low = {c: v for c, v in coverage.items() if v < MIN_PLAN_COVERAGE}
        if low:
            warnings.warn(f"plan coverage below {MIN_PLAN_COVERAGE:.0%} at cutoff {cutoff.date()}: "
                          + ", ".join(f"{c}={v:.1%}" for c, v in low.items())
                          + " (missing values are passed as NaN)", stacklevel=2)

    cats = set(categorical_cols(project, history))
    for c in known:
        if c in cats:
            fut[c] = fut[c].astype(object).where(fut[c].notna(), "").astype(str)
        else:
            fut[c] = pd.to_numeric(fut[c], errors="coerce").astype(np.float32)
    return fut[[SERIES, TS, HORIZON, *known]]
