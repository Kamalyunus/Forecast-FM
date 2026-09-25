"""Vectorized scoring of backtest predictions: overall, per fold, per
horizon bucket, per demand class, per lifecycle (established / short
history / new), per slice column.

Stocked-out days are not scored (sales there are censored demand). Rows with
no forecast are not scored either, and are counted in `n_missing_pred` so a
gap is visible rather than silently flattering WAPE.

Every table is built from additive partial sums (`partials`), so sharded
runs combine shard by shard (`combine`) and get exactly the metrics of one
unsharded run.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .backtest import LIFECYCLE, Q_PREFIX
from .config import ProjectConfig
from .data import SERIES, STOCKOUT
from .plans import HORIZON

METRICS = ("wape", "bias", "mase", "wql")


def bucket_labels(horizon: pd.Series, edges: list[int]) -> pd.Series:
    bins = [0, *edges]
    labels = [f"h{lo + 1:02d}-{hi}" for lo, hi in zip(bins[:-1], bins[1:], strict=True)]
    return pd.cut(horizon, bins=bins, labels=labels)


def _quantiles(preds: pd.DataFrame) -> list[float]:
    return sorted(float(c[len(Q_PREFIX):]) for c in preds.columns if c.startswith(Q_PREFIX))


def prepare(preds: pd.DataFrame, project: ProjectConfig, classes: pd.Series | None = None,
            statics: pd.DataFrame | None = None) -> tuple[pd.DataFrame, int]:
    """(scorable rows with error columns, number of rows missing a forecast)."""
    has_truth = preds["y_true"].notna() & ~preds[STOCKOUT].fillna(False).astype(bool)
    missing = has_truth & preds["y_pred"].isna()
    p = preds[has_truth & ~missing].copy()
    y, f = p["y_true"].to_numpy(np.float64), p["y_pred"].to_numpy(np.float64)
    p["_y"], p["_ae"], p["_e"] = y, np.abs(f - y), f - y
    for q in _quantiles(p):
        d = y - p[f"{Q_PREFIX}{q:g}"].to_numpy(np.float64)
        p[f"_pl{q:g}"] = np.maximum(q * d, (q - 1) * d)
        p[f"_cov{q:g}"] = (d <= 0).astype(np.float64)
    p["bucket"] = bucket_labels(p[HORIZON], project.horizon_buckets)
    sid = p[SERIES].astype(str)
    if classes is not None:
        cl = classes.copy()
        cl.index = cl.index.astype(str)
        p["demand_class"] = sid.map(cl).fillna("no_history").astype(str)
    if statics is not None:
        st = statics.copy()
        st.index = st.index.astype(str)
        for c in project.slice_cols:
            p[c] = sid.map(st[c]).astype(str)
    if LIFECYCLE not in p.columns:
        p[LIFECYCLE] = "established"
    return p, int(missing.sum())


def table_specs(project: ProjectConfig, with_classes: bool, with_statics: bool) -> dict[str, list[str]]:
    specs = {"overall": [], "fold": ["fold"], "bucket": ["bucket"], LIFECYCLE: [LIFECYCLE]}
    if with_classes:
        specs["demand_class"] = ["demand_class"]
        specs["bucket_x_class"] = ["bucket", "demand_class"]
    if with_statics:
        for c in project.slice_cols:
            if c != "demand_class":
                specs[c] = [c]
    return specs


def partial(p: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Additive sufficient statistics per group: sums of y, |e|, e, pinball
    and coverage per quantile, row count, and the MASE ratio sum/count over
    (fold, series)."""
    qs = _quantiles(p)
    cols = ["_y", "_ae", "_e", *[f"_pl{q:g}" for q in qs], *[f"_cov{q:g}" for q in qs]]
    key = by or ["_all"]
    q = p.assign(_all="all") if not by else p
    sums = q.groupby(key, observed=True)[cols].sum()
    sums["n"] = q.groupby(key, observed=True).size()
    s = q[q["mase_scale"] > 0]
    ser = s.groupby([*key, "fold", SERIES], observed=True).agg(ae=("_ae", "mean"), sc=("mase_scale", "first"))
    ratio = (ser["ae"] / ser["sc"]).groupby(level=list(range(len(key))), observed=True)
    sums["_mase_sum"] = ratio.sum().reindex(sums.index).fillna(0.0)
    sums["_mase_n"] = ratio.size().reindex(sums.index).fillna(0).astype(np.int64)
    return sums.reset_index()


def finalize(parts: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    key = by or ["_all"]
    sums = parts.groupby(key, observed=True).sum(numeric_only=True)
    qs = sorted(float(c[3:]) for c in sums.columns if c.startswith("_pl"))
    out = pd.DataFrame(index=sums.index)
    out["n"] = sums["n"].astype(np.int64)
    denom = sums["_y"].replace(0, np.nan)
    out["wape"] = sums["_ae"] / denom
    out["bias"] = sums["_e"] / denom
    if qs:
        out["wql"] = sum(2 * sums[f"_pl{q:g}"] for q in qs) / len(qs) / denom
        for q in qs:
            out[f"cov{q:g}"] = sums[f"_cov{q:g}"] / out["n"]
    out["mase"] = sums["_mase_sum"] / sums["_mase_n"].replace(0, np.nan)
    return out.reset_index(drop=not by)


def summarize(p: pd.DataFrame, by: list[str] | None = None) -> pd.DataFrame:
    by = by or []
    return finalize(partial(p, by), by)


def partials(preds: pd.DataFrame, project: ProjectConfig, classes: pd.Series | None = None,
             statics: pd.DataFrame | None = None) -> dict:
    """{table name: partial frame} plus `_missing_pred`: combine across
    shards with `combine`, turn into metrics with `result`."""
    p, missing = prepare(preds, project, classes, statics)
    specs = table_specs(project, classes is not None, statics is not None)
    out = {name: partial(p, by) for name, by in specs.items()}
    out["_missing_pred"] = missing
    return out


def combine(parts: list[dict]) -> dict:
    out: dict = {}
    for d in parts:
        for k, v in d.items():
            if k == "_missing_pred":
                out[k] = out.get(k, 0) + v
            else:
                out.setdefault(k, []).append(v)
    return {k: (pd.concat(v, ignore_index=True) if isinstance(v, list) else v) for k, v in out.items()}


def result(parts: dict, project: ProjectConfig) -> dict:
    specs = table_specs(project, "demand_class" in parts, any(c in parts for c in project.slice_cols))
    tables = {name: finalize(parts[name], by) for name, by in specs.items() if name in parts}
    overall = _clean(tables.pop("overall").iloc[0].to_dict())
    overall["n_missing_pred"] = int(parts.get("_missing_pred", 0))
    return {"overall": overall, "tables": tables}


def score(preds: pd.DataFrame, project: ProjectConfig, classes: pd.Series | None = None,
          statics: pd.DataFrame | None = None) -> dict:
    """`statics`: one row per series (index series_id) with the slice_cols."""
    return result(combine([partials(preds, project, classes, statics)]), project)


def _clean(d: dict) -> dict:
    """numpy scalars -> JSON-safe Python values; NaN -> None."""
    out = {}
    for k, v in d.items():
        if isinstance(v, (np.floating, float)):
            out[k] = None if np.isnan(v) else float(v)
        elif isinstance(v, np.integer):
            out[k] = int(v)
        else:
            out[k] = v
    return out
