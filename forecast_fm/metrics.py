"""Vectorized scoring of backtest predictions: overall, per fold, per
horizon bucket, per demand class. Stocked-out days are not scored (sales
there are censored demand)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .backtest import Q_PREFIX
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


def prepare(preds: pd.DataFrame, project: ProjectConfig, classes: pd.Series | None = None) -> pd.DataFrame:
    p = preds[preds["y_true"].notna() & ~preds[STOCKOUT].fillna(False).astype(bool)].copy()
    y, f = p["y_true"].to_numpy(np.float64), p["y_pred"].to_numpy(np.float64)
    p["_y"], p["_ae"], p["_e"] = y, np.abs(f - y), f - y
    for q in _quantiles(p):
        d = y - p[f"{Q_PREFIX}{q:g}"].to_numpy(np.float64)
        p[f"_pl{q:g}"] = np.maximum(q * d, (q - 1) * d)
        p[f"_cov{q:g}"] = (d <= 0).astype(np.float64)
    p["bucket"] = bucket_labels(p[HORIZON], project.horizon_buckets)
    if classes is not None:
        p["demand_class"] = p[SERIES].map(classes).astype(str)
    return p


def summarize(p: pd.DataFrame, by: list[str] | None = None) -> pd.DataFrame:
    by = by or []
    qs = _quantiles(p)
    agg_cols = ["_y", "_ae", "_e", *[f"_pl{q:g}" for q in qs], *[f"_cov{q:g}" for q in qs]]
    g = p.groupby(by, observed=True) if by else None
    sums = g[agg_cols].sum() if g is not None else p[agg_cols].sum().to_frame().T
    counts = g.size() if g is not None else pd.Series([len(p)])
    out = pd.DataFrame(index=sums.index)
    out["n"] = counts.to_numpy()
    denom = sums["_y"].replace(0, np.nan)
    out["wape"] = sums["_ae"] / denom
    out["bias"] = sums["_e"] / denom
    if qs:
        out["wql"] = sum(2 * sums[f"_pl{q:g}"] for q in qs) / len(qs) / denom
        for q in qs:
            out[f"cov{q:g}"] = sums[f"_cov{q:g}"] / out["n"]
    # MASE: mean over (fold, series) of MAE / in-sample seasonal-naive MAE
    s = p[p["mase_scale"] > 0]
    ser = s.groupby([*by, "fold", SERIES], observed=True).agg(ae=("_ae", "mean"), sc=("mase_scale", "first"))
    ratio = ser["ae"] / ser["sc"]
    if by:
        mase = ratio.groupby(level=list(range(len(by))), observed=True).mean()
    else:
        mase = pd.Series([ratio.mean()])
    out["mase"] = mase.reindex(out.index) if by else mase.to_numpy()
    return out.reset_index() if by else out.reset_index(drop=True)


def score(preds: pd.DataFrame, project: ProjectConfig, classes: pd.Series | None = None) -> dict:
    p = prepare(preds, project, classes)
    overall = summarize(p).iloc[0].to_dict()
    tables = {"fold": summarize(p, ["fold"]), "bucket": summarize(p, ["bucket"])}
    if classes is not None:
        tables["demand_class"] = summarize(p, ["demand_class"])
        tables["bucket_x_class"] = summarize(p, ["bucket", "demand_class"])
    return {"overall": _clean(overall), "tables": tables}


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
