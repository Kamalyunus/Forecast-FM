"""Rolling-origin backtest: the single evaluation harness.

At each cutoff the model gets `history` (ts <= cutoff) and the future frame
(series, ts, horizon, known covariates set by covariate_eval_policy). Target
values and past covariates after the cutoff are never passed to the model;
they are joined back only for scoring, after predict returns.

Every series is forecast, by one of two routes, recorded in `lifecycle`:

    established     history >= min_history_days at the origin -> the model
    short_history   0 < history < min_history_days -> cold start
    new             no history; launches within the horizon -> cold start
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import cold_start
from .config import ExperimentConfig, ProjectConfig
from .data import SERIES, STOCKOUT, TS, Y
from .folds import fold_cutoffs, fold_origins, origin_limit
from .models import create_model
from .plans import HORIZON, future_frame

Q_PREFIX = "q_"
LIFECYCLE = cold_start.LIFECYCLE


def qcol(q: float) -> str:
    return f"{Q_PREFIX}{q:g}"


def mase_scale(history: pd.DataFrame, m: int) -> pd.Series:
    """Per-series in-sample MAE of the seasonal naive forecast (lag m) on
    in-stock days: the MASE denominator."""
    h = history[[SERIES, Y, STOCKOUT]]
    y = h[Y].where(~h[STOCKOUT])
    lag = y.groupby(h[SERIES], observed=True).shift(m)
    return (y - lag).abs().groupby(h[SERIES], observed=True).mean()


def _split_by_history(history: pd.DataFrame, project: ProjectConfig, cutoff: pd.Timestamp):
    """(established series ids, short-history frame for cold start)."""
    g = history.groupby(SERIES, observed=True)
    first = g[TS].min()
    age = ((cutoff - first).dt.days + 1).astype(np.int64)  # days of history at the origin
    need = max(int(project.min_history_days), 1)
    established = age.index[age >= need]
    short_ids = age.index[age < need]
    short = pd.DataFrame(columns=[SERIES, cold_start.LAUNCH, "hist_age", "hist_sum"])
    if len(short_ids):
        h = history[history[SERIES].isin(short_ids)]
        sums = h[Y].where(~h[STOCKOUT]).groupby(h[SERIES], observed=True).sum()
        statics = h.drop_duplicates(SERIES).set_index(SERIES)[
            [c for c in project.static_cols if c in h.columns]]
        short = statics.reindex(short_ids).reset_index().rename(columns={"index": SERIES})
        short[cold_start.LAUNCH] = first.reindex(short_ids).to_numpy()
        short["hist_age"] = age.reindex(short_ids).to_numpy()
        short["hist_sum"] = sums.reindex(short_ids).fillna(0).to_numpy()
    return established, short


def _new_in_window(panel: pd.DataFrame, project: ProjectConfig, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Backtest: series first seen within the horizon, treated as planned
    launches (launch date and statics known at the cutoff)."""
    first = panel.groupby(SERIES, observed=True)[TS].min()
    end = cutoff + pd.Timedelta(days=project.horizon)
    ids = first.index[(first > cutoff) & (first <= end)]
    if not len(ids):
        return pd.DataFrame(columns=[SERIES, cold_start.LAUNCH, "hist_age", "hist_sum"])
    rows = panel[panel[SERIES].isin(ids)].drop_duplicates(SERIES).set_index(SERIES)
    out = rows[[c for c in project.static_cols if c in rows.columns]].reset_index()
    out[cold_start.LAUNCH] = first.reindex(out[SERIES]).to_numpy()
    out["hist_age"] = 0
    out["hist_sum"] = 0.0
    return out


def fit_at(panel: pd.DataFrame, cutoff: pd.Timestamp, project: ProjectConfig, exp: ExperimentConfig):
    """Create the experiment's model and fit it on history <= cutoff (the
    series established at the cutoff)."""
    cutoff = pd.Timestamp(cutoff)
    history = panel[panel[TS] <= cutoff]
    established, _ = _split_by_history(history, project, cutoff)
    model = create_model(exp.model, project.model_params(exp.model, exp.model_params))
    model.fit(history[history[SERIES].isin(established)], project, cutoff)
    return model


def forecast_at(panel: pd.DataFrame, cutoff: pd.Timestamp, project: ProjectConfig,
                exp: ExperimentConfig, plans: pd.DataFrame | None, production: bool = False,
                new_series: pd.DataFrame | None = None, model=None) -> tuple[pd.DataFrame, dict]:
    """One origin: fit on history <= cutoff, forecast horizon 1..H for every
    series (model for established ones, cold start for the rest).

    `model`: an already-fitted model (fit at an earlier retrain date); it then
    only reads history <= this origin as context. `new_series` (production):
    upcoming series with SERIES, launch_date and statics. In a backtest they
    come from the panel (first seen in-window)."""
    cutoff = pd.Timestamp(cutoff)
    history = panel[panel[TS] <= cutoff]
    established, short = _split_by_history(history, project, cutoff)
    model_history = history[history[SERIES].isin(established)] if len(short) else history
    if production:
        new = new_series if new_series is not None else pd.DataFrame(columns=[SERIES, cold_start.LAUNCH])
        new = new.assign(hist_age=0, hist_sum=0.0)
    else:
        new = _new_in_window(panel, project, cutoff)

    frames, stats = [], {}
    if len(established):
        actuals = None
        if not production:
            known = [c for c in project.known_covariate_cols if c in panel.columns]
            window = panel[(panel[TS] > cutoff) & (panel[TS] <= cutoff + pd.Timedelta(days=project.horizon))]
            actuals = window[[SERIES, TS, *known]]  # declared-known columns only
        future = future_frame(model_history, cutoff, project, plans, actuals=actuals,
                              production=production)
        if model is None:
            model = create_model(exp.model, project.model_params(exp.model, exp.model_params))
            model.fit(model_history, project, cutoff)
        point, qd = model.predict(model_history, future, project)
        out = future[[SERIES, TS, HORIZON]].copy()
        out[SERIES] = out[SERIES].astype(str)
        out["y_pred"] = np.asarray(point, dtype=np.float32)
        if qd:
            for q, v in qd.items():
                out[qcol(q)] = np.asarray(v, dtype=np.float32)
        out[LIFECYCLE] = "established"
        frames.append(out)
        stats.update(getattr(model, "stats", {}) or {})

    cold = pd.concat([f for f in (short, new) if len(f)], ignore_index=True) if (len(short) or len(new)) \
        else pd.DataFrame()
    if len(cold):
        cfc, cstats = cold_start.forecast(cold, history, project, cutoff)
        if len(cfc):
            kinds = pd.Series(np.where(cold["hist_age"].to_numpy() > 0, "short_history", "new"),
                              index=cold[SERIES].astype(str).to_numpy())
            cfc[LIFECYCLE] = cfc[SERIES].map(kinds).to_numpy()
            keep = [c for c in cfc.columns if not c.startswith(Q_PREFIX) or not frames
                    or c in frames[0].columns]
            frames.append(cfc[keep].astype({"y_pred": np.float32}))
        stats["cold_start"] = cstats
    if not frames:
        return pd.DataFrame(columns=[SERIES, TS, HORIZON, "y_pred", LIFECYCLE]), stats
    return pd.concat(frames, ignore_index=True), stats


def backtest(panel: pd.DataFrame, project: ProjectConfig, exp: ExperimentConfig,
             plans: pd.DataFrame | None = None, holdout: bool = False,
             cutoffs: list[pd.Timestamp] | None = None,
             origins: list[list[pd.Timestamp]] | None = None) -> tuple[pd.DataFrame, list[dict]]:
    """Per fold: fit the model once at the fold's cutoff (its retrain date),
    then forecast from each of the fold's origins (`origin_step_days`), the
    model reading history <= each origin. Scored on the H days after each
    origin."""
    first, last = panel[TS].min(), panel[TS].max()
    if cutoffs is None:
        cutoffs = fold_cutoffs(first, last, project, holdout=holdout)
    if origins is None:
        limit = last if holdout else origin_limit(first, last, project)
        origins = fold_origins(cutoffs, project, limit)
    frames, stats = [], []
    H = pd.Timedelta(days=project.horizon)
    for k, (cutoff, fold_origs) in enumerate(zip(cutoffs, origins, strict=True)):
        print(f"[backtest] fold {k} cutoff {cutoff.date()} ({exp.model}), "
              f"{len(fold_origs)} forecast origin(s)")
        model = fit_at(panel, cutoff, project, exp)
        fit_stats = dict(getattr(model, "stats", {}) or {})
        for origin in fold_origs:
            pred, st = forecast_at(panel, origin, project, exp, plans, model=model)
            truth = panel[(panel[TS] > origin) & (panel[TS] <= origin + H)]
            truth = truth[[SERIES, TS, Y, STOCKOUT]].assign(**{SERIES: truth[SERIES].astype(str)})
            pred = pred.merge(truth, on=[SERIES, TS], how="left").rename(columns={Y: "y_true"})
            scale = mase_scale(panel[panel[TS] <= origin], project.season_length)
            scale.index = scale.index.astype(str)
            pred["mase_scale"] = pred[SERIES].map(scale).astype(np.float32)
            pred.insert(0, "fold", k)
            pred.insert(1, "cutoff", cutoff)
            pred.insert(2, "origin", origin)
            pred["model_age_days"] = np.int32((origin - cutoff).days)
            frames.append(pred)
            stats.append({"fold": k, "cutoff": str(cutoff.date()), "origin": str(origin.date()),
                          **fit_stats, **st})
    return pd.concat(frames, ignore_index=True), stats
