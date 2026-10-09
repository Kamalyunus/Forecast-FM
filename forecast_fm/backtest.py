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


def first_dates(panel: pd.DataFrame) -> pd.Series:
    """First row date per series, computed once per run: every origin's
    established / short / new split derives from it."""
    return panel.groupby(SERIES, observed=True)[TS].min()


def _split_by_history(history: pd.DataFrame, project: ProjectConfig, cutoff: pd.Timestamp,
                      first: pd.Series | None = None):
    """(established series ids, short-history frame for cold start)."""
    if first is None:
        first = first_dates(history)
    else:
        first = first[first <= cutoff]
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


def _new_in_window(panel: pd.DataFrame, project: ProjectConfig, cutoff: pd.Timestamp,
                   first: pd.Series | None = None) -> pd.DataFrame:
    """Backtest: series first seen within the horizon, treated as planned
    launches (launch date and statics known at the cutoff)."""
    if first is None:
        first = first_dates(panel)
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


def fit_at(panel: pd.DataFrame, cutoff: pd.Timestamp, project: ProjectConfig, exp: ExperimentConfig,
           first: pd.Series | None = None):
    """Create the experiment's model and fit it on history <= cutoff (the
    series established at the cutoff)."""
    cutoff = pd.Timestamp(cutoff)
    history = panel[panel[TS] <= cutoff]
    established, _ = _split_by_history(history, project, cutoff, first)
    model = create_model(exp.model, project.model_params(exp.model, exp.model_params))
    model.fit(history[history[SERIES].isin(established)], project, cutoff)
    return model


def forecast_at(panel: pd.DataFrame, cutoff: pd.Timestamp, project: ProjectConfig,
                exp: ExperimentConfig, plans: pd.DataFrame | None, production: bool = False,
                new_series: pd.DataFrame | None = None, model=None,
                data_start: pd.Timestamp | None = None,
                first: pd.Series | None = None) -> tuple[pd.DataFrame, dict]:
    """One origin: fit on history <= cutoff, forecast horizon 1..H for every
    series (model for established ones, cold start for the rest). A series
    that gets no forecast from either route (cold_start.method: none, no
    analog profile) is still returned, with y_pred NaN, so the metrics count
    it under n_missing_pred instead of silently dropping it.

    `model`: an already-fitted model (fit at an earlier retrain date); it then
    only reads history <= this origin as context. `new_series` (production):
    upcoming series with SERIES, launch_date and statics. In a backtest they
    come from the panel (first seen in-window). `data_start`: the dataset's
    first date (for launch profiles); `first`: precomputed first dates."""
    cutoff = pd.Timestamp(cutoff)
    history = panel[panel[TS] <= cutoff]
    established, short = _split_by_history(history, project, cutoff, first)
    model_history = history[history[SERIES].isin(established)] if len(short) else history
    if production:
        new = new_series if new_series is not None else pd.DataFrame(columns=[SERIES, cold_start.LAUNCH])
        new = new.assign(hist_age=0, hist_sum=0.0)
    else:
        new = _new_in_window(panel, project, cutoff, first)

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
        cfc, cstats = cold_start.forecast(cold, history, project, cutoff, data_start)
        if len(cfc):
            kinds = pd.Series(np.where(cold["hist_age"].to_numpy() > 0, "short_history", "new"),
                              index=cold[SERIES].astype(str).to_numpy())
            cfc[LIFECYCLE] = cfc[SERIES].map(kinds).to_numpy()
            keep = [c for c in cfc.columns if not c.startswith(Q_PREFIX) or not frames
                    or c in frames[0].columns]
            frames.append(cfc[keep].astype({"y_pred": np.float32}))
        stats["cold_start"] = cstats
    # every series the origin owes a forecast, with NaN where no route produced one
    expected = pd.Index(established.astype(str))
    if len(cold):
        expected = expected.union(pd.Index(cold[SERIES].astype(str)))
    have = set(pd.concat([f[SERIES].astype(str) for f in frames])) if frames else set()
    missing = [sid for sid in expected if sid not in have]
    if missing:
        kinds = {} if not len(cold) else dict(zip(
            cold[SERIES].astype(str), np.where(cold["hist_age"].to_numpy() > 0, "short_history", "new"),
            strict=True))
        H = project.horizon
        gap = pd.DataFrame({SERIES: np.repeat(np.array(missing, dtype=object), H),
                            TS: np.tile(cutoff + pd.to_timedelta(np.arange(1, H + 1), unit="D"),
                                        len(missing)),
                            HORIZON: np.tile(np.arange(1, H + 1), len(missing))})
        gap["y_pred"] = np.float32(np.nan)
        gap[LIFECYCLE] = gap[SERIES].map(lambda s: kinds.get(s, "established"))
        frames.append(gap)
        stats["n_missing_forecast"] = len(missing)
    if not frames:
        return pd.DataFrame(columns=[SERIES, TS, HORIZON, "y_pred", LIFECYCLE]), stats
    return pd.concat(frames, ignore_index=True), stats


def backtest(panel: pd.DataFrame, project: ProjectConfig, exp: ExperimentConfig,
             plans: pd.DataFrame | None = None, holdout: bool = False,
             cutoffs: list[pd.Timestamp] | None = None,
             origins: list[list[pd.Timestamp]] | None = None,
             data_start: pd.Timestamp | None = None) -> tuple[pd.DataFrame, list[dict]]:
    """Per fold: fit the model once at the fold's cutoff (its retrain date),
    then forecast from each of the fold's origins (`origin_step_days`), the
    model reading history <= each origin. Scored on the H days after each
    origin. The MASE denominator (seasonal-naive in-sample MAE) is computed
    once per fold at its retrain date. `data_start`: the whole dataset's first
    date when `panel` is one shard of it."""
    first, last = panel[TS].min(), panel[TS].max()
    data_start = pd.Timestamp(data_start) if data_start is not None else first
    first_all = first_dates(panel)
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
        model = fit_at(panel, cutoff, project, exp, first_all)
        fit_stats = dict(getattr(model, "stats", {}) or {})
        scale = mase_scale(panel[panel[TS] <= cutoff], project.season_length)
        scale.index = scale.index.astype(str)
        after = panel[panel[TS] > cutoff]  # every origin's truth lies here
        for origin in fold_origs:
            pred, st = forecast_at(panel, origin, project, exp, plans, model=model,
                                   data_start=data_start, first=first_all)
            truth = after[(after[TS] > origin) & (after[TS] <= origin + H)]
            truth = truth[[SERIES, TS, Y, STOCKOUT]].assign(**{SERIES: truth[SERIES].astype(str)})
            pred = pred.merge(truth, on=[SERIES, TS], how="left").rename(columns={Y: "y_true"})
            pred["mase_scale"] = pred[SERIES].map(scale).astype(np.float32)
            pred.insert(0, "fold", k)
            pred.insert(1, "cutoff", cutoff)
            pred.insert(2, "origin", origin)
            pred["model_age_days"] = np.int32((origin - cutoff).days)
            frames.append(pred)
            stats.append({"fold": k, "cutoff": str(cutoff.date()), "origin": str(origin.date()),
                          **fit_stats, **st})
    return pd.concat(frames, ignore_index=True), stats
