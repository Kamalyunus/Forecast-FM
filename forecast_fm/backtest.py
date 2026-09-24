"""Rolling-origin backtest: the single evaluation harness.

At each cutoff the model gets `history` (ts <= cutoff) and the future frame
(series, ts, horizon, known covariates set by covariate_eval_policy). Target
values and past covariates after the cutoff are never passed to the model;
they are joined back only for scoring, after predict returns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ExperimentConfig, ProjectConfig
from .data import SERIES, STOCKOUT, TS, Y
from .folds import fold_cutoffs
from .models import create_model
from .plans import HORIZON, future_frame

Q_PREFIX = "q_"


def qcol(q: float) -> str:
    return f"{Q_PREFIX}{q:g}"


def mase_scale(history: pd.DataFrame, m: int) -> pd.Series:
    """Per-series in-sample MAE of the seasonal naive forecast (lag m) on
    in-stock days: the MASE denominator."""
    h = history[[SERIES, Y, STOCKOUT]]
    y = h[Y].where(~h[STOCKOUT])
    lag = y.groupby(h[SERIES], observed=True).shift(m)
    return (y - lag).abs().groupby(h[SERIES], observed=True).mean()


def forecast_at(panel: pd.DataFrame, cutoff: pd.Timestamp, project: ProjectConfig,
                exp: ExperimentConfig, plans: pd.DataFrame | None, production: bool = False
                ) -> tuple[pd.DataFrame, dict]:
    """One origin: fit on history <= cutoff, forecast horizon 1..H."""
    cutoff = pd.Timestamp(cutoff)
    history = panel[panel[TS] <= cutoff]
    actuals = None
    if not production:
        known = [c for c in project.known_covariate_cols if c in panel.columns]
        window = panel[(panel[TS] > cutoff) & (panel[TS] <= cutoff + pd.Timedelta(days=project.horizon))]
        actuals = window[[SERIES, TS, *known]]  # declared-known columns only
    future = future_frame(history, cutoff, project, plans, actuals=actuals, production=production)
    model = create_model(exp.model, project.model_params(exp.model, exp.model_params))
    model.fit(history, project, cutoff)
    point, qd = model.predict(history, future, project)
    out = future[[SERIES, TS, HORIZON]].copy()
    out["y_pred"] = np.asarray(point, dtype=np.float32)
    if qd:
        for q, v in qd.items():
            out[qcol(q)] = np.asarray(v, dtype=np.float32)
    return out, dict(getattr(model, "stats", {}) or {})


def backtest(panel: pd.DataFrame, project: ProjectConfig, exp: ExperimentConfig,
             plans: pd.DataFrame | None = None, holdout: bool = False) -> tuple[pd.DataFrame, list[dict]]:
    cutoffs = fold_cutoffs(panel[TS].min(), panel[TS].max(), project, holdout=holdout)
    frames, stats = [], []
    for k, cutoff in enumerate(cutoffs):
        print(f"[backtest] fold {k} cutoff {cutoff.date()} ({exp.model})")
        pred, st = forecast_at(panel, cutoff, project, exp, plans)
        truth = panel[(panel[TS] > cutoff) & (panel[TS] <= cutoff + pd.Timedelta(days=project.horizon))]
        pred = pred.merge(truth[[SERIES, TS, Y, STOCKOUT]], on=[SERIES, TS], how="left")
        pred = pred.rename(columns={Y: "y_true"})
        scale = mase_scale(panel[panel[TS] <= cutoff], project.season_length)
        pred["mase_scale"] = pred[SERIES].map(scale).astype(np.float32)
        pred.insert(0, "fold", k)
        pred.insert(1, "cutoff", cutoff)
        frames.append(pred)
        stats.append({"fold": k, "cutoff": str(cutoff.date()), **st})
    return pd.concat(frames, ignore_index=True), stats
