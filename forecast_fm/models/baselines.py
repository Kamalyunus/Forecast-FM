"""Reference models. Vectorized across series; no per-series loops."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..data import SERIES, STOCKOUT, TS, Y
from ..plans import HORIZON
from .base import Forecast, Forecaster


def _lookup(history: pd.DataFrame, keys: pd.DataFrame) -> np.ndarray:
    """y at (series_id, ts) for each key row; NaN where absent or stocked out."""
    h = history[[SERIES, TS, Y, STOCKOUT]]
    m = keys.merge(h, on=[SERIES, TS], how="left")
    y = m[Y].to_numpy(dtype=np.float32, copy=True)
    y[m[STOCKOUT].fillna(False).to_numpy(dtype=bool)] = np.nan
    return y


def _series_mean(history: pd.DataFrame, future: pd.DataFrame) -> np.ndarray:
    h = history[~history[STOCKOUT]]
    mean = h.groupby(SERIES, observed=True)[Y].mean()
    return future[SERIES].map(mean).astype(float).fillna(0.0).to_numpy()


class Naive(Forecaster):
    """Last in-stock observation, held flat."""

    def predict(self, history, future, project) -> Forecast:
        h = history[~history[STOCKOUT]].sort_values(TS)
        last = h.groupby(SERIES, observed=True)[Y].last()
        return future[SERIES].map(last).astype(float).fillna(0.0).to_numpy(), None


class SeasonalNaive(Forecaster):
    """The value one season earlier: y(cutoff - m + 1 + (h-1) mod m). Falls
    back to the series mean where that day is missing or stocked out."""

    PARAM_KEYS = frozenset({"season_length"})

    def predict(self, history, future, project) -> Forecast:
        m = int(self.params.get("season_length", project.season_length))
        cutoff = future[TS].min() - pd.Timedelta(days=1)
        h = future[HORIZON].to_numpy(dtype=np.int64)
        src = cutoff - pd.Timedelta(days=m - 1) + pd.to_timedelta((h - 1) % m, unit="D")
        y = _lookup(history, pd.DataFrame({SERIES: future[SERIES].to_numpy(), TS: src}))
        fb = _series_mean(history, future)
        return np.where(np.isnan(y), fb, y), None


class Croston(Forecaster):
    """Croston with the Syntetos-Boylan bias correction (SBA), for
    intermittent demand. Runs the recursion over time on a (series x window)
    matrix, so the loop is over days, not series."""

    PARAM_KEYS = frozenset({"alpha", "window"})

    def predict(self, history, future, project) -> Forecast:
        alpha = float(self.params.get("alpha", 0.1))
        window = int(self.params.get("window", 365))
        cutoff = history[TS].max()
        h = history[history[TS] > cutoff - pd.Timedelta(days=window)]
        y = h[Y].where(~h[STOCKOUT])
        mat = (pd.DataFrame({SERIES: h[SERIES], TS: h[TS], Y: y})
               .pivot_table(index=SERIES, columns=TS, values=Y, aggfunc="sum", dropna=False,
                            observed=True)
               .reindex(columns=pd.date_range(cutoff - pd.Timedelta(days=window - 1), cutoff)))
        Y_ = mat.to_numpy(dtype=np.float64)
        n = Y_.shape[0]
        z = np.full(n, np.nan)  # demand size
        p = np.full(n, np.nan)  # inter-demand interval
        q = np.ones(n)          # periods since last demand
        for t in range(Y_.shape[1]):
            v = Y_[:, t]
            hit = np.nan_to_num(v) > 0
            first = hit & np.isnan(z)
            z = np.where(first, v, z)
            p = np.where(first, q, p)
            upd = hit & ~first
            z = np.where(upd, z + alpha * (v - z), z)
            p = np.where(upd, p + alpha * (q - p), p)
            q = np.where(hit, 1.0, q + (~np.isnan(v)))
        rate = np.nan_to_num((1 - alpha / 2) * z / p)
        fc = pd.Series(rate, index=mat.index)
        return future[SERIES].map(fc).astype(float).fillna(0.0).to_numpy(), None
