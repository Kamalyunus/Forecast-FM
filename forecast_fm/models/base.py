"""The model interface every family implements.

A model sees only `history` (panel rows with ts <= cutoff) and the `future`
frame (series, ts, horizon, known covariates). It never sees realized
targets or past-only covariates after the cutoff: the backtest does not hand
them over.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import ProjectConfig

Forecast = tuple[np.ndarray, dict[float, np.ndarray] | None]


class Forecaster:
    PARAM_KEYS: frozenset[str] = frozenset()

    def __init__(self, params: dict | None = None):
        self.params = dict(params or {})
        bad = set(self.params) - self.PARAM_KEYS
        if bad:
            raise ValueError(f"{type(self).__name__}: unknown params {sorted(bad)}; "
                             f"valid: {sorted(self.PARAM_KEYS)}")

    def fit(self, history: pd.DataFrame, project: ProjectConfig, cutoff: pd.Timestamp) -> None:
        """Called once per fold with that fold's history only."""

    def predict(self, history: pd.DataFrame, future: pd.DataFrame, project: ProjectConfig) -> Forecast:
        """Return (point forecast aligned to `future` rows, {quantile: values}
        or None for point-only models)."""
        raise NotImplementedError
