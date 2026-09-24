from __future__ import annotations

import numpy as np
import pandas as pd

from forecast_fm.config import ExperimentConfig, ProjectConfig
from forecast_fm.data import build_panel, prepare_raw


def make_raw(n_series: int = 6, n_days: int = 200, start: str = "2024-01-01", seed: int = 0,
             drop_frac: float = 0.0) -> pd.DataFrame:
    """Synthetic daily demand in the user's column names: weekly season,
    promo lift, a price, sessions, stockouts, statics."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=n_days, freq="D")
    rows = []
    for i in range(n_series):
        level = 5 + 3 * i
        promo = (rng.random(n_days) < 0.1).astype(int)
        price = np.round(10 * (1 - 0.2 * promo), 2)
        season = 1 + 0.3 * np.sin(2 * np.pi * np.arange(n_days) / 7)
        y = rng.poisson(level * season * (1 + promo)).astype(float)
        if i % 3 == 2:  # intermittent series
            y = y * (rng.random(n_days) < 0.2)
        in_stock = np.ones(n_days, dtype=int)
        in_stock[rng.random(n_days) < 0.03] = 0
        rows.append(pd.DataFrame({
            "date": dates, "sku": f"S{i}", "units": y, "price": price, "promo_flag": promo,
            "promo_type": np.where(promo > 0, "pct_off", "none"),
            "sessions": rng.normal(100, 5, n_days).round(1), "in_stock": in_stock,
            "category": "A" if i < n_series // 2 else "B",
            "demand_label": "intermittent" if i % 3 == 2 else "BAU",
        }))
    raw = pd.concat(rows, ignore_index=True)
    if drop_frac:
        keep = rng.random(len(raw)) >= drop_frac
        first = ~raw.duplicated("sku")  # keep each series' first row
        raw = raw[keep | first].reset_index(drop=True)
    return raw


def make_project(**kw) -> ProjectConfig:
    base = dict(
        name="test", data_path="unused", timestamp_col="date", target_col="units",
        series_id_cols=["sku"], known_covariate_cols=["price", "promo_flag", "promo_type"],
        past_covariate_cols=["sessions", "in_stock"], static_cols=["category"],
        demand_label_col="demand_label", in_stock_col="in_stock",
        covariate_fill={"promo_flag": "zero", "promo_type": "nan"},
        horizon=14, horizon_buckets=[7, 14], season_length=7,
        n_folds=2, fold_step=14, min_train_periods=60, holdout_folds=1,
        quantiles=[0.1, 0.5, 0.9],
    )
    base.update(kw)
    return ProjectConfig(**base)


def make_exp(model: str = "seasonal_naive", **params) -> ExperimentConfig:
    return ExperimentConfig(name="t", hypothesis="h", model=model, model_params=params)


def panel_from(raw: pd.DataFrame, project: ProjectConfig) -> pd.DataFrame:
    return build_panel(prepare_raw(raw, project), project)
