"""Project and experiment configuration.

`project.yaml` is the fixed policy: data columns, covariate classes, horizon,
backtest folds, metrics. It is identical across experiments, so results are
comparable. An experiment config changes exactly one thing (the model or its
params) and states the hypothesis it tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml

POLICIES = ("actual", "plan", "carry_forward")


@dataclass
class ProjectConfig:
    name: str = "forecast-fm"
    data_path: str = "data/raw/sales_sample.parquet"
    timestamp_col: str = "date"
    target_col: str = "units"
    series_id_cols: list[str] = field(default_factory=lambda: ["sku"])
    freq: str = "D"

    # covariate classes (see project.yaml for the rules)
    known_covariate_cols: list[str] = field(default_factory=list)
    past_covariate_cols: list[str] = field(default_factory=list)
    static_cols: list[str] = field(default_factory=list)
    covariate_types: dict[str, str] = field(default_factory=dict)  # col -> numeric|categorical

    # demand classes: a user label column (static), else computed ADI/CV2
    demand_label_col: str | None = None
    # rows where this column is 0/False are stockouts: target is censored,
    # so it is hidden from the model and excluded from scoring
    in_stock_col: str | None = None
    # grid end per series: "global" (zeros until the panel's last date) or
    # "series" (stop at the series' own last row)
    grid_end: str = "global"
    # how covariates are filled on dates with no raw row: ffill | zero | nan.
    # Default: known -> ffill (a price persists), past -> nan (unobserved).
    covariate_fill: dict[str, str] = field(default_factory=dict)

    covariate_eval_policy: dict[str, str] = field(default_factory=dict)
    planned_covariates_path: str | None = None

    horizon: int = 90
    horizon_buckets: list[int] = field(default_factory=lambda: [35, 90])
    season_length: int = 7

    n_folds: int = 4
    fold_step: int = 91
    min_train_periods: int = 730
    holdout_folds: int = 1

    quantiles: list[float] = field(default_factory=lambda: [0.1, 0.5, 0.9, 0.95])
    primary_metric: str = "wape"
    # relative change in the primary metric below which a verdict is
    # "inconclusive"
    verdict_threshold: float = 0.01
    success_criteria: dict = field(default_factory=dict)

    def __post_init__(self):
        if isinstance(self.series_id_cols, str):
            self.series_id_cols = [self.series_id_cols]
        overlap = set(self.known_covariate_cols) & set(self.past_covariate_cols)
        if overlap:
            raise ValueError(f"columns cannot be both known and past covariates: {sorted(overlap)}")
        for c, p in self.covariate_eval_policy.items():
            if p not in POLICIES:
                raise ValueError(f"covariate_eval_policy[{c}]={p!r}; valid: {POLICIES}")
            if c not in self.known_covariate_cols:
                raise ValueError(f"covariate_eval_policy names {c!r}, which is not a known covariate")
        if "plan" in self.covariate_eval_policy.values() and not self.planned_covariates_path:
            raise ValueError("covariate_eval_policy uses 'plan' but planned_covariates_path is unset")
        if self.grid_end not in ("global", "series"):
            raise ValueError("grid_end must be 'global' or 'series'")
        if sorted(self.horizon_buckets) != self.horizon_buckets or self.horizon_buckets[-1] != self.horizon:
            raise ValueError("horizon_buckets must be ascending and end at horizon")
        if 0.5 not in self.quantiles:
            self.quantiles = sorted(set(self.quantiles) | {0.5})

        for c, f in self.covariate_fill.items():
            if f not in ("ffill", "zero", "nan"):
                raise ValueError(f"covariate_fill[{c}]={f!r}; valid: ffill, zero, nan")

    def policy(self, col: str) -> str:
        return self.covariate_eval_policy.get(col, "actual")

    def fill(self, col: str) -> str:
        return self.covariate_fill.get(col, "ffill" if col in self.known_covariate_cols else "nan")

    @property
    def covariate_cols(self) -> list[str]:
        return [*self.known_covariate_cols, *self.past_covariate_cols]

    @property
    def raw_columns(self) -> list[str]:
        cols = [self.timestamp_col, self.target_col, *self.series_id_cols,
                *self.known_covariate_cols, *self.past_covariate_cols, *self.static_cols]
        for c in (self.demand_label_col, self.in_stock_col):
            if c:
                cols.append(c)
        return list(dict.fromkeys(cols))


@dataclass
class ExperimentConfig:
    name: str
    hypothesis: str
    model: str
    model_params: dict = field(default_factory=dict)
    mechanism: str = ""
    rationale: str = ""
    based_on: str | None = None


def _from_dict(cls, d: dict):
    names = {f.name for f in fields(cls)}
    unknown = set(d) - names
    if unknown:
        raise ValueError(f"{cls.__name__}: unknown keys {sorted(unknown)}")
    return cls(**d)


def load_project(path: str | Path = "project.yaml") -> ProjectConfig:
    return _from_dict(ProjectConfig, yaml.safe_load(Path(path).read_text()) or {})


def load_experiment(path: str | Path) -> ExperimentConfig:
    return _from_dict(ExperimentConfig, yaml.safe_load(Path(path).read_text()) or {})
