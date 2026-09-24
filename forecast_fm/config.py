"""Project and experiment configuration.

`project.yaml` is the fixed policy: data columns, covariate classes, horizon,
backtest folds, metrics. It is identical across experiments, so results are
comparable. An experiment config changes exactly one thing (the model or its
params) and states the hypothesis it tests.

Loading `project.yaml` supports, in this order:

1. `extends: base.yaml`: start from another file (relative to this one) and
   override it. Chains are allowed.
2. Sections: any of SECTIONS (`data:`, `covariates:`, `backtest:` ...) may
   group keys; they are flattened, so a key means the same in or out of a
   section.
3. `profiles:` named overlays picked with `--profile NAME` (or the
   FORECAST_FM_PROFILE env var), e.g. `sample` vs `full`, `mac` vs `cuda`.
4. `--set key=value` overrides from the command line; values are parsed as
   YAML, dotted keys reach into mappings (`covariate_eval_policy.price=plan`).
5. `${VAR}` / `${VAR:-default}` environment variables inside any string.

Unknown keys are errors, with a did-you-mean suggestion.
"""

from __future__ import annotations

import copy
import difflib
import os
import re
from dataclasses import MISSING, asdict, dataclass, field, fields
from pathlib import Path

import pandas as pd
import yaml

POLICIES = ("actual", "plan", "carry_forward")
FILLS = ("ffill", "zero", "nan")
DUPLICATES = ("error", "sum", "mean", "max", "first", "last")
NEGATIVE = ("keep", "clip", "nan", "error")
MISSING_TARGET = ("zero", "nan")
PRIMARY_METRICS = ("wape", "mase", "wql")
SECTIONS = ("data", "columns", "filters", "preprocessing", "covariates", "plans", "task",
            "backtest", "evaluation", "paths")
PROFILE_ENV = "FORECAST_FM_PROFILE"


@dataclass
class ProjectConfig:
    name: str = "forecast-fm"

    # --- data ----------------------------------------------------------------
    # a file, a directory of parquet files, a glob ("data/raw/part-*.parquet"),
    # or a list of any of those
    data_path: str | list[str] = "data/raw/sales_sample.parquet"
    timestamp_col: str = "date"
    target_col: str = "units"
    series_id_cols: list[str] = field(default_factory=lambda: ["sku"])
    freq: str = "D"

    # --- row / series filters (applied to raw rows, before the grid) ---------
    start_date: str | None = None          # drop rows before this date
    end_date: str | None = None            # drop rows after this date
    row_filter: str | None = None          # pandas query, e.g. "channel == 'web'"
    min_history_days: int = 0              # drop series with a shorter grid

    # --- preprocessing ---------------------------------------------------------
    # new raw columns from pandas expressions over existing ones, e.g.
    # {discount_pct: "1 - price / regular_price"}; evaluated in order, so a
    # derived column may use an earlier one. Also applied to plan snapshots.
    derived_columns: dict[str, str] = field(default_factory=dict)
    duplicates: str = "error"              # (series, date) repeats: error | sum | mean | max | first | last
    negative_target: str = "keep"          # keep | clip (to 0) | nan | error
    missing_target: str = "zero"           # demand on days with no raw row: zero | nan
    # grid end per series: "global" (the panel's last date) or "series" (its
    # own last row)
    grid_end: str = "global"

    # --- covariates ------------------------------------------------------------
    known_covariate_cols: list[str] = field(default_factory=list)
    past_covariate_cols: list[str] = field(default_factory=list)
    static_cols: list[str] = field(default_factory=list)
    covariate_types: dict[str, str] = field(default_factory=dict)  # col -> numeric | categorical
    # how covariates are filled on dates with no raw row: ffill | zero | nan.
    # Default: known -> ffill (a price persists), past -> nan (unobserved).
    covariate_fill: dict[str, str] = field(default_factory=dict)
    # demand classes: a user label column (static), else computed ADI/CV2
    demand_label_col: str | None = None
    # stockouts: the target is censored, so hidden from the model and not
    # scored. Either a column where 0/False = stocked out, or a pandas
    # expression over raw columns that is True on stockout rows.
    in_stock_col: str | None = None
    stockout_expr: str | None = None

    # --- plans (horizon values of known covariates) ----------------------------
    covariate_eval_policy: dict[str, str] = field(default_factory=dict)
    planned_covariates_path: str | list[str] | None = None
    plan_as_of_col: str = "as_of"
    plan_timestamp_col: str | None = None      # default: timestamp_col
    plan_columns: dict[str, str] = field(default_factory=dict)  # plan file name -> project name
    min_plan_coverage: float = 0.99            # warn below this share per covariate
    plan_max_age_days: int | None = None       # warn when the latest snapshot is older

    # --- task ------------------------------------------------------------------
    horizon: int = 90
    horizon_buckets: list[int] = field(default_factory=lambda: [35, 90])
    season_length: int = 7

    # --- backtest ----------------------------------------------------------------
    n_folds: int = 4
    fold_step: int = 91
    min_train_periods: int = 730
    holdout_folds: int = 1
    cutoffs: list[str] | None = None           # explicit validation cutoffs (override the rule)
    holdout_cutoffs: list[str] | None = None   # explicit holdout cutoffs

    # --- evaluation ------------------------------------------------------------
    quantiles: list[float] = field(default_factory=lambda: [0.1, 0.5, 0.9, 0.95])
    service_level: float | None = None         # decision quantile; added to quantiles
    primary_metric: str = "wape"
    verdict_threshold: float = 0.01            # relative change below which a verdict is inconclusive
    slice_cols: list[str] = field(default_factory=list)  # statics to break metrics down by
    success_criteria: dict = field(default_factory=dict)

    # --- models ------------------------------------------------------------------
    # defaults per model family, under each experiment's model_params
    # (the experiment wins), e.g. {chronos2: {device: mps, batch_size: 512}}
    model_defaults: dict[str, dict] = field(default_factory=dict)

    # --- paths -------------------------------------------------------------------
    reports_dir: str = "reports"
    models_dir: str = "models"

    def __post_init__(self):
        if isinstance(self.series_id_cols, str):
            self.series_id_cols = [self.series_id_cols]
        errors = []

        def one_of(name, value, valid):
            if value not in valid:
                errors.append(f"{name}={value!r}; valid: {', '.join(valid)}")

        if self.freq != "D":
            errors.append(f"freq={self.freq!r}: only daily data (D) is supported")
        one_of("duplicates", self.duplicates, DUPLICATES)
        one_of("negative_target", self.negative_target, NEGATIVE)
        one_of("missing_target", self.missing_target, MISSING_TARGET)
        one_of("grid_end", self.grid_end, ("global", "series"))
        one_of("primary_metric", self.primary_metric, PRIMARY_METRICS)

        known, past = set(self.known_covariate_cols), set(self.past_covariate_cols)
        static = set(self.static_cols)
        for a, b, sa, sb in ((known, past, "known", "past"), (known, static, "known", "static"),
                             (past, static, "past", "static")):
            if a & b:
                errors.append(f"columns cannot be both {sa} and {sb}: {sorted(a & b)}")
        for c, p in self.covariate_eval_policy.items():
            one_of(f"covariate_eval_policy[{c}]", p, POLICIES)
            if c not in known:
                errors.append(f"covariate_eval_policy names {c!r}, which is not a known covariate")
        if "plan" in self.covariate_eval_policy.values() and not self.planned_covariates_path:
            errors.append("covariate_eval_policy uses 'plan' but planned_covariates_path is unset")
        for c, f in self.covariate_fill.items():
            one_of(f"covariate_fill[{c}]", f, FILLS)
        for c, t in self.covariate_types.items():
            one_of(f"covariate_types[{c}]", t, ("numeric", "categorical"))
        bad_slices = [c for c in self.slice_cols if c not in static and c != self.demand_label_col]
        if bad_slices:
            errors.append(f"slice_cols {bad_slices} must be static_cols (one value per series)")
        if self.in_stock_col and self.stockout_expr:
            errors.append("set in_stock_col or stockout_expr, not both")

        if self.horizon < 1:
            errors.append("horizon must be >= 1")
        if (sorted(self.horizon_buckets) != list(self.horizon_buckets) or not self.horizon_buckets
                or self.horizon_buckets[-1] != self.horizon):
            errors.append("horizon_buckets must be ascending and end at horizon")
        if self.fold_step < 1 or self.n_folds < 1 or self.holdout_folds < 0:
            errors.append("fold_step and n_folds must be >= 1, holdout_folds >= 0")
        for name in ("start_date", "end_date"):
            _check_date(getattr(self, name), name, errors)
        for name in ("cutoffs", "holdout_cutoffs"):
            for d in getattr(self, name) or []:
                _check_date(d, name, errors)
        if not 0 < self.min_plan_coverage <= 1:
            errors.append("min_plan_coverage must be in (0, 1]")

        qs = list(self.quantiles)
        if self.service_level is not None:
            if not 0 < self.service_level < 1:
                errors.append("service_level must be in (0, 1)")
            qs.append(self.service_level)
        if any(not 0 < q < 1 for q in qs):
            errors.append("quantiles must be in (0, 1)")
        self.quantiles = sorted(set(qs) | {0.5})
        if errors:
            raise ValueError("project config:\n  - " + "\n  - ".join(errors))

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

    def model_params(self, model: str, params: dict | None) -> dict:
        """Experiment params over this project's defaults for the family."""
        merged = deep_merge(self.model_defaults.get(model, {}), params or {})
        if model == "chronos2":
            merged.setdefault("cache_dir", str(Path(self.reports_dir) / "chronos2_ft"))
        return merged

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ExperimentConfig:
    name: str
    hypothesis: str
    model: str
    model_params: dict = field(default_factory=dict)
    mechanism: str = ""
    rationale: str = ""
    based_on: str | None = None


# --- loading ------------------------------------------------------------------


def _check_date(value, name: str, errors: list) -> None:
    if value is None:
        return
    try:
        pd.Timestamp(value)
    except (ValueError, TypeError):
        errors.append(f"{name}: {value!r} is not a date")


def deep_merge(base: dict, over: dict) -> dict:
    """`over` wins; mappings merge recursively, everything else is replaced."""
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def flatten_sections(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if k in SECTIONS and isinstance(v, dict):
            for kk, vv in v.items():
                if kk in out:
                    raise ValueError(f"key {kk!r} is set twice (section {k!r} and elsewhere)")
                out[kk] = vv
        else:
            if k in out:
                raise ValueError(f"key {k!r} is set twice")
            out[k] = v
    return out


_ENV = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def interpolate_env(value):
    if isinstance(value, str):
        def repl(m):
            name, default = m.group(1), m.group(2)
            if name in os.environ:
                return os.environ[name]
            if default is not None:
                return default
            raise ValueError(f"environment variable ${{{name}}} is not set (use ${{{name}:-default}})")
        return _ENV.sub(repl, value)
    if isinstance(value, dict):
        return {k: interpolate_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [interpolate_env(v) for v in value]
    return value


def _read_layers(path: Path, seen: tuple = ()) -> dict:
    path = path.resolve()
    if path in seen:
        raise ValueError(f"extends cycle: {' -> '.join(str(p) for p in (*seen, path))}")
    d = yaml.safe_load(path.read_text()) or {}
    parent = d.pop("extends", None)
    profiles = d.pop("profiles", {}) or {}
    d = flatten_sections(d)
    if parent:
        base = _read_layers(path.parent / parent, (*seen, path))
        profiles = deep_merge(base.pop("profiles", {}), profiles)
        d = deep_merge(base, d)
    d["profiles"] = profiles
    return d


def parse_override(expr: str) -> tuple[list[str], object]:
    if "=" not in expr:
        raise ValueError(f"--set expects key=value, got {expr!r}")
    key, raw = expr.split("=", 1)
    return key.strip().split("."), yaml.safe_load(raw) if raw.strip() else None


def apply_overrides(d: dict, overrides: list[str]) -> dict:
    d = copy.deepcopy(d)
    for expr in overrides or []:
        keys, value = parse_override(expr)
        node = d
        for k in keys[:-1]:
            node = node.setdefault(k, {})
            if not isinstance(node, dict):
                raise ValueError(f"--set {expr!r}: {k!r} is not a mapping")
        node[keys[-1]] = value
    return d


def _from_dict(cls, d: dict, source: str = ""):
    names = [f.name for f in fields(cls)]
    unknown = [k for k in d if k not in names]
    if unknown:
        hints = []
        for k in unknown:
            close = difflib.get_close_matches(k, names + list(SECTIONS), n=1)
            hints.append(f"{k!r}" + (f" (did you mean {close[0]!r}?)" if close else ""))
        raise ValueError(f"{source or cls.__name__}: unknown keys {', '.join(hints)}")
    # `key:` with every entry commented out loads as None: use the default
    empty = {f.name for f in fields(cls) if f.default_factory is not MISSING}
    return cls(**{k: v for k, v in d.items() if not (v is None and k in empty)})


def resolve_project_dict(path: str | Path = "project.yaml", profile: str | None = None,
                         overrides: list[str] | None = None) -> dict:
    d = _read_layers(Path(path))
    profiles = d.pop("profiles")
    profile = profile or os.environ.get(PROFILE_ENV) or None
    if profile:
        if profile not in profiles:
            raise ValueError(f"profile {profile!r} not in {path}; defined: {sorted(profiles) or 'none'}")
        d = deep_merge(d, flatten_sections(profiles[profile] or {}))
    d = apply_overrides(d, overrides or [])
    return interpolate_env(d)


def load_project(path: str | Path = "project.yaml", profile: str | None = None,
                 overrides: list[str] | None = None) -> ProjectConfig:
    return _from_dict(ProjectConfig, resolve_project_dict(path, profile, overrides), str(path))


def load_experiment(path: str | Path) -> ExperimentConfig:
    d = interpolate_env(yaml.safe_load(Path(path).read_text()) or {})
    return _from_dict(ExperimentConfig, d, str(path))
