"""Run a backtest or a production forecast, optionally sharded by series.

Sharding: series are split into N shards by a stable hash of the series id
(or of `shard_by` statics, so a cross-learning group stays together). Each
shard is loaded, forecast and released on its own, so memory is bounded by
one shard, not the catalog. Fold cutoffs and the grid end come from one
cheap streaming pass over the whole dataset, so every shard uses the same
dates.

    backtest   metrics are combined from per-shard additive sums: identical
               to an unsharded run for every series the model forecasts.
               Cold-start launch profiles pool analogs within a shard; set
               shard_by = cold_start.profile_cols to make those identical too
               (random shards of a large catalog hold plenty of analogs).
    forecast   the long daily file is written as one parquet part per shard
               (<out>/part-00003.parquet); read the whole directory with
               pandas.read_parquet(<out>). A shard whose part exists is
               skipped (resume after an interruption) unless force=True.
               Shards can also run as separate processes (--shard i).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import cold_start, metrics
from .backtest import LIFECYCLE, backtest, forecast_at
from .config import ExperimentConfig, ProjectConfig
from .data import ID_SEP, SERIES, TS, date_span, demand_classes, load_panel, make_series_id, shard_of
from .device import describe
from .folds import fold_cutoffs
from .plans import load_plans


def _span(project: ProjectConfig, data, n_shards: int, panel: pd.DataFrame | None):
    if n_shards == 1 and panel is not None:
        return panel[TS].min(), panel[TS].max()
    return date_span(project, data)


def _statics(panel: pd.DataFrame, project: ProjectConfig) -> pd.DataFrame | None:
    if not project.slice_cols:
        return None
    return panel.drop_duplicates(SERIES).set_index(SERIES)[project.slice_cols]


def run_backtest(project: ProjectConfig, exp: ExperimentConfig, data=None, shards: int = 1,
                 predictions_dir: str | Path | None = None) -> tuple[dict, pd.DataFrame | None]:
    """(result, predictions). Predictions are returned only unsharded; with
    `predictions_dir` every shard's predictions are written there."""
    params = project.model_params(exp.model, exp.model_params)
    if shards > 1 and params.get("fine_tune"):
        raise ValueError("a sharded backtest would fine-tune one model per shard: backtest "
                         "fine-tune recipes unsharded on the sample (then `finetune` once and "
                         "`forecast --shards N` with the saved checkpoint)")
    panel = load_panel(project, data) if shards == 1 else None
    first, last = _span(project, data, shards, panel)
    cutoffs = fold_cutoffs(first, last, project)
    parts, stats, kept = [], [], None
    for i in range(shards):
        t0 = time.perf_counter()
        if shards > 1:
            panel = load_panel(project, data, shard=(i, shards), end=last)
            if panel.empty:
                continue
            print(f"[shard {i + 1}/{shards}] {panel[SERIES].nunique():,} series")
        ids = set(panel[SERIES].astype(str).unique())
        plans = load_plans(project, cutoffs=cutoffs, series=ids)
        preds, st = backtest(panel, project, exp, plans, cutoffs=cutoffs)
        classes = demand_classes(panel, project, end=cutoffs[0])
        parts.append(metrics.partials(preds, project, classes, _statics(panel, project)))
        stats += [{"shard": i, **s} for s in st] if shards > 1 else st
        if predictions_dir:
            Path(predictions_dir).mkdir(parents=True, exist_ok=True)
            preds.to_parquet(Path(predictions_dir) / f"part-{i:05d}.parquet", index=False)
        if shards == 1:
            kept = preds
        del panel, preds, plans
        if shards > 1:
            print(f"[shard {i + 1}/{shards}] done in {time.perf_counter() - t0:.0f}s")
    if not parts:
        raise ValueError("no series in any shard")
    result = metrics.result(metrics.combine(parts), project)
    result.update(stats=stats, env=describe(), cutoffs=cutoffs, shards=shards)
    return result, kept


def new_series_shard(new: pd.DataFrame, project: ProjectConfig, shards: int) -> np.ndarray:
    """Shard of each upcoming series: by its shard_by statics when it has them
    (so it pools analogs with its group), else by id, which is also how its
    plan rows were routed."""
    by_id = shard_of(new[SERIES], shards)
    cols = project.shard_by
    if not cols or not all(c in new.columns for c in cols):
        return by_id
    has = new[cols].notna().all(axis=1).to_numpy()
    by_static = shard_of(make_series_id(new, cols), shards)
    return np.where(has, by_static, by_id)


def _split_ids(out: pd.DataFrame, project: ProjectConfig) -> pd.DataFrame:
    """Add the user's id columns back (sku, warehouse, ...) from series_id."""
    cols = project.series_id_cols
    if len(cols) == 1:
        out[cols[0]] = out[SERIES]
    else:
        parts = out[SERIES].str.split(ID_SEP, n=len(cols) - 1, expand=True)
        for j, c in enumerate(cols):
            out[c] = parts[j]
    return out


def run_forecast(project: ProjectConfig, exp: ExperimentConfig, data=None, shards: int = 1,
                 only: list[int] | None = None, out_dir: str | Path | None = None,
                 force: bool = False) -> Path:
    """Production forecast from the data's last date: the long daily file
    (one row per series x day x horizon step, with quantiles) for every
    series, including new ones (cold start), as parquet parts in out_dir."""
    panel = load_panel(project, data) if shards == 1 else None
    if shards == 1:
        origin, known_ids = panel[TS].max(), set(panel[SERIES].astype(str).unique())
    else:
        _, origin, known_ids = date_span(project, data, with_series=True)
    out_dir = Path(out_dir or Path(project.reports_dir) / f"forecast_{origin:%Y%m%d}")
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = only if only is not None else list(range(shards))
    for i in todo:
        part = out_dir / f"part-{i:05d}.parquet"
        if part.exists() and not force:
            print(f"[forecast] shard {i}: {part} exists, skipped (force to redo)")
            continue
        t0 = time.perf_counter()
        if shards > 1:
            panel = load_panel(project, data, shard=(i, shards), end=origin)
        ids = set(panel[SERIES].astype(str).unique()) if len(panel) else set()

        def in_shard(s: pd.Series, i=i, ids=ids) -> np.ndarray:
            if shards == 1:
                return np.ones(len(s), dtype=bool)
            # this shard's series, plus plan-only (new) series hashed here;
            # never a series whose history lives in another shard
            unseen = ~s.isin(known_ids).to_numpy()
            return s.isin(ids).to_numpy() | (unseen & (shard_of(s, shards) == i))

        plans = load_plans(project, cutoffs=[origin], series=in_shard)
        new = cold_start.production_new_series(project, panel, origin, plans)
        new = new[~new[SERIES].isin(known_ids)]
        if shards > 1 and len(new):
            new = new[new_series_shard(new, project, shards) == i]
        if panel.empty and not len(new):
            pred, stats = pd.DataFrame(columns=[SERIES, TS, "horizon", "y_pred", LIFECYCLE]), {}
        else:
            pred, stats = forecast_at(panel, origin, project, exp, plans, production=True,
                                      new_series=new)
        pred.insert(0, "origin", origin)
        pred["model"] = exp.name
        pred = _split_ids(pred, project)
        tmp = part.with_suffix(".tmp")
        pred.to_parquet(tmp, index=False)
        tmp.rename(part)  # a crash never leaves a half-written part
        counts = pred.drop_duplicates(SERIES)[LIFECYCLE].value_counts().to_dict()
        # "_"-prefixed files are ignored by parquet readers of the directory
        (out_dir / f"_part-{i:05d}.json").write_text(json.dumps(
            {"shard": i, "shards": shards, "series": counts, "rows": len(pred),
             "seconds": round(time.perf_counter() - t0, 1), "stats": stats}, indent=2, default=str))
        print(f"[forecast] shard {i + 1}/{shards}: {sum(counts.values()):,} series {counts} -> {part}")
        del panel, pred, plans

    done = sorted(out_dir.glob("part-*.parquet"))
    if len(done) == shards:
        (out_dir / "_manifest.json").write_text(json.dumps(
            {"origin": str(origin.date()), "experiment": exp.name, "model": exp.model,
             "model_params": project.model_params(exp.model, exp.model_params),
             "horizon": project.horizon, "quantiles": project.quantiles, "shards": shards,
             "parts": [p.name for p in done], "env": describe()}, indent=2, default=str))
        print(f"[forecast] complete: {out_dir} (pandas.read_parquet('{out_dir}') reads all parts)")
    else:
        print(f"[forecast] {len(done)}/{shards} shards written in {out_dir}")
    return out_dir
