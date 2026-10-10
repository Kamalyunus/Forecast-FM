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
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

from . import cold_start, metrics
from .backtest import LIFECYCLE, backtest, forecast_at
from .config import ExperimentConfig, ProjectConfig
from .data import ID_SEP, SERIES, TS, date_span, demand_classes, load_panel, make_series_id, shard_of
from .device import describe
from .folds import fold_cutoffs, fold_origins, origin_limit
from .plans import load_plans
from .provenance import code_digest, digest, input_snapshot


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
    inputs = input_snapshot(project, data)
    panel = load_panel(project, data) if shards == 1 else None
    first, last = _span(project, data, shards, panel)
    cutoffs = fold_cutoffs(first, last, project)
    origins = fold_origins(cutoffs, project, origin_limit(first, last, project))
    all_origins = sorted({o for fold in origins for o in fold})
    parts, stats, kept = [], [], None
    for i in range(shards):
        t0 = time.perf_counter()
        if shards > 1:
            panel = load_panel(project, data, shard=(i, shards), end=last)
            if panel.empty:
                continue
            print(f"[shard {i + 1}/{shards}] {panel[SERIES].nunique():,} series")
        ids = set(panel[SERIES].astype(str).unique())
        plans = load_plans(project, cutoffs=all_origins, series=ids)
        preds, st = backtest(panel, project, exp, plans, cutoffs=cutoffs, origins=origins,
                             data_start=first)
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
    inputs.check()
    result = metrics.result(metrics.combine(parts), project)
    result.update(stats=stats, env=describe(), cutoffs=cutoffs, shards=shards,
                  origins=[[str(o.date()) for o in fold] for fold in origins],
                  input_fingerprints=inputs.fingerprints)
    return result, kept


def _claim_run(out_dir: Path, identity: dict, force: bool, partial: bool) -> None:
    """Reuse parts only with the same recipe, inputs, and execution identity.
    Called under the metadata lock; a changed identity needs a full restart.
    """
    run_file = out_dir / "_run.json"
    want = identity
    if run_file.exists():
        have = json.loads(run_file.read_text())
        if have == want:
            return
        if not force:
            raise ValueError(f"{out_dir} was started with --shards {have.get('shards')} at origin "
                             f"{have.get('origin')} and different or unverified inputs/settings; "
                             "finish it with the same settings, choose another "
                             f"--out, or pass --force to start over")
    elif any(out_dir.glob("part-*.parquet")):
        if not force:
            raise ValueError(f"{out_dir} has forecast parts without a verified run identity; "
                             "choose another --out or pass --force to start over")
    if run_file.exists() or any(out_dir.glob("part-*.parquet")):
        if partial:
            raise ValueError("cannot restart changed inputs/settings with --shard: "
                             "use --force without --shard or choose another --out")
        for f in [*out_dir.glob("part-*.parquet"), *out_dir.glob("_part-*.json"),
                  out_dir / "_manifest.json"]:
            f.unlink(missing_ok=True)
    tmp = out_dir / "_run.json.tmp"
    tmp.write_text(json.dumps(want, indent=2, default=str))
    tmp.replace(run_file)


@contextmanager
def _run_lock(out_dir: Path, force: bool):
    """Normal shard workers may coexist; a forced restart must run alone.

    flock is available on the supported macOS/Linux targets. Locks are
    released by the OS even if a worker crashes.
    """
    import fcntl

    with (out_dir / "_active.lock").open("a") as active:
        try:
            fcntl.flock(active, (fcntl.LOCK_EX if force else fcntl.LOCK_SH) | fcntl.LOCK_NB)
        except BlockingIOError as e:
            raise ValueError("forecast directory is in use; wait for active workers before --force") from e
        try:
            yield
        finally:
            fcntl.flock(active, fcntl.LOCK_UN)


@contextmanager
def _metadata_lock(out_dir: Path, name: str = "_metadata.lock"):
    import fcntl

    with (out_dir / name).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


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
    if shards < 1:
        raise ValueError("--shards must be >= 1")
    params = project.model_params(exp.model, exp.model_params)
    inputs = input_snapshot(project, data, production=True, model_params=params)
    panel = load_panel(project, data) if shards == 1 else None
    if shards == 1:
        first, origin, known_ids = panel[TS].min(), panel[TS].max(), set(panel[SERIES].astype(str).unique())
    else:
        first, origin, known_ids = date_span(project, data, with_series=True)
    out_dir = Path(out_dir or Path(project.reports_dir) / f"forecast_{origin:%Y%m%d}")
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = only if only is not None else list(range(shards))
    bad = [i for i in todo if not 0 <= i < shards]
    if bad:
        raise ValueError(f"--shard {bad} out of range for --shards {shards}")
    inputs.check()
    identity = json.loads(json.dumps({
        "version": 2, "shards": shards, "origin": str(origin.date()),
        "project": project.to_dict(), "experiment": exp.name, "model": exp.model,
        "model_params": params, "inputs": inputs.fingerprints,
        "code": code_digest(), "env": describe(),
    }, default=str))
    with _run_lock(out_dir, force):
        with _metadata_lock(out_dir):
            _claim_run(out_dir, identity, force, partial=only is not None)
        return _forecast_parts(project, exp, data, shards, todo, out_dir, force,
                               panel, first, origin, known_ids, inputs, identity)


def _forecast_parts(project, exp, data, shards, todo, out_dir, force,
                    panel, first, origin, known_ids, inputs, identity):
    for i in todo:
        with _metadata_lock(out_dir, f"_shard-{i:05d}.lock"):
            part = out_dir / f"part-{i:05d}.parquet"
            if part.exists() and not force:
                print(f"[forecast] shard {i}: {part} exists, skipped (force to redo)")
                continue
            (out_dir / "_manifest.json").unlink(missing_ok=True)
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
                                          new_series=new, data_start=first)
            pred.insert(0, "origin", origin)
            pred["model"] = exp.name
            pred = _split_ids(pred, project)
            tmp = out_dir / f"_part-{i:05d}.tmp"
            try:
                pred.to_parquet(tmp, index=False)
                inputs.check()
                tmp.replace(part)  # a crash never leaves a half-written part
            finally:
                tmp.unlink(missing_ok=True)
            counts = pred.drop_duplicates(SERIES)[LIFECYCLE].value_counts().to_dict()
            # "_"-prefixed files are ignored by parquet readers of the directory
            (out_dir / f"_part-{i:05d}.json").write_text(json.dumps(
                {"shard": i, "shards": shards, "series": counts, "rows": len(pred),
                 "seconds": round(time.perf_counter() - t0, 1), "stats": stats}, indent=2, default=str))
            print(f"[forecast] shard {i + 1}/{shards}: {sum(counts.values()):,} series {counts} -> {part}")
            del panel, pred, plans

    inputs.check()
    with _metadata_lock(out_dir):
        done = sorted(out_dir.glob("part-*.parquet"))
        if len(done) == shards:
            manifest = out_dir / "_manifest.json.tmp"
            manifest.write_text(json.dumps(
                {"origin": str(origin.date()), "experiment": exp.name, "model": exp.model,
                 "model_params": project.model_params(exp.model, exp.model_params),
                 "horizon": project.horizon, "quantiles": project.quantiles, "shards": shards,
                 "parts": [p.name for p in done], "env": describe(),
                 "run_identity": digest(identity), "input_fingerprints": inputs.fingerprints},
                indent=2, default=str))
            manifest.replace(out_dir / "_manifest.json")
            print(f"[forecast] complete: {out_dir} (pandas.read_parquet('{out_dir}') reads all parts)")
        else:
            print(f"[forecast] {len(done)}/{shards} shards written in {out_dir}")
    return out_dir
