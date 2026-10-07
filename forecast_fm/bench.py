"""Measure, on this machine, what fine-tuning and forecasting cost, then
estimate the full plan before committing hours to it.

    python -m forecast_fm bench configs/05_chronos2_lora.yaml --catalog 3000000

Times on a random subset of series (streamed from the data, never the whole
catalog):

    forecast   zero-shot predict: series per second, and memory
    training   two short fine-tunes (a few steps, then --steps) of the
               config's fine_tune settings: seconds per step, with the fixed
               setup cost (loading inputs, saving the checkpoint) separated

and extrapolates to: hours per fine-tune (num_steps x rounds), a backtest
(one fine-tune per fold, forecasts at every origin), a production forecast
of the whole catalog, and the shard count that fits a memory budget.
Estimates scale linearly; real runs on very different series lengths or
covariate counts will differ, so rerun bench when those change.
"""

from __future__ import annotations

import dataclasses
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from .config import ExperimentConfig, ProjectConfig
from .data import SERIES, date_span, load_panel
from .device import describe
from .folds import fold_cutoffs, fold_origins, origin_limit
from .plans import future_frame


def _peak_rss_mb() -> float:
    import resource

    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 2**20 if sys.platform == "darwin" else r / 1024  # bytes on macOS, KiB on Linux


def bench(project: ProjectConfig, exp: ExperimentConfig, data=None, n_series: int = 2000,
          train_steps: int = 50, catalog: int | None = None, backtest_series: int | None = None,
          memory_gb: float = 24.0, seed: int = 0) -> dict:
    from .models.chronos2 import Chronos2

    params = project.model_params(exp.model, exp.model_params)
    if exp.model != "chronos2":
        raise ValueError("bench measures the chronos2 model; give a chronos2 config")
    first, last, all_ids = date_span(project, data, with_series=True)
    rng = np.random.default_rng(seed)
    pick = set(rng.choice(sorted(all_ids), size=min(n_series, len(all_ids)), replace=False))
    catalog = catalog or len(all_ids)

    rss0 = _peak_rss_mb()
    t0 = time.perf_counter()
    panel = load_panel(project, data, series=pick)
    load_s = time.perf_counter() - t0
    n = int(panel[SERIES].nunique())
    rss_panel = _peak_rss_mb() - rss0

    # forecasting: zero-shot; covariate values do not change the cost
    zs = {k: v for k, v in params.items() if k != "fine_tune"}
    model = Chronos2(zs)
    t0 = time.perf_counter()
    model._load(last)
    model_load_s = time.perf_counter() - t0
    flat = dataclasses.replace(project, covariate_eval_policy={c: "carry_forward"
                                                               for c in project.known_covariate_cols})
    fut = future_frame(panel, last, flat, None, verbose=False)
    t0 = time.perf_counter()
    model.predict(panel, fut, flat)
    predict_s = time.perf_counter() - t0
    sps = n / predict_s if predict_s else float("inf")
    rss_after = _peak_rss_mb() - rss0
    out: dict = {
        "machine": describe(), "series_measured": n, "catalog_series": catalog,
        "data_load_seconds": round(load_s, 2), "model_load_seconds": round(model_load_s, 2),
        "forecast": {"series_per_second": round(sps, 1), "seconds": round(predict_s, 2),
                     "variates_per_series": model.stats.get("n_variates"),
                     "accelerator_memory_mb": model.stats.get("accel_memory_mb")},
        "memory": {"panel_mb_per_series": round(rss_panel / max(n, 1), 4),
                   "fixed_mb": round(max(rss_after - rss_panel, 0), 1)},
    }

    ft = params.get("fine_tune")
    if ft:
        short = max(2, min(5, train_steps // 5))
        if train_steps <= short:
            raise ValueError(f"--steps must be > {short}")
        times = {}
        for steps in (short, train_steps):
            ftp = {k: v for k, v in ft.items() if k not in ("train_mix", "rounds")}
            ftp.update(num_steps=steps, rounds=1, log_every=steps)
            m = Chronos2({**params, "fine_tune": ftp})
            m._load(last)
            with tempfile.TemporaryDirectory() as d:
                t0 = time.perf_counter()
                m.train(panel, project, Path(d))
                times[steps] = time.perf_counter() - t0
        per_step = max((times[train_steps] - times[short]) / (train_steps - short), 1e-9)
        setup = max(times[short] - short * per_step, 0.0)
        out["training"] = {"seconds_per_step": float(f"{per_step:.4g}"), "setup_seconds": round(setup, 1),
                           "batch_size": ft.get("batch_size", 256), "mode": ft.get("mode", "full"),
                           "context_length": ft.get("context_length") or params.get("context_length")}

    # --- extrapolate the plan ------------------------------------------------
    plan: dict = {}
    hours = lambda s: round(s / 3600, 2)  # noqa: E731
    if ft:
        rounds = int(ft.get("rounds", 1))
        steps = int(ft.get("num_steps", 1000))
        per_round_series = int((ft.get("train_mix") or {}).get("max_series") or n)
        setup_per_round = out["training"]["setup_seconds"] * per_round_series / max(n, 1)
        one_ft = steps * out["training"]["seconds_per_step"] + rounds * setup_per_round
        plan["finetune_hours"] = hours(one_ft)
    cutoffs = fold_cutoffs(first, last, project)
    origins = fold_origins(cutoffs, project, origin_limit(first, last, project))
    n_origins = sum(len(f) for f in origins)
    bt_series = backtest_series or len(all_ids)
    plan["backtest"] = {"fine_tunes": len(cutoffs) if ft else 0, "forecast_origins": n_origins,
                        "series": bt_series,
                        "forecast_hours": hours(bt_series * n_origins / sps)}
    plan["backtest"]["total_hours"] = round(plan["backtest"]["forecast_hours"]
                                            + (plan.get("finetune_hours", 0) * len(cutoffs)), 2)
    plan["production_forecast_hours"] = hours(catalog / sps)
    budget = memory_gb * 1024 * 0.8 - out["memory"]["fixed_mb"]
    per_series = max(out["memory"]["panel_mb_per_series"], 1e-6)
    per_shard = max(int(budget / per_series), 1)
    plan["shards_for_memory_budget"] = {"memory_gb": memory_gb, "series_per_shard": per_shard,
                                        "shards": int(np.ceil(catalog / per_shard))}
    out["plan"] = plan
    return out


def render(res: dict) -> str:
    f, p = res["forecast"], res["plan"]
    s = p["shards_for_memory_budget"]
    lines = [
        f"device {res['machine'].get('auto_device')} | measured on {res['series_measured']:,} series",
        f"forecast: {f['series_per_second']:,} series/s ({f['variates_per_series']} variates/series)",
    ]
    if "training" in res:
        t = res["training"]
        lines.append(f"training: {t['seconds_per_step']}s/step at batch {t['batch_size']} ({t['mode']}), "
                     f"+{t['setup_seconds']}s setup per run")
        lines.append(f"one fine-tune (num_steps x rounds): ~{p['finetune_hours']} h")
    b = p["backtest"]
    lines += [
        f"backtest: {b['fine_tunes']} fine-tune(s) + {b['forecast_origins']} origins x {b['series']:,} "
        f"series -> ~{b['total_hours']} h",
        f"production forecast of {res['catalog_series']:,} series: ~{p['production_forecast_hours']} h",
        f"memory: {s['series_per_shard']:,} series/shard within {s['memory_gb']} GB "
        f"-> --shards {s['shards']}",
    ]
    return "\n".join(lines)


def run(project, exp, out_path: str | Path, **kw) -> dict:
    res = bench(project, exp, **kw)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(res, indent=2, default=str))
    print(render(res))
    print(f"[bench] details -> {out_path}")
    return res

