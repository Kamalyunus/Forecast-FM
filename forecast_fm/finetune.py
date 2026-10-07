"""Production fine-tuning: train Chronos-2 once on all history up to a date,
save it as a named checkpoint, and forecast from it until the next retrain.

    python -m forecast_fm finetune configs/05_chronos2_lora.yaml
    python -m forecast_fm forecast models/<name>/forecast_config.yaml

The output directory holds:

    finetuned-ckpt/        the weights (a LoRA adapter or a full model)
    forecast_fm_model.json provenance: base model, recipe, training window,
                           data fingerprint, code commit, environment
    forecast_config.yaml   the recipe with model_id pointing here and
                           fine_tune removed: ready for `forecast`

Leakage: the checkpoint trains on rows with ts <= as_of only. Its manifest
records that date as `train_end`, and the chronos2 model refuses to
forecast from any cutoff before it, so a saved checkpoint can never be
backtested on data it was trained on. Backtest the recipe (a config with
`fine_tune:`, retrained per fold) and use the saved checkpoint for
production.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from .config import ExperimentConfig, ProjectConfig
from .data import SERIES, TS
from .device import describe
from .ledger import code_state
from .models.chronos2 import CKPT, MANIFEST, Chronos2

FORMAT_VERSION = 1


def default_out(project: ProjectConfig, exp: ExperimentConfig, as_of: pd.Timestamp) -> Path:
    slug = re.sub(r"[^a-z0-9]+", "-", exp.name.lower()).strip("-")
    return Path(project.models_dir) / f"{slug}-{as_of:%Y%m%d}"


def _streaming_rounds(project: ProjectConfig, params: dict, data, as_of: pd.Timestamp):
    """Round source that never loads the catalog: one streaming pass builds
    the per-series profile (class, length, recent activity); each round then
    reads only its own series from disk."""
    from .data import load_panel
    from .train_mix import MAX_POOL_SERIES, select_from_profile, stream_profile

    ft = params["fine_tune"]
    mix = ft.get("train_mix") or {}
    limit = int(ft.get("max_pool_series", MAX_POOL_SERIES))
    prof = stream_profile(project, data, as_of, int(mix.get("lookback_days", 365)))
    print(f"[finetune] profiled {len(prof):,} series up to {as_of.date()}")

    def draw(k, used):
        if mix:
            ids, report = select_from_profile(prof, project, mix, exclude=used, round_index=k,
                                              max_pool_series=limit)
        elif k == 0:
            if len(prof) > limit:
                raise ValueError(f"{len(prof):,} series would be loaded for fine-tuning (limit {limit:,}): "
                                 "set fine_tune.train_mix.max_series and fine_tune.rounds")
            ids, report = pd.Index(prof.index), None
        else:
            return None
        if not len(ids):
            return None
        panel = load_panel(project, data, series=set(ids))
        return panel[panel[TS] <= as_of], ids, report

    return draw


def finetune(project: ProjectConfig, exp: ExperimentConfig, panel: pd.DataFrame | None = None,
             as_of: str | pd.Timestamp | None = None, out: str | Path | None = None,
             force: bool = False, config_path: str | Path | None = None, data=None) -> Path:
    """Fine-tune on history <= as_of and save the checkpoint. With `panel`,
    training series are drawn from it in memory; without, the catalog at
    `data` (default: project.data_path) is profiled by streaming and each
    round loads only its own series, so millions of SKUs never sit in memory."""
    params = project.model_params(exp.model, exp.model_params)
    if exp.model != "chronos2" or not params.get("fine_tune"):
        raise ValueError("finetune needs a chronos2 config with a `fine_tune:` block "
                         "(e.g. configs/05_chronos2_lora.yaml)")
    if panel is not None:
        last = panel[TS].max()
    else:
        from .data import date_span

        _, last = date_span(project, data)
    as_of = last if as_of is None else pd.Timestamp(as_of)
    if as_of > last:
        raise ValueError(f"--as-of {as_of.date()} is after the last date in the data ({last.date()})")
    out = Path(out) if out else default_out(project, exp, as_of)
    if out.exists() and not force:
        raise FileExistsError(f"{out} exists; checkpoints are not overwritten (pass --force, "
                              "or choose another --out)")

    model = Chronos2(params)
    model._load(as_of)
    if panel is not None:
        history = panel[panel[TS] <= as_of]
        source = model.memory_rounds(history, project)
        print(f"[finetune] {exp.name}: {history[SERIES].nunique():,} series in memory, up to "
              f"{as_of.date()}, on {model.device} ({model.dtype})")
    else:
        history = None
        source = _streaming_rounds(project, params, data, as_of)
        print(f"[finetune] {exp.name}: streaming from {data or project.data_path}, up to "
              f"{as_of.date()}, on {model.device} ({model.dtype})")
    known = list(project.known_covariate_cols)
    past = list(project.past_covariate_cols) if params.get("past_covariates", True) else []

    # train into a staging dir so a failed run never leaves a half checkpoint
    stage = out.with_name(out.name + ".partial")
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True)
    facts = model.train(history, project, stage, rounds_source=source)

    base_id = str(params.get("model_id", "amazon/chronos-2"))
    manifest = {
        "format_version": FORMAT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "train_start": facts["train_start"],
        "train_end": str(as_of.date()),
        "base_model_id": base_id,
        "recipe": {
            "experiment": exp.name, "hypothesis": exp.hypothesis, "model_params": params,
            "config_path": str(config_path) if config_path else None,
            "config_sha256": (hashlib.sha256(Path(config_path).read_bytes()).hexdigest()
                              if config_path else None),
        },
        "project": {"name": project.name, "data_path": project.data_path,
                    "horizon": project.horizon, "known_covariates": known,
                    "past_covariates": past, "static_cols": project.static_cols,
                    "covariate_eval_policy": project.covariate_eval_policy},
        "data_hash": facts["data_hash"],
        "training": facts,
        "code": code_state(),
        "env": describe(),
    }
    (stage / MANIFEST).write_text(json.dumps(manifest, indent=2, default=str))

    serve = dict(params, model_id=str(out))
    serve.pop("fine_tune", None)
    serve.pop("cache_dir", None)
    forecast_exp = ExperimentConfig(
        name=f"{exp.name}-{as_of:%Y%m%d}", model="chronos2", model_params=serve,
        hypothesis=f"production forecast from checkpoint trained through {as_of.date()}",
        rationale=f"finetune of recipe {exp.name!r}", based_on=exp.based_on)
    (stage / "forecast_config.yaml").write_text(yaml.safe_dump(asdict(forecast_exp), sort_keys=False))

    if out.exists():
        shutil.rmtree(out)
    stage.rename(out)
    print(f"[finetune] saved {out}/{CKPT} ({facts['n_series_trained']:,}/{facts['n_series']:,} "
          f"series long enough to train on, {facts['train_seconds']}s)")
    print(f"[finetune] forecast with: python -m forecast_fm forecast {out}/forecast_config.yaml")
    return out
