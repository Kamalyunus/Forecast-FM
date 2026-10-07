"""Chronos-2 (amazon/chronos-2, Apache-2.0): pretrained in-context forecaster.

Requires torch + chronos-forecasting (`pip install -e ".[foundation]"`). The
registry entry exists regardless and raises an informative error when they
are missing.

How the project's covariate classes map onto the model:

    known covariates   history values + horizon values (from the future
                       frame, so covariate_eval_policy applies)
    past covariates    history only; structurally never in the horizon
    categoricals       passed as strings; Chronos-2 target-encodes them
    statics            NOT covariates (constant within a series, they carry
                       no signal under per-series encoding). They drive
                       `group_by` cross-learning groups instead.
    stockouts          target becomes NaN (missing, not zero demand)

Leakage: the context is built from `history` (<= cutoff) only; horizon
covariate values come only from the future frame. Fine-tuning trains on the
fold's history only, and its checkpoint cache key includes the cutoff and a
hash of that history, so a cached checkpoint can never serve another fold.

Scale: inputs are built per chunk of series and results are written into a
preallocated (n_series, horizon, n_quantiles) float32 array, so 700k series
never sit in memory as input dicts at once.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import ProjectConfig
from ..data import SERIES, STOCKOUT, TS, Y, categorical_cols
from ..device import empty_cache, peak_memory_mb, prepare, resolve_device, resolve_dtype
from ..plans import HORIZON
from ..train_mix import MAX_POOL_SERIES, profile, select_from_profile
from ..train_mix import validate as validate_mix
from .base import Forecast, Forecaster

DEFAULT_MODEL_ID = "amazon/chronos-2"
FINE_TUNE_KEYS = frozenset({"num_steps", "learning_rate", "mode", "batch_size", "context_length",
                            "train_mix", "rounds", "max_pool_series", "log_every"})

# loaded base pipelines keyed by (model_id, device, dtype): the backtest
# builds a fresh model per fold; reloading the weights each time is waste
_PIPELINES: dict[tuple[str, str, str], object] = {}


def _chronos():
    try:
        import torch  # noqa: F401
        from chronos import Chronos2Pipeline
    except ImportError as e:
        raise ImportError("model 'chronos2' needs torch + chronos-forecasting: "
                          'pip install -e ".[foundation]"') from e
    return Chronos2Pipeline


def _place(pipe, device: str, dtype: str):
    pipe.model.to(device=device, dtype=resolve_dtype(dtype, device))
    pipe.model.eval()
    return pipe


MANIFEST = "forecast_fm_model.json"
CKPT = "finetuned-ckpt"


class TrainedOnFutureError(ValueError):
    """A saved fine-tuned checkpoint was trained on data after the cutoff it is
    asked to forecast from: evaluating it there would be leakage."""


def read_manifest(model_id: str) -> dict | None:
    """The provenance manifest of a fine-tuned checkpoint (a `finetune` output
    or a backtest cache entry), found next to the weights or one level up, so
    pointing at `<dir>/finetuned-ckpt` directly cannot skip it. None for a
    base model (HF id, s3://, or a plain directory)."""
    p = Path(str(model_id))
    for d in (p, p.parent):
        f = d / MANIFEST
        if f.is_file():
            return json.loads(f.read_text())
    return None


def looks_finetuned(model_id: str) -> bool:
    """A local directory that holds fine-tuned weights (a LoRA adapter or a
    `finetuned-ckpt`), whatever its manifest status."""
    p = Path(str(model_id))
    return p.is_dir() and (p.name == CKPT or (p / CKPT).is_dir()
                           or (p / "adapter_config.json").is_file())


def checkpoint_path(model_id: str) -> str:
    """Where the weights are: a `finetune` output keeps them in finetuned-ckpt/."""
    p = Path(str(model_id))
    return str(p / CKPT) if (p / MANIFEST).is_file() else str(model_id)


def code_fingerprint() -> str:
    """Changes whenever the code that builds training inputs, or the chronos
    library, changes: a stale cached checkpoint is never reused."""
    from importlib.metadata import PackageNotFoundError, version

    h = hashlib.sha256()
    here = Path(__file__).resolve().parent.parent
    for rel in ("models/chronos2.py", "train_mix.py", "data.py"):
        h.update((here / rel).read_bytes())
    try:
        h.update(version("chronos-forecasting").encode())
    except PackageNotFoundError:
        pass
    return h.hexdigest()[:16]


def load_pipeline(model_id: str, device: str, dtype: str = "float32"):
    """Load once per process from the HF hub, a local directory, s3://, or a
    `finetune` output directory."""
    key = (model_id, device, dtype)
    if key not in _PIPELINES:
        prepare(device)
        pipe = _place(_chronos().from_pretrained(checkpoint_path(model_id)), device, dtype)
        print(f"[chronos2] loaded {model_id} on {device} ({dtype}); "
              f"model_context_length={getattr(pipe, 'model_context_length', '?')}, "
              f"model_prediction_length={getattr(pipe, 'model_prediction_length', '?')}")
        _PIPELINES[key] = pipe
    return _PIPELINES[key]


def interp_quantiles(levels: list[float], q: np.ndarray, wanted: list[float]) -> np.ndarray:
    """q: (..., n_levels) at the model's levels -> (..., len(wanted)).
    Linear between levels, clamped outside the model's range."""
    lv = np.asarray(levels, dtype=np.float64)
    out = np.empty(q.shape[:-1] + (len(wanted),), dtype=np.float32)
    for j, w in enumerate(wanted):
        hi = int(np.searchsorted(lv, w))
        if hi < len(lv) and np.isclose(lv[hi], w):
            out[..., j] = q[..., hi]
        elif hi == 0:
            out[..., j] = q[..., 0]
        elif hi >= len(lv):
            out[..., j] = q[..., -1]
        else:
            a = (w - lv[hi - 1]) / (lv[hi] - lv[hi - 1])
            out[..., j] = (1 - a) * q[..., hi - 1] + a * q[..., hi]
    return out


class Chronos2(Forecaster):
    """Params:

    model_id        HF id, local dir or s3:// prefix ("amazon/chronos-2")
    device          auto | cpu | mps | cuda | cuda:N ("auto": cuda > mps > cpu)
    dtype           float32 | bfloat16 | float16 ("float32")
    context_length  history days fed per series (default: all, up to the
                    model's limit)
    batch_size      variates per forward pass, covariates included (256)
    chunk_series    series per predict call without group_by; bounds host
                    memory (4096)
    past_covariates feed project.past_covariate_cols as history (true)
    group_by        static cols; series sharing values are forecast jointly
                    with cross-learning ([] = independent series)
    group_size      max series per cross-learning group (100)
    fine_tune       {num_steps, learning_rate, mode: full|lora, batch_size,
                    context_length}; omitted = zero-shot
    cache_dir       fine-tuned checkpoint cache ("reports/chronos2_ft")
    allow_unverified_checkpoint
                    load fine-tuned weights that have no forecast_fm manifest
                    (false: refused, since their training window is unknown)
    """

    PARAM_KEYS = frozenset({"model_id", "device", "dtype", "context_length", "batch_size",
                            "chunk_series", "past_covariates", "group_by", "group_size",
                            "fine_tune", "cache_dir", "allow_unverified_checkpoint"})

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        ft = self.params.get("fine_tune")
        if ft is not None:
            if not isinstance(ft, dict):
                raise ValueError(f"chronos2 fine_tune must be a mapping of {sorted(FINE_TUNE_KEYS)}")
            bad = set(ft) - FINE_TUNE_KEYS
            if bad:
                raise ValueError(f"chronos2 fine_tune: unknown keys {sorted(bad)}")
            if ft.get("mode", "full") not in ("full", "lora"):
                raise ValueError("chronos2 fine_tune.mode must be 'full' or 'lora'")
            if ft.get("train_mix") is not None:
                validate_mix(ft["train_mix"])
            rounds = int(ft.get("rounds", 1))
            if rounds < 1:
                raise ValueError("chronos2 fine_tune.rounds must be >= 1")
            if rounds > 1 and not (ft.get("train_mix") or {}).get("max_series"):
                raise ValueError("chronos2 fine_tune.rounds > 1 needs train_mix.max_series: the "
                                 "number of series drawn per round")
        if not isinstance(self.params.get("group_by") or [], list):
            raise ValueError("chronos2 group_by must be a list of static_cols")
        self.stats: dict = {}

    # --- input assembly ----------------------------------------------------

    def _covariates(self, history: pd.DataFrame, project: ProjectConfig) -> tuple[list[str], list[str]]:
        known = [c for c in project.known_covariate_cols if c in history.columns]
        past = []
        if self.params.get("past_covariates", True):
            past = [c for c in project.past_covariate_cols if c in history.columns and c not in known]
        return known, past

    def _context(self, history: pd.DataFrame, project: ProjectConfig,
                 cutoff: pd.Timestamp | None = None) -> dict:
        """Row bounds per series plus the history arrays, built once.
        Categoricals stay as integer codes (strings are built per series in
        _input: a string per row would not fit in memory at scale). With a
        cutoff, `gap` counts the days between each series' last row and the
        cutoff, so a context can be padded to end at the cutoff."""
        h = history
        keys = _series_keys(h)
        if not _sorted_by_series_ts(keys, h[TS].to_numpy()):
            h = h.sort_values([SERIES, TS], kind="stable")
            keys = _series_keys(h)
        change = np.flatnonzero(keys[1:] != keys[:-1]) + 1
        starts = np.concatenate([[0], change]).astype(np.int64)
        ends = np.concatenate([change, [len(keys)]]).astype(np.int64)
        sids = h[SERIES].to_numpy()[starts] if len(h) else np.array([])

        y = h[Y].to_numpy(dtype=np.float32).copy()
        y[h[STOCKOUT].to_numpy(dtype=bool)] = np.nan
        known, past = self._covariates(h, project)
        cats = set(categorical_cols(project, h))
        arrays: dict = {}
        for c in known + past:
            if c in cats:
                cat = pd.Categorical(h[c])
                codes = cat.codes.astype(np.int32)
                labels = np.array([*map(str, cat.categories), ""], dtype=str)
                codes = np.where(codes < 0, len(labels) - 1, codes)  # missing -> "" token
                arrays[c] = (codes, labels)
            else:
                arrays[c] = h[c].to_numpy(dtype=np.float32)
        gap = np.zeros(len(starts), dtype=np.int64)
        if cutoff is not None and len(h):
            last = h[TS].to_numpy()[ends - 1]
            gap = ((np.datetime64(pd.Timestamp(cutoff)) - last) // np.timedelta64(1, "D")).astype(np.int64)
            gap = np.maximum(gap, 0)
        return dict(frame=h, sids=pd.Index(sids), starts=starts, ends=ends, y=y, known=known,
                    past=past, cats=cats, arrays=arrays, gap=gap)

    def _future_arrays(self, ctx: dict, future: pd.DataFrame, H: int) -> dict[str, np.ndarray]:
        """(n_series, H) per known covariate, rows aligned to ctx['sids'].
        Absent cells: NaN (numeric) or "" (categorical)."""
        rows = ctx["sids"].get_indexer(future[SERIES])
        hz = future[HORIZON].to_numpy(dtype=np.int64) - 1
        ok = (rows >= 0) & (hz >= 0) & (hz < H)
        out = {}
        for c in ctx["known"]:
            if c in ctx["cats"]:
                mat = np.full((len(ctx["sids"]), H), "", dtype=object)
                vals = future[c].astype(str).to_numpy()
            else:
                mat = np.full((len(ctx["sids"]), H), np.nan, dtype=np.float32)
                vals = future[c].to_numpy(dtype=np.float32)
            mat[rows[ok], hz[ok]] = vals[ok]
            out[c] = mat.astype(str) if c in ctx["cats"] else mat
        return out

    def _input(self, ctx: dict, i: int, fut: dict | None, max_ctx: int) -> dict:
        s, e = int(ctx["starts"][i]), int(ctx["ends"][i])
        gap = int(ctx["gap"][i])
        if max_ctx:
            s = max(s, e - max(max_ctx - gap, 1))
        y = ctx["y"][s:e]
        if gap:  # the series' rows stop before the cutoff: pad so step 1 = cutoff + 1
            y = np.concatenate([y, np.full(gap, np.nan, dtype=np.float32)])
        d: dict = {"target": y}
        covs = ctx["known"] + ctx["past"]
        if covs:
            pc = {}
            for c in covs:
                arr = ctx["arrays"][c]
                if isinstance(arr, tuple):
                    codes, labels = arr
                    v = labels[codes[s:e]]
                    pc[c] = np.concatenate([v, np.full(gap, "", dtype=v.dtype)]) if gap else v
                else:
                    v = arr[s:e]
                    pc[c] = np.concatenate([v, np.full(gap, np.nan, dtype=np.float32)]) if gap else v
            d["past_covariates"] = pc
        if ctx["known"]:
            d["future_covariates"] = ({c: None for c in ctx["known"]} if fut is None
                                      else {c: fut[c][i] for c in ctx["known"]})
        return d

    def _chunks(self, ctx: dict) -> tuple[list[np.ndarray], bool]:
        n = len(ctx["sids"])
        group_by = self.params.get("group_by") or []
        if not group_by:
            size = int(self.params.get("chunk_series", 4096))
            return [np.arange(i, min(i + size, n)) for i in range(0, n, size)], False
        frame = ctx["frame"]
        missing = [c for c in group_by if c not in frame.columns]
        if missing:
            raise ValueError(f"chronos2 group_by {missing} not in the panel; declare them in static_cols")
        keys = frame[group_by].iloc[ctx["starts"]].astype(str).agg("|".join, axis=1).to_numpy()
        codes = pd.factorize(keys)[0]
        order = np.argsort(codes, kind="stable")
        bounds = np.flatnonzero(np.diff(codes[order])) + 1
        size = int(self.params.get("group_size", 100))
        return [g[j:j + size] for g in np.split(order, bounds) for j in range(0, len(g), size)], True

    # --- fine-tune checkpoint cache ------------------------------------------

    def _ft_key(self, history: pd.DataFrame, project: ProjectConfig, cutoff) -> str:
        known, past = self._covariates(history, project)
        model_id = self.params.get("model_id", DEFAULT_MODEL_ID)
        payload = {
            "model_id": model_id,
            # a finetune output reused as a base: its identity is its manifest
            "base_manifest": read_manifest(model_id),
            "dtype": self.params.get("dtype", "float32"),
            "context_length": self.params.get("context_length"),
            "fine_tune": self.params["fine_tune"],
            "known": known, "past": past, "horizon": project.horizon,
            "policy": project.covariate_eval_policy,
            "cutoff": str(pd.Timestamp(cutoff).date()),
            "n_series": int(history[SERIES].nunique()), "data_hash": history_hash(history, known, past),
            "code": code_fingerprint(),
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:16]

    def _load(self, cutoff: pd.Timestamp) -> None:
        model_id = str(self.params.get("model_id", DEFAULT_MODEL_ID))
        # checked BEFORE loading: a checkpoint trained past the cutoff has seen
        # the targets it would be scored on
        manifest = read_manifest(model_id)
        if manifest is not None and pd.Timestamp(manifest["train_end"]) > pd.Timestamp(cutoff):
            raise TrainedOnFutureError(
                f"{model_id} was fine-tuned on data through {manifest['train_end']}; it cannot "
                f"forecast from cutoff {pd.Timestamp(cutoff).date()}. To backtest a recipe, put "
                "`fine_tune:` in the config (it retrains per fold); use a saved checkpoint only "
                "for forecasts from its train_end onwards.")
        if manifest is None and looks_finetuned(model_id) and not self.params.get(
                "allow_unverified_checkpoint"):
            raise TrainedOnFutureError(
                f"{model_id} holds fine-tuned weights with no {MANIFEST}, so its training window is "
                "unknown and it could have seen the backtest's future. Use the directory written "
                "by `finetune`, or set allow_unverified_checkpoint: true if you know it is safe.")
        self.device = resolve_device(str(self.params.get("device", "auto")))
        self.dtype = str(self.params.get("dtype", "float32"))
        self.pipe = load_pipeline(model_id, self.device, self.dtype)
        self.stats = {"device": self.device, "dtype": self.dtype}
        if manifest is not None:
            self.stats.update(checkpoint=model_id, checkpoint_train_end=manifest["train_end"],
                              checkpoint_age_days=(pd.Timestamp(cutoff)
                                                   - pd.Timestamp(manifest["train_end"])).days)

    def memory_rounds(self, history: pd.DataFrame, project: ProjectConfig):
        """Round source over in-memory history: each call (round k, ids already
        used) returns (history of the round's series, their ids, report), or
        None once the pool is exhausted."""
        ft = self.params["fine_tune"]
        mix = ft.get("train_mix")
        limit = int(ft.get("max_pool_series", MAX_POOL_SERIES))
        if not mix:
            n = int(history[SERIES].nunique())
            if n > limit:
                raise ValueError(f"{n:,} series would be loaded for fine-tuning (limit {limit:,}): set "
                                 "fine_tune.train_mix.max_series and fine_tune.rounds")

            def everything(k, used):
                return (history, pd.Index(history[SERIES].astype(str).unique()), None) if k == 0 else None
            return everything
        prof = profile(history, project, int(mix.get("lookback_days", 365)))

        def draw(k, used):
            ids, report = select_from_profile(prof, project, mix, exclude=used, round_index=k,
                                              max_pool_series=limit)
            if not len(ids):
                return None
            return history[history[SERIES].astype(str).isin(set(ids))], ids, report
        return draw

    def train(self, history: pd.DataFrame | None, project: ProjectConfig, out_dir: Path,
              rounds_source=None) -> dict:
        """Fine-tune the loaded pipeline; the result is saved as one full model
        in out_dir/finetuned-ckpt. Every row given is training data: slice to
        the cutoff first.

        With fine_tune.rounds = K, training runs K rounds of num_steps / K
        steps, each on a fresh stratified draw of train_mix.max_series series
        disjoint from earlier rounds (`rounds_source` supplies them: in memory,
        or streamed from disk by `finetune`). LoRA adapters are merged into the
        weights after each round, and the learning rate steps down linearly
        across rounds (constant within a round), approximating one schedule
        over the whole run. Returns training facts for the manifest."""
        import shutil

        ft = self.params["fine_tune"]
        rounds = int(ft.get("rounds", 1))
        total_steps = int(ft.get("num_steps", 1000))
        per_round = -(-total_steps // rounds)
        lr = float(ft.get("learning_rate", 1e-6))
        mode = ft.get("mode", "full")
        max_ctx = int(ft.get("context_length") or self.params.get("context_length") or 0)
        source = rounds_source or self.memory_rounds(history, project)
        n_all = int(history[SERIES].nunique()) if history is not None else None

        used: set[str] = set()
        round_facts, curve = [], []
        data_hash, train_start, n_trained = 0, None, 0
        t_all = time.perf_counter()
        for k in range(rounds):
            got = source(k, used)
            if got is None:
                print(f"[chronos2] round {k + 1}: no series left to draw; stopping after {k} round(s)")
                break
            hist_k, ids, report = got
            used |= set(map(str, ids))
            ctx = self._context(hist_k, project)
            lengths = ctx["ends"] - ctx["starts"]
            eligible = int((lengths >= 2 * project.horizon).sum())  # chronos min_past = horizon
            if eligible == 0:
                raise ValueError(f"no series has >= {2 * project.horizon} days of history "
                                 "(horizon of context + horizon of target); cannot fine-tune")
            # full series: the trainer samples windows across the whole history
            # and crops each window's context to `context_length` itself
            inputs = [self._input(ctx, i, None, 0) for i in range(len(ctx["sids"]))]
            known, past = self._covariates(hist_k, project)
            data_hash += history_hash(hist_k, known, past)
            start = hist_k[TS].min()
            train_start = start if train_start is None else min(train_start, start)
            n_trained += eligible
            lr_k = lr if rounds == 1 else lr * (1 - k / rounds)
            recorder = _loss_recorder(curve, offset=k * per_round)
            stage = Path(out_dir) / f".round-{k}"
            t0 = time.perf_counter()
            extra = {} if rounds == 1 else {"lr_scheduler_type": "constant"}
            self.pipe = self.pipe.fit(
                inputs,
                prediction_length=project.horizon,
                finetune_mode=mode,
                learning_rate=lr_k,
                num_steps=per_round,
                batch_size=int(ft.get("batch_size", 256)),
                context_length=max_ctx or None,
                output_dir=str(stage),
                finetuned_ckpt_name=CKPT,
                remove_printer_callback=True,
                callbacks=[recorder] if recorder is not None else None,
                logging_steps=max(1, int(ft.get("log_every", max(1, per_round // 50)))),
                **extra,
            )
            merge = getattr(self.pipe.model, "merge_and_unload", None)
            if merge is not None:  # LoRA: fold the adapter into the weights before the next round
                self.pipe.model = merge()
            _place(self.pipe, self.device, self.dtype)
            shutil.rmtree(stage, ignore_errors=True)
            losses = [v for s, v in curve if s > k * per_round]
            round_facts.append({"round": k, "series": int(len(ids)), "series_trained": eligible,
                                "steps": per_round, "learning_rate": lr_k,
                                "seconds": round(time.perf_counter() - t0, 1),
                                "first_loss": losses[0] if losses else None,
                                "last_loss": losses[-1] if losses else None,
                                "mix": report})
            print(f"[chronos2] round {k + 1}/{rounds}: {len(ids):,} series, {per_round} steps, "
                  f"lr {lr_k:.2e}, {round_facts[-1]['seconds']}s"
                  + (f", loss {losses[0]:.4f} -> {losses[-1]:.4f}" if losses else ""))
        if not round_facts:
            raise ValueError("fine-tuning drew no series")
        # one full model (adapters merged), loadable without the base checkpoint
        self.pipe.save_pretrained(Path(out_dir) / CKPT)
        facts = {"n_series": n_all if n_all is not None else int(len(used)),
                 "n_series_trained": n_trained, "rounds": round_facts,
                 "total_steps": per_round * len(round_facts),
                 "train_seconds": round(time.perf_counter() - t_all, 1),
                 "data_hash": data_hash, "train_start": str(pd.Timestamp(train_start).date()),
                 "loss_curve": _thin(curve, 200)}
        if len(round_facts) == 1 and round_facts[0]["mix"] is not None:
            facts["train_mix"] = round_facts[0]["mix"]  # back-compatible single-round report
        return facts

    # --- Forecaster interface ------------------------------------------------

    def fit(self, history: pd.DataFrame, project: ProjectConfig, cutoff: pd.Timestamp) -> None:
        self._load(cutoff)
        ft = self.params.get("fine_tune")
        if ft is None:
            return
        key = self._ft_key(history, project, cutoff)
        out_dir = Path(self.params.get("cache_dir", "reports/chronos2_ft")) / key
        ckpt = out_dir / CKPT
        if ckpt.exists():
            print(f"[chronos2] fine-tune cache hit {key} (cutoff {pd.Timestamp(cutoff).date()})")
            self.pipe = _place(_chronos().from_pretrained(str(ckpt)), self.device, self.dtype)
            self.stats["fine_tune_cache"] = "hit"
            meta = out_dir / "key.json"
            if meta.exists() and "train_mix" in (saved := json.loads(meta.read_text())):
                self.stats["train_mix"] = saved["train_mix"]
            return
        facts = self.train(history, project, out_dir)
        meta = {"cutoff": str(pd.Timestamp(cutoff).date()), "fine_tune": ft}
        if "train_mix" in facts:
            meta["train_mix"] = self.stats["train_mix"] = facts["train_mix"]
        (out_dir / "key.json").write_text(json.dumps(meta, indent=2, default=str))
        # the guard reads this: a cached fold checkpoint can never be reused
        # at an earlier cutoff, whoever points model_id at it
        (out_dir / MANIFEST).write_text(json.dumps(
            {"train_end": meta["cutoff"], "kind": "backtest_cache", "fine_tune": ft}, default=str))
        self.stats.update(fine_tune_cache="miss", fine_tune_seconds=facts["train_seconds"])

    def predict(self, history: pd.DataFrame, future: pd.DataFrame, project: ProjectConfig) -> Forecast:
        H = int(future[HORIZON].max())
        cutoff = pd.Timestamp(future[TS].min()) - pd.Timedelta(days=1)
        ctx = self._context(history, project, cutoff)
        fut = self._future_arrays(ctx, future, H) if ctx["known"] else None
        levels = list(self.pipe.quantiles)
        wanted = sorted(set(project.quantiles) | {0.5})
        n_var = 1 + len(ctx["known"]) + len(ctx["past"])
        batch = int(self.params.get("batch_size", 256))
        max_ctx = int(self.params.get("context_length") or 0)

        cube = np.zeros((len(ctx["sids"]), H, len(wanted)), dtype=np.float32)
        chunks, cross = self._chunks(ctx)
        t0 = time.perf_counter()
        for idx in chunks:
            inputs = [self._input(ctx, int(i), fut, max_ctx) for i in idx]
            preds = self.pipe.predict(
                inputs, prediction_length=H, cross_learning=cross,
                # a cross-learning group must fit ONE forward pass, or it
                # silently splits across batches
                batch_size=max(batch, len(inputs) * n_var) if cross else batch,
            )
            q = np.stack([p.detach().float().cpu().numpy()[0] for p in preds])  # (n, levels, H)
            cube[idx] = interp_quantiles(levels, q.transpose(0, 2, 1), wanted)
        secs = time.perf_counter() - t0
        empty_cache(self.device)
        self.stats.update(n_series=len(ctx["sids"]), n_variates=n_var, predict_seconds=round(secs, 2),
                          series_per_second=round(len(ctx["sids"]) / secs, 1) if secs else None,
                          accel_memory_mb=peak_memory_mb(self.device))

        rows = ctx["sids"].get_indexer(future[SERIES])
        hz = future[HORIZON].to_numpy(dtype=np.int64) - 1
        out = np.where((rows >= 0)[:, None], cube[np.maximum(rows, 0), hz], 0.0)
        if np.nanmin(ctx["y"], initial=0.0) >= 0:
            out = np.clip(out, 0, None)  # non-negative history -> non-negative forecasts
        out = np.sort(out, axis=1)
        qd = {w: out[:, j] for j, w in enumerate(wanted)}
        return qd[0.5], {q: qd[q] for q in project.quantiles}


def _loss_recorder(curve: list, offset: int):
    """A transformers callback appending (global step, training loss)."""
    try:
        from transformers import TrainerCallback
    except ImportError:
        return None

    class _Rec(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kw):
            if logs and "loss" in logs:
                curve.append((offset + int(state.global_step), float(logs["loss"])))

    return _Rec()


def _thin(curve: list, n: int) -> list:
    if len(curve) <= n:
        return curve
    idx = np.linspace(0, len(curve) - 1, n).round().astype(int)
    return [curve[i] for i in idx]


def history_hash(history: pd.DataFrame, known: list[str], past: list[str]) -> int:
    """Order-sensitive fingerprint of exactly the data a model trains on."""
    cols = [SERIES, TS, Y, STOCKOUT, *known, *past]
    return int(pd.util.hash_pandas_object(history[cols], index=False).sum())


def _series_keys(h: pd.DataFrame) -> np.ndarray:
    s = h[SERIES]
    if isinstance(s.dtype, pd.CategoricalDtype):
        return s.cat.codes.to_numpy()
    return pd.factorize(s, sort=True)[0]


def _sorted_by_series_ts(keys: np.ndarray, ts: np.ndarray) -> bool:
    if len(keys) < 2:
        return True
    dk = np.diff(keys)
    if (dk < 0).any():
        return False
    same = dk == 0
    return bool((np.diff(ts)[same] > np.timedelta64(0, "ns")).all())
