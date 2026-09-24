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
from .base import Forecast, Forecaster

DEFAULT_MODEL_ID = "amazon/chronos-2"
FINE_TUNE_KEYS = frozenset({"num_steps", "learning_rate", "mode", "batch_size", "context_length"})

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
    """The provenance manifest of a checkpoint saved by `finetune`, or None
    for a base model (HF id, s3://, or a plain directory)."""
    path = Path(str(model_id)) / MANIFEST
    return json.loads(path.read_text()) if path.is_file() else None


def checkpoint_path(model_id: str) -> str:
    """Where the weights are: a `finetune` output keeps them in finetuned-ckpt/."""
    return str(Path(model_id) / CKPT) if read_manifest(model_id) else str(model_id)


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
    """

    PARAM_KEYS = frozenset({"model_id", "device", "dtype", "context_length", "batch_size",
                            "chunk_series", "past_covariates", "group_by", "group_size",
                            "fine_tune", "cache_dir"})

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

    def _context(self, history: pd.DataFrame, project: ProjectConfig) -> dict:
        """Row bounds per series plus the history arrays, built once."""
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
        arrays = {c: (h[c].astype(str).to_numpy() if c in cats else h[c].to_numpy(dtype=np.float32))
                  for c in known + past}
        return dict(frame=h, sids=pd.Index(sids), starts=starts, ends=ends, y=y, known=known,
                    past=past, cats=cats, arrays=arrays)

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
        if max_ctx:
            s = max(s, e - max_ctx)
        d: dict = {"target": ctx["y"][s:e]}
        covs = ctx["known"] + ctx["past"]
        if covs:
            d["past_covariates"] = {c: ctx["arrays"][c][s:e] for c in covs}
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
        self.device = resolve_device(str(self.params.get("device", "auto")))
        self.dtype = str(self.params.get("dtype", "float32"))
        self.pipe = load_pipeline(model_id, self.device, self.dtype)
        self.stats = {"device": self.device, "dtype": self.dtype}
        if manifest is not None:
            self.stats.update(checkpoint=model_id, checkpoint_train_end=manifest["train_end"],
                              checkpoint_age_days=(pd.Timestamp(cutoff)
                                                   - pd.Timestamp(manifest["train_end"])).days)

    def train(self, history: pd.DataFrame, project: ProjectConfig, out_dir: Path) -> dict:
        """Fine-tune the loaded pipeline on `history` (every row of it is
        training data: slice it to the cutoff first). Weights land in
        out_dir/finetuned-ckpt. Returns training facts for the manifest."""
        ft = self.params["fine_tune"]
        ctx = self._context(history, project)
        max_ctx = int(ft.get("context_length") or self.params.get("context_length") or 0)
        lengths = ctx["ends"] - ctx["starts"]
        n_eligible = int((lengths >= 2 * project.horizon).sum())  # chronos min_past = horizon
        if n_eligible == 0:
            raise ValueError(f"no series has >= {2 * project.horizon} days of history "
                             "(horizon of context + horizon of target); cannot fine-tune")
        inputs = [self._input(ctx, i, None, max_ctx) for i in range(len(ctx["sids"]))]
        t0 = time.perf_counter()
        self.pipe = self.pipe.fit(
            inputs,
            prediction_length=project.horizon,
            finetune_mode=ft.get("mode", "full"),
            learning_rate=float(ft.get("learning_rate", 1e-6)),
            num_steps=int(ft.get("num_steps", 1000)),
            batch_size=int(ft.get("batch_size", 256)),
            context_length=max_ctx or None,
            output_dir=str(out_dir),
            finetuned_ckpt_name=CKPT,
            remove_printer_callback=True,
        )
        _place(self.pipe, self.device, self.dtype)
        return {"n_series": len(ctx["sids"]), "n_series_trained": n_eligible,
                "train_seconds": round(time.perf_counter() - t0, 1)}

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
            return
        facts = self.train(history, project, out_dir)
        (out_dir / "key.json").write_text(json.dumps({"cutoff": str(pd.Timestamp(cutoff).date()),
                                                      "fine_tune": ft}, indent=2))
        self.stats.update(fine_tune_cache="miss", fine_tune_seconds=facts["train_seconds"])

    def predict(self, history: pd.DataFrame, future: pd.DataFrame, project: ProjectConfig) -> Forecast:
        H = int(future[HORIZON].max())
        ctx = self._context(history, project)
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
