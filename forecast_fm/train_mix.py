"""Choose which series a fine-tune trains on: `fine_tune.train_mix`.

The Chronos-2 trainer draws a series uniformly at random for every training
window, so the class mix of the training set IS the class mix of what the
model learns. Left alone, a catalog that is mostly intermittent series
trains a model on mostly-zero windows. `train_mix` sets that mix explicitly:

    fine_tune:
      train_mix:
        classes: [BAU, seasonal, promo, event, intermittent]  # may train (default: all)
        max_share: {intermittent: 0.25}   # cap a class's share of the training series
        min_nonzero_days: 4               # drop near-dead series ...
        lookback_days: 365                # ... counted over the last N days before the cutoff
        max_series: 50000                 # cap the total, keeping the shares
        seed: 42

Classes are the demand classes (`demand_label_col`, else the computed
smooth / erratic / intermittent / lumpy). Everything is computed from
`history` alone, i.e. data <= the fold's cutoff: the choice of training
series never looks at the future. Only training is affected; every series
is still forecast and scored.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ProjectConfig
from .data import SERIES, STOCKOUT, TS, Y, demand_classes

MIX_KEYS = frozenset({"classes", "max_share", "min_nonzero_days", "lookback_days", "max_series", "seed"})


def validate(mix: dict) -> None:
    if not isinstance(mix, dict):
        raise ValueError(f"fine_tune.train_mix must be a mapping of {sorted(MIX_KEYS)}")
    bad = set(mix) - MIX_KEYS
    if bad:
        raise ValueError(f"fine_tune.train_mix: unknown keys {sorted(bad)}; valid: {sorted(MIX_KEYS)}")
    shares = mix.get("max_share") or {}
    if any(not 0 < float(v) <= 1 for v in shares.values()):
        raise ValueError("fine_tune.train_mix.max_share values must be in (0, 1]")
    if sum(float(v) for v in shares.values() if float(v) < 1) >= 1:
        raise ValueError("fine_tune.train_mix.max_share caps must sum to < 1")
    for k in ("min_nonzero_days", "lookback_days", "max_series"):
        if k in mix and int(mix[k]) < 0:
            raise ValueError(f"fine_tune.train_mix.{k} must be >= 0")


def _capped_counts(avail: pd.Series, caps: dict[str, float]) -> pd.Series:
    """Largest n_c <= avail_c with n_c <= cap_c * N for every capped class,
    where N = sum n_c and uncapped classes keep everything."""
    caps = {c: float(s) for c, s in caps.items() if c in avail.index and float(s) < 1}
    binding: set[str] = set()
    for _ in range(len(caps) + 1):
        free = float(avail[[c for c in avail.index if c not in binding]].sum())
        total = free / (1 - sum(caps[c] for c in binding)) if binding else free
        new = {c for c, s in caps.items() if avail[c] > s * total + 1e-9}
        if new == binding:
            break
        binding |= new
    n = avail.copy()
    for c in binding:
        n[c] = min(int(avail[c]), int(np.floor(caps[c] * total + 1e-9)))
    return n.astype(int)


def select(history: pd.DataFrame, project: ProjectConfig, mix: dict) -> tuple[pd.Index, dict]:
    """(series ids to train on, report). `history` must already be sliced to
    the cutoff; nothing after its last date is used."""
    validate(mix)
    cutoff = history[TS].max()
    classes = demand_classes(history, project).astype(str)
    lengths = history.groupby(SERIES, observed=True).size()
    classes = classes.reindex(lengths.index)

    lookback = int(mix.get("lookback_days", 365))
    recent = history[(history[TS] > cutoff - pd.Timedelta(days=lookback)) & ~history[STOCKOUT]]
    nonzero = (recent[Y] > 0).groupby(recent[SERIES], observed=True).sum().reindex(lengths.index,
                                                                                   fill_value=0)

    report: dict = {"cutoff": str(pd.Timestamp(cutoff).date()), "n_series": int(len(lengths))}
    ok = pd.Series(True, index=lengths.index)
    too_short = lengths < 2 * project.horizon  # the trainer needs context + horizon
    report["excluded_too_short"] = int(too_short.sum())
    ok &= ~too_short
    if mix.get("classes"):
        allowed = {str(c) for c in mix["classes"]}
        out = ok & ~classes.isin(allowed)
        report["excluded_class"] = int(out.sum())
        ok &= classes.isin(allowed)
    if mix.get("min_nonzero_days"):
        low = ok & (nonzero < int(mix["min_nonzero_days"]))
        report["excluded_low_activity"] = int(low.sum())
        ok &= ~low

    avail = classes[ok].value_counts().sort_index()
    target = _capped_counts(avail, mix.get("max_share") or {})
    max_series = mix.get("max_series")
    if max_series and target.sum() > int(max_series):
        from .sample import allocate

        target = allocate(target, int(max_series), floor=0)

    rng = np.random.default_rng(int(mix.get("seed", 0)))
    chosen = []
    for c in sorted(target.index):
        members = np.sort(classes.index[(classes == c).to_numpy() & ok.to_numpy()].astype(str))
        k = int(target[c])
        if k >= len(members):
            chosen.append(members)
        elif k > 0:
            chosen.append(np.sort(rng.choice(members, size=k, replace=False)))
    ids = pd.Index(np.concatenate(chosen) if chosen else np.array([], dtype=str))

    total = max(len(ids), 1)
    report.update(
        available={str(c): int(v) for c, v in avail.items()},
        selected={str(c): int(v) for c, v in target.items()},
        share={str(c): round(int(v) / total, 4) for c, v in target.items()},
        n_selected=int(len(ids)),
    )
    if len(ids) == 0:
        raise ValueError(f"fine_tune.train_mix left no series to train on: {report}")
    return ids, report
