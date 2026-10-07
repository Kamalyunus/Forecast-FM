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


MAX_POOL_SERIES = 200_000  # default fine_tune.max_pool_series: above this, a cap is required


def profile(history: pd.DataFrame, project: ProjectConfig, lookback_days: int = 365) -> pd.DataFrame:
    """Per series (index: series id as str): demand class, days of history at
    the cutoff (`length`), and in-stock days with a sale in the last
    `lookback_days` (`recent_nonzero`). From `history` (<= cutoff) only."""
    cutoff = history[TS].max()
    lengths = history.groupby(SERIES, observed=True).size()
    classes = demand_classes(history, project).astype(str).reindex(lengths.index)
    recent = history[(history[TS] > cutoff - pd.Timedelta(days=lookback_days)) & ~history[STOCKOUT]]
    nonzero = (recent[Y] > 0).groupby(recent[SERIES], observed=True).sum().reindex(lengths.index,
                                                                                   fill_value=0)
    out = pd.DataFrame({"demand_class": classes.to_numpy(), "length": lengths.to_numpy(),
                        "recent_nonzero": nonzero.to_numpy()}, index=lengths.index.astype(str))
    out.attrs["cutoff"] = pd.Timestamp(cutoff)
    return out


def stream_profile(project: ProjectConfig, path, as_of: pd.Timestamp, lookback_days: int = 365,
                   batch_rows: int = 2_000_000) -> pd.DataFrame:
    """The same profile as `profile`, built by streaming the raw data (rows <=
    as_of) with per-series running sums, so a multi-million-SKU catalog never
    has to fit in memory. Classes follow data.demand_classes exactly (label
    column, else ADI / CV^2 of in-stock non-zero demand)."""
    from .data import prepare_raw, read_table

    as_of = pd.Timestamp(as_of)
    recent_from = as_of - pd.Timedelta(days=lookback_days)
    label = project.demand_label_col
    parts: list[pd.DataFrame] = []

    def collect(df: pd.DataFrame) -> np.ndarray:
        r = prepare_raw(df, project, allow_empty=True)
        r = r[r[TS] <= as_of]
        if r.empty:
            return np.zeros(len(df), dtype=bool)
        y = r[Y].to_numpy(dtype=np.float64)
        ok = ~r[STOCKOUT].to_numpy(dtype=bool)
        nz = ok & (np.nan_to_num(y) > 0)
        f = pd.DataFrame({
            SERIES: r[SERIES].astype(str).to_numpy(), "first": r[TS].to_numpy(), "last": r[TS].to_numpy(),
            "stockout": (~ok).astype(np.int64), "n_nz": nz.astype(np.int64),
            "s1": np.where(nz, y, 0.0), "s2": np.where(nz, y * y, 0.0),
            "recent": (nz & (r[TS] > recent_from).to_numpy()).astype(np.int64),
        })
        g = f.groupby(SERIES, sort=False).agg(first=("first", "min"), last=("last", "max"),
                                              stockout=("stockout", "sum"), n_nz=("n_nz", "sum"),
                                              s1=("s1", "sum"), s2=("s2", "sum"), recent=("recent", "sum"))
        if label:
            first_rows = r.sort_values(TS).drop_duplicates(SERIES)
            g["label"] = pd.Series(first_rows[label].astype(str).to_numpy(),
                                   index=first_rows[SERIES].astype(str).to_numpy()).reindex(g.index)
            g["label_ts"] = g["first"]
        parts.append(g)
        return np.zeros(len(df), dtype=bool)

    read_table(path or project.data_path, keep=collect, batch_rows=batch_rows)
    if not parts:
        raise ValueError(f"no rows on or before {as_of.date()}")
    allp = pd.concat(parts)
    agg = {"first": "min", "last": "max", "stockout": "sum", "n_nz": "sum", "s1": "sum", "s2": "sum",
           "recent": "sum"}
    g = allp.groupby(level=0).agg(agg)
    end = as_of if project.grid_end == "global" else g["last"]
    length = ((end - g["first"]).dt.days + 1).astype(np.int64)
    if label:
        lab = allp.sort_values("label_ts").groupby(level=0)["label"].first()
        classes = lab.reindex(g.index).astype(str)
    else:
        n = (length - g["stockout"]).clip(lower=1)
        mean = g["s1"] / g["n_nz"].replace(0, np.nan)
        var = g["s2"] / g["n_nz"].replace(0, np.nan) - mean ** 2
        adi, cv2 = n / g["n_nz"].replace(0, np.nan), var.clip(lower=0) / mean ** 2
        classes = pd.Series(np.select(
            [g["n_nz"] == 0, (adi < 1.32) & (cv2 < 0.49), adi < 1.32, cv2 < 0.49],
            ["no_demand", "smooth", "erratic", "intermittent"], default="lumpy"), index=g.index)
    out = pd.DataFrame({"demand_class": classes, "length": length, "recent_nonzero": g["recent"]})
    out.index = out.index.astype(str)
    out.attrs["cutoff"] = as_of
    return out


def select_from_profile(prof: pd.DataFrame, project: ProjectConfig, mix: dict,
                        exclude: set[str] | None = None, round_index: int = 0,
                        max_pool_series: int = MAX_POOL_SERIES) -> tuple[pd.Index, dict]:
    """(series ids to train on, report) for one training round. Series in
    `exclude` (earlier rounds) are never drawn again; each round uses seed +
    round_index, so rounds are disjoint, stratified the same way, and
    reproducible. Returns an empty index when the pool is exhausted."""
    validate(mix)
    classes, lengths = prof["demand_class"], prof["length"]
    report: dict = {"cutoff": str(pd.Timestamp(prof.attrs.get("cutoff")).date())
                    if prof.attrs.get("cutoff") is not None else None,
                    "round": round_index, "n_series": int(len(prof))}
    ok = pd.Series(True, index=prof.index)
    too_short = lengths < 2 * project.horizon  # the trainer needs context + horizon
    report["excluded_too_short"] = int(too_short.sum())
    ok &= ~too_short
    if mix.get("classes"):
        allowed = {str(c) for c in mix["classes"]}
        out = ok & ~classes.isin(allowed)
        report["excluded_class"] = int(out.sum())
        ok &= classes.isin(allowed)
    if mix.get("min_nonzero_days"):
        low = ok & (prof["recent_nonzero"] < int(mix["min_nonzero_days"]))
        report["excluded_low_activity"] = int(low.sum())
        ok &= ~low
    if exclude:
        used = ok & prof.index.isin(list(exclude))
        report["excluded_earlier_rounds"] = int(used.sum())
        ok &= ~prof.index.isin(list(exclude))

    avail = classes[ok].value_counts().sort_index()
    target = _capped_counts(avail, mix.get("max_share") or {})
    max_series = mix.get("max_series")
    if max_series and target.sum() > int(max_series):
        from .sample import allocate

        target = allocate(target, int(max_series), floor=0)
    if not max_series and target.sum() > max_pool_series:
        # memory guard: building inputs for millions of series would not fit
        raise ValueError(
            f"{int(target.sum()):,} series would be loaded for fine-tuning (limit {max_pool_series:,}). "
            "Set fine_tune.train_mix.max_series (e.g. 50000) and use fine_tune.rounds to cover more "
            "series, or raise fine_tune.max_pool_series if you know they fit in memory.")

    rng = np.random.default_rng(int(mix.get("seed", 0)) + round_index)
    chosen = []
    for c in sorted(target.index):
        members = np.sort(prof.index[(classes == c).to_numpy() & ok.to_numpy()].astype(str))
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
    return ids, report


def select(history: pd.DataFrame, project: ProjectConfig, mix: dict) -> tuple[pd.Index, dict]:
    """(series ids to train on, report) from in-memory history (<= cutoff)."""
    validate(mix)
    ids, report = select_from_profile(profile(history, project, int(mix.get("lookback_days", 365))),
                                      project, mix)
    if len(ids) == 0:
        raise ValueError(f"fine_tune.train_mix left no series to train on: {report}")
    return ids, report
