"""Forecasts for new series (cold start): launch profiles from analogs.

A foundation model needs history to read; a SKU launching tomorrow has none.
Series with no history at the origin, or less than `min_history_days`, are
forecast here instead:

    launch profile   for every age a (days since the series' first row),
                     the mean and quantiles of daily demand across past
                     launches ("analogs"): series whose first row falls
                     after the data starts (so their launch was observed)
                     and whose age-a day is <= the cutoff. Pooled by
                     `profile_cols` (coarsest first, e.g. [category,
                     subcategory]): the finest level with >= min_analogs
                     analogs at that age wins, then coarser, then all series.
    new series       forecast(t) = profile at age (t - launch date); zero
                     before launch.
    short history    forecast from its current age onward, scaled by how it
                     has sold versus the profile so far (shrunk towards 1,
                     clipped to [0.2, 5]).

Leakage: profiles are built from `history` (<= cutoff) only. In a backtest a
series first seen within the horizon is treated as a planned launch: its
launch date and statics are taken as known at the cutoff, as a launch plan
would make them.

Config (`cold_start:` in project.yaml): method (launch_profile | zero | none),
profile_cols, min_analogs (20), new_series_path, launch_date_col
(launch_date), from_plans (true).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ProjectConfig
from .data import SERIES, STOCKOUT, TS, Y, make_series_id, read_table
from .plans import AS_OF, HORIZON

LAUNCH = "launch_date"
LIFECYCLE = "lifecycle"
AGE = "_age"


def settings(project: ProjectConfig) -> dict:
    cs = project.cold_start
    return {
        "method": cs.get("method", "launch_profile"),
        "profile_cols": list(cs.get("profile_cols") or []),
        "min_analogs": int(cs.get("min_analogs", 20)),
        "new_series_path": cs.get("new_series_path"),
        "launch_date_col": cs.get("launch_date_col", LAUNCH),
        "from_plans": bool(cs.get("from_plans", True)),
    }


def _levels(cols: list[str]) -> list[list[str]]:
    """[category, subcategory] -> [[category, subcategory], [category], []]."""
    return [cols[:k] for k in range(len(cols), -1, -1)]


def _key(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    if not cols:
        return pd.Series("*", index=df.index)
    return make_series_id(df, cols)


def launch_profiles(history: pd.DataFrame, project: ProjectConfig, max_age: int,
                    quantiles: list[float]) -> dict:
    """{level (tuple of cols): DataFrame indexed by (key, age) with columns
    mean, q_<q>..., n}. Only analogs with an observed launch count."""
    cfg = settings(project)
    first = history.groupby(SERIES, observed=True)[TS].min()
    data_start = history[TS].min()
    launched = first[first > data_start]  # launch observed, not left-censored
    h = history[history[SERIES].isin(launched.index) & ~history[STOCKOUT] & history[Y].notna()]
    lmap = dict(zip(np.asarray(launched.index.astype(str)), launched.to_numpy(), strict=True))
    launch = (pd.to_datetime(h[SERIES].astype(str).map(lmap)) if len(h)
              else pd.Series(pd.NaT, index=h.index, dtype="datetime64[ns]"))
    h = h.assign(**{AGE: (h[TS] - launch).dt.days})
    h = h[h[AGE] < max_age]
    out = {}
    for level in _levels(cfg["profile_cols"]):
        cols = [c for c in level if c in h.columns]
        if len(cols) != len(level):
            continue
        k = _key(h, cols)
        g = h.groupby([k.rename("key"), h[AGE]], observed=True)[Y]
        stats = pd.DataFrame({"mean": g.mean(), "n": g.size()})
        if quantiles:
            qs = g.quantile(quantiles).unstack()
            qs.columns = [f"q_{q:g}" for q in qs.columns]
            stats = stats.join(qs)
        out[tuple(level)] = stats[stats["n"] >= cfg["min_analogs"]]
    return out


def _lookup(profiles: dict, statics: pd.DataFrame, ages: np.ndarray, cols: list[str]) -> pd.DataFrame:
    """Profile rows for each (series row, age): the finest level with enough
    analogs at that age; ages past the last supported age carry its value."""
    n = len(ages)
    out = pd.DataFrame(np.nan, index=range(n), columns=cols)
    found = np.zeros(n, dtype=bool)
    for level, prof in profiles.items():
        if prof.empty or found.all() or any(c not in statics.columns for c in level):
            continue
        keys = _key(statics, list(level)).to_numpy() if level else np.full(n, "*", dtype=object)
        # carry forward: clamp ages to the last supported age per key
        last_age = prof.reset_index().groupby("key")[AGE].max()
        cap = pd.Series(keys).map(last_age).to_numpy(dtype=float)
        a = np.where(np.isnan(cap), -1, np.minimum(ages, np.nan_to_num(cap))).astype(np.int64)
        idx = pd.MultiIndex.from_arrays([keys, a])
        vals = prof.reindex(idx)[cols].to_numpy()
        hit = ~found & ~np.isnan(vals[:, 0]) & (a >= 0)
        out.loc[hit, :] = vals[hit]
        found |= hit
    return out.fillna(0.0)


def forecast(cold: pd.DataFrame, history: pd.DataFrame, project: ProjectConfig,
             cutoff: pd.Timestamp) -> tuple[pd.DataFrame, dict]:
    """cold: one row per cold series with SERIES, LAUNCH (datetime), statics,
    and `hist_sum` / `hist_age` for short-history series (0 for new ones).
    Returns the long frame [series_id, ts, horizon, y_pred, q_*] and stats."""
    cfg = settings(project)
    H = project.horizon
    qs = sorted(set(project.quantiles) | {0.5})
    qcols = [f"q_{q:g}" for q in qs]
    if cold.empty or cfg["method"] == "none":
        return pd.DataFrame(columns=[SERIES, TS, HORIZON, "y_pred", *qcols]), {"n_cold": 0}

    n = len(cold)
    h = np.tile(np.arange(1, H + 1), n)
    rep = np.repeat(np.arange(n), H)
    ts = pd.Timestamp(cutoff) + pd.to_timedelta(h, unit="D")
    launch = pd.to_datetime(cold[LAUNCH]).to_numpy()[rep]
    ages = ((ts.to_numpy() - launch) // np.timedelta64(1, "D")).astype(np.int64)
    out = pd.DataFrame({SERIES: cold[SERIES].astype(str).to_numpy()[rep], TS: ts, HORIZON: h})
    if cfg["method"] == "zero":
        out["y_pred"] = 0.0
        for c in qcols:
            out[c] = 0.0
        return out, {"n_cold": n, "method": "zero"}

    max_age = int(max(ages.max(), 0)) + 1
    # a cold series is never its own analog
    pool = history[~history[SERIES].astype(str).isin(set(cold[SERIES].astype(str)))]
    profiles = launch_profiles(pool, project, max_age, qs)
    cols = ["mean", *qcols]
    statics = cold.iloc[rep].reset_index(drop=True)
    vals = _lookup(profiles, statics, np.maximum(ages, 0), cols)

    # short history: scale by how the series sold vs the profile at its ages
    scale = np.ones(n)
    short = cold["hist_age"].to_numpy() > 0
    if short.any():
        s_idx = np.flatnonzero(short)
        s_rep = np.repeat(s_idx, cold["hist_age"].to_numpy()[s_idx])
        s_age = np.concatenate([np.arange(a) for a in cold["hist_age"].to_numpy()[s_idx]])
        past = _lookup(profiles, cold.iloc[s_rep].reset_index(drop=True), s_age, ["mean"])["mean"]
        expected = pd.Series(past.to_numpy()).groupby(s_rep).sum().reindex(s_idx).to_numpy()
        prior = 7 * max(float(np.nanmean(past)) if len(past) else 0.0, 1e-6)
        ratio = (cold["hist_sum"].to_numpy()[s_idx] + prior) / (expected + prior)
        scale[s_idx] = np.clip(ratio, 0.2, 5.0)
    factor = scale[rep] * (ages >= 0)  # zero before launch
    out["y_pred"] = vals["mean"].to_numpy() * factor
    for c in qcols:
        out[c] = vals[c].to_numpy() * factor
    q = np.sort(out[qcols].to_numpy(), axis=1)  # scaling keeps order; guard ties/NaN
    out[qcols] = q
    n_analog_levels = {"/".join(k) or "all": int(v.index.get_level_values(0).nunique())
                       for k, v in profiles.items()}
    return out, {"n_cold": n, "n_short_history": int(short.sum()), "method": "launch_profile",
                 "profile_keys": n_analog_levels}


def production_new_series(project: ProjectConfig, history: pd.DataFrame, cutoff: pd.Timestamp,
                          plans: pd.DataFrame | None, keep=None) -> pd.DataFrame:
    """Upcoming series for a production forecast: the new_series_path file
    (series id cols, statics, optional launch date) and, with from_plans,
    series in the latest plan snapshot that have no history. `keep` filters
    series ids (sharding)."""
    cfg = settings(project)
    have = set(history[SERIES].astype(str).unique())
    frames = []
    if cfg["new_series_path"]:
        f = read_table(cfg["new_series_path"])
        missing = [c for c in project.series_id_cols if c not in f.columns]
        if missing:
            raise ValueError(f"cold_start.new_series_path lacks series id columns {missing}")
        f[SERIES] = make_series_id(f, project.series_id_cols)
        lc = cfg["launch_date_col"]
        f[LAUNCH] = pd.to_datetime(f[lc]) if lc in f.columns else pd.Timestamp(cutoff) + pd.Timedelta(days=1)
        frames.append(f[[SERIES, LAUNCH, *[c for c in project.static_cols if c in f.columns]]])
    if cfg["from_plans"] and plans is not None and len(plans):
        snap = plans[plans[AS_OF] <= pd.Timestamp(cutoff)]
        if len(snap):
            snap = snap[snap[AS_OF] == snap[AS_OF].max()]
            first = snap.groupby(snap[SERIES].astype(str))[TS].min()
            frames.append(pd.DataFrame({SERIES: first.index,
                                        LAUNCH: np.maximum(first.to_numpy(),
                                                           np.datetime64(pd.Timestamp(cutoff)
                                                                         + pd.Timedelta(days=1)))}))
    if not frames:
        return pd.DataFrame(columns=[SERIES, LAUNCH])
    new = pd.concat(frames, ignore_index=True)
    new[SERIES] = new[SERIES].astype(str)
    new = new[~new[SERIES].isin(have)].drop_duplicates(SERIES, keep="first")
    if keep is not None:
        new = new[keep(new[SERIES])]
    return new.reset_index(drop=True)
