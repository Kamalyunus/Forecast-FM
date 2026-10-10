"""Backtest fold cutoffs: the same dates for every experiment."""

from __future__ import annotations

import pandas as pd

from .config import ProjectConfig


def fold_cutoffs(first: pd.Timestamp, last: pd.Timestamp, project: ProjectConfig,
                 holdout: bool = False) -> list[pd.Timestamp]:
    """Validation cutoffs, ascending. Explicit `cutoffs` / `holdout_cutoffs`
    in project.yaml win over the rule below. With `holdout=True`, the
    `holdout_folds` most recent cutoffs instead: the test set that only
    promotion evaluates.

    Fold i's cutoff is `last - horizon - i*fold_step`, so its whole horizon
    fits inside the data. Validation starts far enough back that its
    horizon windows never reach into a holdout window: when fold_step <
    horizon the windows overlap, so the gap is ceil(horizon / fold_step)
    folds, not 1.
    """
    last, first = pd.Timestamp(last), pd.Timestamp(first)
    if holdout and not project.holdout_cutoffs and not project.holdout_folds:
        return []  # no holdout configured
    explicit = project.holdout_cutoffs if holdout else project.cutoffs
    if explicit:
        cutoffs = sorted(pd.Timestamp(c) for c in explicit)
        late = [c for c in cutoffs if c + pd.Timedelta(days=project.horizon) > last]
        if late:
            raise ValueError(f"cutoffs {[str(c.date()) for c in late]}: horizon runs past the "
                             f"last date {last.date()}")
        if not holdout and (project.holdout_cutoffs or project.holdout_folds):
            # explicit or rule-based, the holdout window is off limits to validation
            first_hold = min(fold_cutoffs(first, last, project, holdout=True))
            if max(cutoffs) + pd.Timedelta(days=project.horizon) > first_hold:
                raise ValueError(f"validation cutoff {max(cutoffs).date()} + horizon overlaps the holdout "
                                 f"window starting {first_hold.date()}: move it earlier, or set "
                                 "holdout_cutoffs explicitly")
        return cutoffs
    if not holdout and project.holdout_cutoffs:
        # Anchor automatic validation to the actual holdout, not to an
        # unrelated window counted backwards from the end of the dataset.
        boundary = min(fold_cutoffs(first, last, project, holdout=True))
        cutoffs = [boundary - pd.Timedelta(days=project.horizon + i * project.fold_step)
                   for i in range(project.n_folds)]
    else:
        if holdout:
            idx = range(project.holdout_folds)
        else:
            gap = -(-project.horizon // project.fold_step)
            start = project.holdout_folds - 1 + gap if project.holdout_folds else 0
            idx = range(start, start + project.n_folds)
        cutoffs = [last - pd.Timedelta(days=project.horizon + i * project.fold_step) for i in idx]
    earliest = first + pd.Timedelta(days=project.min_train_periods)
    kept = sorted(c for c in cutoffs if c >= earliest)
    if len(kept) < len(cutoffs):
        print(f"[folds] history supports only {len(kept)}/{len(cutoffs)} "
              f"{'holdout' if holdout else 'validation'} folds "
              f"(min_train_periods={project.min_train_periods})")
    if not kept:
        raise ValueError("no fold cutoff leaves min_train_periods of history; "
                         "lower min_train_periods or n_folds")
    return kept


def origin_limit(first: pd.Timestamp, last: pd.Timestamp, project: ProjectConfig) -> pd.Timestamp:
    """The date no validation forecast window may pass: the first holdout
    cutoff (validation never scores the holdout period), else the last date."""
    if project.holdout_cutoffs or project.holdout_folds:
        return min(fold_cutoffs(first, last, project, holdout=True))
    return pd.Timestamp(last)


def fold_origins(cutoffs: list[pd.Timestamp], project: ProjectConfig,
                 limit: pd.Timestamp) -> list[list[pd.Timestamp]]:
    """Forecast origins per fold. A fold's model is fit once at its cutoff and
    forecasts from cutoff, cutoff + step, ... before the next fold's cutoff,
    each origin's horizon ending by `limit`. Without origin_step_days: just
    the cutoff."""
    step = project.origin_step_days
    H = pd.Timedelta(days=project.horizon)
    out = []
    for i, c in enumerate(cutoffs):
        if c + H > limit:
            raise ValueError(f"fold cutoff {c.date()} + horizon passes {limit.date()}, the first holdout "
                             "cutoff: validation may not score the holdout window")
        if not step:
            out.append([c])
            continue
        nxt = cutoffs[i + 1] if i + 1 < len(cutoffs) else None
        origins, o = [], c
        while (nxt is None or o < nxt) and o + H <= limit:
            origins.append(o)
            o = o + pd.Timedelta(days=step)
        if not origins:
            raise ValueError(f"fold cutoff {c.date()} + horizon passes {limit.date()}, the first holdout "
                             "cutoff: validation may not score the holdout window")
        out.append(origins)
    return out
