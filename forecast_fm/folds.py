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
    explicit = project.holdout_cutoffs if holdout else project.cutoffs
    if explicit:
        cutoffs = sorted(pd.Timestamp(c) for c in explicit)
        late = [c for c in cutoffs if c + pd.Timedelta(days=project.horizon) > last]
        if late:
            raise ValueError(f"cutoffs {[str(c.date()) for c in late]}: horizon runs past the "
                             f"last date {last.date()}")
        if not holdout and project.holdout_cutoffs:
            first_hold = min(pd.Timestamp(c) for c in project.holdout_cutoffs)
            if max(cutoffs) + pd.Timedelta(days=project.horizon) > first_hold:
                raise ValueError("validation cutoffs' horizons overlap the first holdout cutoff")
        return cutoffs
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
