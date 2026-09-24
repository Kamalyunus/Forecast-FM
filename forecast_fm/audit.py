"""Data audit: the facts to review with the user before any experiment."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from .config import ProjectConfig
from .data import SERIES, STOCKOUT, TS, Y, categorical_cols, demand_classes
from .folds import fold_cutoffs


def audit(panel: pd.DataFrame, raw_rows: int, project: ProjectConfig, out: str | Path) -> str:
    n_series = panel[SERIES].nunique()
    lengths = panel.groupby(SERIES, observed=True).size()
    cats = set(categorical_cols(project, panel))
    classes = demand_classes(panel, project)
    lines = [
        f"# Data audit — {project.name} ({date.today()})",
        "",
        f"- raw rows: {raw_rows:,}; panel rows after daily grid: {len(panel):,} "
        f"({1 - raw_rows / max(len(panel), 1):.1%} filled as zero-demand days)",
        f"- series: {n_series:,}; dates {panel[TS].min().date()} .. {panel[TS].max().date()}",
        f"- history length (days): min {lengths.min()}, median {int(lengths.median())}, max {lengths.max()}; "
        f"series shorter than min_train_periods ({project.min_train_periods}): "
        f"{int((lengths < project.min_train_periods).sum()):,}",
        f"- zero-demand share: {(panel[Y] == 0).mean():.1%}; negative targets: {int((panel[Y] < 0).sum()):,}",
        f"- stockout days (masked): {panel[STOCKOUT].mean():.2%}"
        + ("" if project.in_stock_col else " (no in_stock_col declared)"),
        f"- panel memory: {panel.memory_usage(deep=True).sum() / 2**20:,.0f} MB",
        "",
        "## Covariate types",
        "",
        "| column | class | type | missing | eval policy |",
        "|---|---|---|---|---|",
    ]
    for c in project.known_covariate_cols:
        lines.append(f"| {c} | known | {'categorical' if c in cats else 'numeric'} | "
                     f"{panel[c].isna().mean():.1%} | {project.policy(c)} |")
    for c in project.past_covariate_cols:
        lines.append(f"| {c} | past | {'categorical' if c in cats else 'numeric'} | "
                     f"{panel[c].isna().mean():.1%} | never in horizon |")
    for c in project.static_cols:
        lines.append(f"| {c} | static | categorical | — | group_by / slices |")
    vc = classes.value_counts()
    lines += ["", "## Demand classes", "", "| class | series | share |", "|---|---|---|"]
    lines += [f"| {k} | {v:,} | {v / len(classes):.1%} |" for k, v in vc.items()]
    try:
        val = fold_cutoffs(panel[TS].min(), panel[TS].max(), project)
        hold = fold_cutoffs(panel[TS].min(), panel[TS].max(), project, holdout=True)
        lines += ["", "## Fold cutoffs", "",
                  f"- validation: {', '.join(str(c.date()) for c in val)}",
                  f"- holdout: {', '.join(str(c.date()) for c in hold)}"]
    except ValueError as e:
        lines += ["", f"**Fold cutoffs: {e}**"]
    text = "\n".join(lines) + "\n"
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    return text

