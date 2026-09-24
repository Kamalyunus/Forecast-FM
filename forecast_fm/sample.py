"""Stratified series sample, drawn BEFORE the ETL (the full panel does not fit
through it for experiments).

Pass 1 streams only the id, target and stratum columns and aggregates per
series. Strata are `--by` label x per-series mean-volume bin. Each stratum's
quota is proportional to its size, with a floor of min(floor, size) so rare
classes stay represented. Pass 2 streams the full rows of the chosen series
into the output file. The manifest records every chosen series with its
stratum and population weight (stratum size / sampled), so the sample can be
reproduced and population metrics re-weighted.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from .config import ProjectConfig
from .data import SERIES, make_series_id


def _dataset(path: str | Path) -> ds.Dataset:
    path = Path(path)
    fmt = "csv" if path.suffix == ".csv" else "parquet"
    return ds.dataset(str(path), format=fmt)


def series_stats(path: str | Path, id_cols: list[str], target: str, by: str | None,
                 batch_rows: int = 2_000_000) -> pd.DataFrame:
    """Per series: [series_id, <id cols>, total, n, <by>] from a streaming scan."""
    cols = list(dict.fromkeys([*id_cols, target, *([by] if by else [])]))
    parts = []
    for batch in _dataset(path).to_batches(columns=cols, batch_size=batch_rows):
        df = batch.to_pandas()
        df[SERIES] = make_series_id(df, id_cols)
        agg = {"total": (target, "sum"), "n": (target, "size")}
        if by:
            agg[by] = (by, "first")
        parts.append(df.groupby(SERIES, sort=False).agg(**agg))
    stats = pd.concat(parts)
    agg = {"total": "sum", "n": "sum", **({by: "first"} if by else {})}
    return stats.groupby(level=0).agg(agg)


def allocate(sizes: pd.Series, n: int, floor: int) -> pd.Series:
    """Proportional quotas (largest remainder) with a per-stratum floor of
    min(floor, size), capped at stratum size, summing to min(n, total)."""
    sizes = sizes.astype(int)
    n = min(n, int(sizes.sum()))
    base = np.minimum(sizes, floor)
    if base.sum() >= n:
        return base
    quota = base.copy()
    for _ in range(len(sizes) + 1):  # redistribute until caps stop binding
        left = n - int(quota.sum())
        room = sizes - quota
        if left <= 0 or room.sum() == 0:
            break
        w = room / room.sum() * left
        add = np.minimum(np.floor(w).astype(int), room)
        rem = left - int(add.sum())
        if rem > 0:
            frac = (w - np.floor(w)).where(room - add > 0, -1)
            top = frac.sort_values(ascending=False, kind="stable").index[:rem]
            add.loc[top] += 1
        quota = quota + np.minimum(add, room)
    return quota


def stratified_sample(stats: pd.DataFrame, n: int, by: str | None, volume_bins: int,
                      floor: int, seed: int) -> pd.DataFrame:
    stats = stats.sort_index()
    mean = stats["total"] / stats["n"]
    vbin = pd.qcut(mean.rank(method="first"), q=min(volume_bins, len(stats)), labels=False)
    label = stats[by].astype(str) if by else pd.Series("all", index=stats.index)
    strata = label + "|v" + vbin.astype(str)
    sizes = strata.value_counts().sort_index()
    quota = allocate(sizes, n, floor)
    rng = np.random.default_rng(seed)
    chosen = []
    for s in sizes.index:  # one draw per stratum (tens of strata), not per series
        members = strata.index[strata.to_numpy() == s]
        k = int(quota[s])
        if k:
            chosen.append(pd.DataFrame({SERIES: rng.choice(members, size=k, replace=False),
                                        "stratum": s, "weight": sizes[s] / k}))
    return pd.concat(chosen, ignore_index=True).sort_values(SERIES, ignore_index=True)


def write_rows(src: str | Path, out: str | Path, id_cols: list[str], keep: set[str],
               batch_rows: int = 2_000_000) -> int:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer, n = None, 0
    try:
        for batch in _dataset(src).to_batches(batch_size=batch_rows):
            df = batch.to_pandas()
            df = df[make_series_id(df, id_cols).isin(keep).to_numpy()]
            if df.empty:
                continue
            table = pa.Table.from_pandas(df, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(str(out), table.schema)
            writer.write_table(table.cast(writer.schema))
            n += len(df)
    finally:
        if writer is not None:
            writer.close()
    return n


def run_sample(project: ProjectConfig, n: int, by: str | None, volume_bins: int, floor: int,
               seed: int, out: str | Path, manifest: str | Path,
               src: str | Path | None = None) -> pd.DataFrame:
    src = src or project.data_path
    stats = series_stats(src, project.series_id_cols, project.target_col, by)
    picked = stratified_sample(stats, n, by, volume_bins, floor, seed)
    Path(manifest).parent.mkdir(parents=True, exist_ok=True)
    picked.to_csv(manifest, index=False)
    rows = write_rows(src, out, project.series_id_cols, set(picked[SERIES]))
    print(f"[sample] {len(picked)}/{len(stats)} series, {rows} rows -> {out}; manifest -> {manifest}")
    print(picked.groupby("stratum").size().rename("series").to_string())
    return picked
