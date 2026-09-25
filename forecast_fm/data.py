"""Load raw demand data and build the daily panel.

The panel is the one table every command works on. It has canonical columns
`series_id` (category), `ts` (datetime64), `y` (float32) and `stockout`
(bool), plus the declared covariates and statics under their own names.

Everything here is vectorized: no per-series Python loops. It has to hold up
at ~700k series x ~1460 days.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import ProjectConfig

SERIES = "series_id"
TS = "ts"
Y = "y"
STOCKOUT = "stockout"
ID_SEP = "|"


def make_series_id(df: pd.DataFrame, id_cols: list[str]) -> pd.Series:
    if len(id_cols) == 1:
        return df[id_cols[0]].astype(str)
    out = df[id_cols[0]].astype(str)
    for c in id_cols[1:]:
        out = out + ID_SEP + df[c].astype(str)
    return out


def _expand(path: str | Path | list) -> list[Path]:
    """A path, a directory, a glob pattern, or a list of those -> files/dirs."""
    items = path if isinstance(path, (list, tuple)) else [path]
    out: list[Path] = []
    for item in items:
        item = str(item)
        if any(ch in item for ch in "*?["):
            matches = sorted(Path().glob(item)) if not Path(item).is_absolute() else \
                sorted(Path("/").glob(item.lstrip("/")))
            if not matches:
                raise FileNotFoundError(f"no files match {item!r}")
            out.extend(matches)
        else:
            out.append(Path(item))
    return out


def _is_parquet(path: Path) -> bool:
    return path.suffix in (".parquet", ".pq") or path.is_dir()


def _read_one(path: Path, columns: list[str] | None, keep, batch_rows: int) -> pd.DataFrame:
    """Whole file, or (with `keep`) streamed in batches keeping only the rows
    keep(batch) marks True, so a shard never holds the full dataset."""
    if _is_parquet(path):
        if keep is None:
            return pd.read_parquet(path, columns=columns)
        import pyarrow.dataset as pads

        parts = []
        for batch in pads.dataset(str(path), format="parquet").to_batches(columns=columns,
                                                                         batch_size=batch_rows):
            df = batch.to_pandas()
            parts.append(df[keep(df)])
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=columns)
    if path.suffix in (".csv", ".gz", ".bz2", ".zip") or path.name.endswith((".csv.gz", ".csv.zip")):
        if keep is None:
            return pd.read_csv(path, usecols=columns)
        parts = [df[keep(df)] for df in pd.read_csv(path, usecols=columns, chunksize=batch_rows)]
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=columns)
    raise ValueError(f"unsupported data file {path} (csv or parquet)")


def read_table(path: str | Path | list, columns: list[str] | None = None, keep=None,
               batch_rows: int = 2_000_000) -> pd.DataFrame:
    parts = [_read_one(p, columns, keep, batch_rows) for p in _expand(path)]
    return parts[0] if len(parts) == 1 else pd.concat(parts, ignore_index=True)


def shard_of(keys: pd.Series, n_shards: int) -> np.ndarray:
    """Stable shard number per key (same key -> same shard in every run)."""
    h = pd.util.hash_array(keys.astype(str).to_numpy(dtype=object), categorize=False)
    return (h % np.uint64(n_shards)).astype(np.int64)


def shard_filter(project: ProjectConfig, shard: tuple[int, int] | None):
    """A row filter for read_table keeping shard i of n (None: keep all).
    Rows are assigned by series id, or by `shard_by` statics so series that
    share them (a cross-learning group) land in the same shard."""
    if shard is None:
        return None
    i, n = shard
    if not 0 <= i < n:
        raise ValueError(f"shard {i} out of range for {n} shards")

    def keep(df: pd.DataFrame) -> np.ndarray:
        key = (make_series_id(df, project.shard_by) if project.shard_by
               else make_series_id(df, project.series_id_cols))
        return shard_of(key, n) == i

    return keep


def apply_derived(df: pd.DataFrame, project: ProjectConfig, strict: bool = True) -> pd.DataFrame:
    """Add `derived_columns` (pandas expressions) in order. With strict=False,
    an expression whose inputs are absent is skipped (plan files carry only
    some columns)."""
    for name, expr in project.derived_columns.items():
        try:
            df[name] = df.eval(expr)
        except Exception as e:  # pandas raises several types for bad names
            if strict:
                raise ValueError(f"derived_columns[{name}] = {expr!r} failed: {e}") from e
    return df


def categorical_cols(project: ProjectConfig, df: pd.DataFrame) -> list[str]:
    """Covariates treated as categorical: pinned in covariate_types, else
    inferred from a non-numeric dtype."""
    out = []
    for c in project.covariate_cols:
        if c not in df.columns:
            continue
        pinned = project.covariate_types.get(c)
        if pinned == "categorical" or (pinned is None and not pd.api.types.is_numeric_dtype(df[c])
                                       and not pd.api.types.is_bool_dtype(df[c])):
            out.append(c)
    return out


def load_raw(project: ProjectConfig, path: str | Path | list | None = None,
             shard: tuple[int, int] | None = None) -> pd.DataFrame:
    """Raw rows, optionally only shard (i, n) of the series."""
    path = path or project.data_path
    # a shard may legitimately be empty (few shard_by groups): not an error
    return prepare_raw(read_table(path, keep=shard_filter(project, shard)), project, source=str(path),
                       allow_empty=shard is not None)


def prepare_raw(df: pd.DataFrame, project: ProjectConfig, source: str = "data",
                allow_empty: bool = False) -> pd.DataFrame:
    """User data -> canonical raw frame (series_id, ts, y, stockout, ...):
    derived columns, row filter, date window, stockout rule, negative-target
    rule, and duplicate aggregation, in that order."""
    df = apply_derived(df.copy(), project)
    if project.row_filter:
        df = df.query(project.row_filter)
    if project.stockout_expr:
        try:
            stock = df.eval(project.stockout_expr)
        except Exception as e:
            raise ValueError(f"stockout_expr {project.stockout_expr!r} failed: {e}") from e
        stock = pd.Series(stock, index=df.index).fillna(False).astype(bool)
    missing = [c for c in project.raw_columns if c not in df.columns]
    if missing:
        raise ValueError(f"{source}: columns declared in project.yaml are missing: {missing}")
    out = df[project.raw_columns].copy()
    if project.stockout_expr:
        out[STOCKOUT] = stock.to_numpy()
    elif project.in_stock_col:
        v = pd.to_numeric(out[project.in_stock_col], errors="coerce")
        out[STOCKOUT] = (v == 0).to_numpy()  # NaN (unknown) counts as in stock
    else:
        out[STOCKOUT] = False
    out[SERIES] = make_series_id(out, project.series_id_cols)
    out = out.rename(columns={project.timestamp_col: TS, project.target_col: Y})
    out[TS] = pd.to_datetime(out[TS]).dt.normalize()
    if project.start_date:
        out = out[out[TS] >= pd.Timestamp(project.start_date)]
    if project.end_date:
        out = out[out[TS] <= pd.Timestamp(project.end_date)]
    if out.empty and not allow_empty:
        raise ValueError(f"{source}: no rows left after filters")

    y = pd.to_numeric(out[Y], errors="coerce")
    if (y < 0).any():
        n = int((y < 0).sum())
        if project.negative_target == "error":
            raise ValueError(f"{source}: {n} negative targets (set negative_target: clip | nan | keep)")
        if project.negative_target == "clip":
            y = y.clip(lower=0)
        elif project.negative_target == "nan":
            y = y.where(y >= 0)
    out[Y] = y
    return _dedupe(out, project)


def _dedupe(raw: pd.DataFrame, project: ProjectConfig) -> pd.DataFrame:
    if project.duplicates == "error":
        return raw
    dup = raw.duplicated([SERIES, TS], keep=False)
    if not dup.any():
        return raw
    agg = {c: "last" for c in raw.columns if c not in (SERIES, TS, Y)}
    agg[STOCKOUT] = "max"
    g = raw[dup].groupby([SERIES, TS], sort=False)
    merged = g.agg(agg)
    # min_count=1: rows whose targets are all missing stay missing, not 0
    merged[Y] = g[Y].sum(min_count=1) if project.duplicates == "sum" else g[Y].agg(project.duplicates)
    merged = merged.reset_index()
    print(f"[data] {int(dup.sum()):,} duplicate (series, date) rows -> {len(merged):,} "
          f"({project.duplicates} of the target)")
    return pd.concat([raw[~dup], merged[raw.columns]], ignore_index=True)


def build_panel(raw: pd.DataFrame, project: ProjectConfig, end: pd.Timestamp | None = None) -> pd.DataFrame:
    """Raw rows -> complete daily grid per series.

    Each series runs from its first row to the panel's last date (or its own
    last row with grid_end: series). Days with no raw row get demand 0, and
    their covariates are filled per `covariate_fill`. Rows are placed by
    index arithmetic (offset of the series + days since its start), not by a
    merge.
    """
    if raw.empty:
        cols = [SERIES, TS, Y, STOCKOUT, *project.covariate_cols, *project.static_cols]
        return pd.DataFrame(columns=list(dict.fromkeys(cols)))
    if raw.duplicated([SERIES, TS]).any():
        n = int(raw.duplicated([SERIES, TS]).sum())
        raise ValueError(f"{n} duplicate (series, date) rows; check series_id_cols "
                         f"{project.series_id_cols} match the data's grain, or set "
                         "duplicates: sum | mean | max | first | last")

    sid = raw[SERIES].astype("category")
    codes = sid.cat.codes.to_numpy()
    n_series = len(sid.cat.categories)
    day = (raw[TS].to_numpy().astype("datetime64[D]")).astype(np.int64)

    start = np.full(n_series, np.iinfo(np.int64).max)
    np.minimum.at(start, codes, day)
    if project.grid_end == "global":
        # `end`: the whole dataset's last date, so every shard's grid ends on
        # the same day even if a shard has no row on it
        last = day.max() if end is None else max(int(np.datetime64(pd.Timestamp(end), "D").astype(np.int64)),
                                                  int(day.max()))
        end = np.full(n_series, last)
    else:
        end = np.full(n_series, np.iinfo(np.int64).min)
        np.maximum.at(end, codes, day)
    lengths = end - start + 1
    offsets = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    total = int(lengths.sum())
    pos = offsets[codes] + (day - start[codes])

    out_codes = np.repeat(np.arange(n_series), lengths)
    within = np.arange(total) - np.repeat(offsets, lengths)
    out_day = np.repeat(start, lengths) + within

    panel = pd.DataFrame({
        SERIES: pd.Categorical.from_codes(out_codes, categories=sid.cat.categories),
        TS: out_day.astype("datetime64[D]").astype("datetime64[ns]"),
    })

    y = np.full(total, 0.0 if project.missing_target == "zero" else np.nan, dtype=np.float32)
    y[pos] = raw[Y].to_numpy(dtype=np.float32)
    panel[Y] = y

    stock = np.zeros(total, dtype=bool)
    if STOCKOUT in raw.columns:
        stock[pos] = raw[STOCKOUT].to_numpy(dtype=bool)
    panel[STOCKOUT] = stock

    cats = set(categorical_cols(project, raw))
    ffill_cols = []
    cat_labels: dict[str, pd.Index] = {}
    for c in project.covariate_cols:
        if c in cats:
            # integer codes, never a Python string per row (memory at scale);
            # -1 = no value, filled like NaN
            codes, labels = pd.factorize(raw[c].astype("string"), use_na_sentinel=True)
            arr = np.full(total, np.nan, dtype=np.float32)
            arr[pos] = np.where(codes < 0, np.nan, codes).astype(np.float32)
            panel[c] = arr
            cat_labels[c] = pd.Index(labels.astype(str))
        else:
            fill = 0.0 if project.fill(c) == "zero" else np.nan
            arr = np.full(total, fill, dtype=np.float32)
            arr[pos] = pd.to_numeric(raw[c], errors="coerce").to_numpy(dtype=np.float32)
            panel[c] = arr
        if project.fill(c) == "ffill":
            ffill_cols.append(c)
    if ffill_cols:
        panel[ffill_cols] = panel.groupby(SERIES, observed=True)[ffill_cols].ffill()
    for c in cats:
        labels = cat_labels[c]
        if "" not in labels:
            labels = labels.append(pd.Index([""]))
        empty = labels.get_loc("")
        codes = panel[c].to_numpy()
        codes = np.where(np.isnan(codes), empty, codes).astype(np.int32)
        panel[c] = pd.Categorical.from_codes(codes, categories=labels)

    statics = [c for c in dict.fromkeys([*project.static_cols, project.demand_label_col]) if c]
    if statics:
        first = raw.sort_values(TS).drop_duplicates(SERIES).set_index(SERIES)[statics]
        first = first.reindex(sid.cat.categories)
        for c in statics:
            panel[c] = pd.Categorical(first[c].astype(str).to_numpy()[out_codes])

    got, want = float(np.nansum(panel[Y].to_numpy(np.float64))), float(raw[Y].sum())
    if not np.isclose(got, want, rtol=1e-6, atol=1e-3):
        raise AssertionError(f"target mass changed building the grid: {want} -> {got}")

    return panel


def load_panel(project: ProjectConfig, path: str | Path | None = None,
               shard: tuple[int, int] | None = None, end: pd.Timestamp | None = None) -> pd.DataFrame:
    return build_panel(load_raw(project, path, shard), project, end=end)


def date_span(project: ProjectConfig, path: str | Path | list | None = None,
              with_series: bool = False):
    """(first, last) date of the whole dataset after filters, by one streaming
    pass: shards need the same fold cutoffs and grid end. With with_series,
    also the set of every series id (to tell a new SKU from one whose
    history lives in another shard)."""
    lo, hi, ids = [], [], set()

    def collect(df: pd.DataFrame) -> np.ndarray:
        try:
            r = prepare_raw(df, project)
        except ValueError as e:
            if "no rows left" not in str(e):
                raise
            return np.zeros(len(df), dtype=bool)
        lo.append(r[TS].min())
        hi.append(r[TS].max())
        if with_series:
            ids.update(r[SERIES].astype(str).unique())
        return np.zeros(len(df), dtype=bool)

    read_table(path or project.data_path, keep=collect)
    if not lo:
        raise ValueError("no rows in the data after filters")
    return (min(lo), max(hi), ids) if with_series else (min(lo), max(hi))


def demand_classes(panel: pd.DataFrame, project: ProjectConfig, end: pd.Timestamp | None = None) -> pd.Series:
    """Demand class per series. The user's label column wins; otherwise the
    Syntetos-Boylan classes (smooth, erratic, intermittent, lumpy) from
    ADI and CV^2 of non-zero demand, using data <= `end` only."""
    if project.demand_label_col:
        lab = panel.drop_duplicates(SERIES).set_index(SERIES)[project.demand_label_col]
        return lab.astype(str).rename("demand_class")
    p = panel if end is None else panel[panel[TS] <= end]
    p = p[~p[STOCKOUT]]
    g = p.groupby(SERIES, observed=True)[Y]
    n = g.size()
    nz = p[p[Y] > 0].groupby(SERIES, observed=True)[Y]
    n_nz = nz.size().reindex(n.index, fill_value=0)
    mean_nz = nz.mean().reindex(n.index)
    std_nz = nz.std(ddof=0).reindex(n.index)
    adi = n / n_nz.replace(0, np.nan)
    cv2 = (std_nz / mean_nz) ** 2
    cls = np.select(
        [n_nz == 0, (adi < 1.32) & (cv2 < 0.49), adi < 1.32, cv2 < 0.49],
        ["no_demand", "smooth", "erratic", "intermittent"],
        default="lumpy",
    )
    out = pd.Series(cls, index=n.index, name="demand_class")
    all_ids = panel[SERIES].cat.categories if hasattr(panel[SERIES], "cat") else panel[SERIES].unique()
    return out.reindex(all_ids, fill_value="no_history")
