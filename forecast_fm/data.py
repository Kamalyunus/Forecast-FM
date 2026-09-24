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


def read_table(path: str | Path, columns: list[str] | None = None) -> pd.DataFrame:
    path = Path(path)
    if path.suffix in (".parquet", ".pq") or path.is_dir():
        return pd.read_parquet(path, columns=columns)
    if path.suffix == ".csv":
        return pd.read_csv(path, usecols=columns)
    raise ValueError(f"unsupported data file {path} (csv or parquet)")


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


def load_raw(project: ProjectConfig, path: str | Path | None = None) -> pd.DataFrame:
    path = path or project.data_path
    return prepare_raw(read_table(path), project, source=str(path))


def prepare_raw(df: pd.DataFrame, project: ProjectConfig, source: str = "data") -> pd.DataFrame:
    """User column names -> canonical raw frame (series_id, ts, y, ...)."""
    missing = [c for c in project.raw_columns if c not in df.columns]
    if missing:
        raise ValueError(f"{source}: columns declared in project.yaml are missing: {missing}")
    df = df[project.raw_columns].copy()
    df[SERIES] = make_series_id(df, project.series_id_cols)
    df = df.rename(columns={project.timestamp_col: TS, project.target_col: Y})
    df[TS] = pd.to_datetime(df[TS]).dt.normalize()
    return df


def build_panel(raw: pd.DataFrame, project: ProjectConfig) -> pd.DataFrame:
    """Raw rows -> complete daily grid per series.

    Each series runs from its first row to the panel's last date (or its own
    last row with grid_end: series). Days with no raw row get demand 0, and
    their covariates are filled per `covariate_fill`. Rows are placed by
    index arithmetic (offset of the series + days since its start), not by a
    merge.
    """
    if project.freq != "D":
        raise ValueError("only daily data (freq: D) is supported")
    if raw.duplicated([SERIES, TS]).any():
        n = int(raw.duplicated([SERIES, TS]).sum())
        raise ValueError(f"{n} duplicate (series, date) rows; check series_id_cols "
                         f"{project.series_id_cols} match the data's grain")

    sid = raw[SERIES].astype("category")
    codes = sid.cat.codes.to_numpy()
    n_series = len(sid.cat.categories)
    day = (raw[TS].to_numpy().astype("datetime64[D]")).astype(np.int64)

    start = np.full(n_series, np.iinfo(np.int64).max)
    np.minimum.at(start, codes, day)
    if project.grid_end == "global":
        end = np.full(n_series, day.max())
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

    y = np.zeros(total, dtype=np.float32)
    y[pos] = raw[Y].to_numpy(dtype=np.float32)
    panel[Y] = y

    stock = np.zeros(total, dtype=bool)
    if project.in_stock_col:
        v = pd.to_numeric(raw[project.in_stock_col], errors="coerce").to_numpy()
        stock[pos] = v == 0  # NaN (unknown) counts as in stock
    panel[STOCKOUT] = stock

    cats = set(categorical_cols(project, raw))
    ffill_cols = []
    for c in project.covariate_cols:
        if c in cats:
            vals = raw[c].astype("string").astype(object)
            arr = np.full(total, None, dtype=object)
            arr[pos] = vals.where(vals.notna(), None).to_numpy()
            panel[c] = arr
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
        panel[c] = panel[c].fillna("").astype("category")

    statics = [c for c in dict.fromkeys([*project.static_cols, project.demand_label_col]) if c]
    if statics:
        first = raw.sort_values(TS).drop_duplicates(SERIES).set_index(SERIES)[statics]
        first = first.reindex(sid.cat.categories)
        for c in statics:
            panel[c] = pd.Categorical(first[c].astype(str).to_numpy()[out_codes])

    got, want = float(panel[Y].to_numpy(np.float64).sum()), float(raw[Y].sum())
    if not np.isclose(got, want, rtol=1e-6, atol=1e-3):
        raise AssertionError(f"target mass changed building the grid: {want} -> {got}")
    return panel


def load_panel(project: ProjectConfig, path: str | Path | None = None) -> pd.DataFrame:
    return build_panel(load_raw(project, path), project)


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
