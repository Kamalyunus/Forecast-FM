import pandas as pd
import pyarrow.parquet as pq

from forecast_fm.sample import allocate, run_sample, series_stats, stratified_sample

from .conftest import make_project, make_raw


def _raw_file(tmp_path, n_series=60, fmt="parquet"):
    raw = make_raw(n_series=n_series, n_days=20)
    raw.loc[raw["sku"].isin([f"S{i}" for i in range(3)]), "demand_label"] = "event"  # rare class
    path = tmp_path / f"raw.{fmt}"
    raw.to_parquet(path, index=False) if fmt == "parquet" else raw.to_csv(path, index=False)
    return raw, path


def test_allocate_floor_and_total():
    sizes = pd.Series({"a": 1000, "b": 50, "c": 3})
    q = allocate(sizes, 300, floor=20)
    assert q.sum() == 300
    assert q["c"] == 3 and q["b"] >= 20 and (q <= sizes).all()


def test_same_seed_same_sample_and_floor(tmp_path):
    _, path = _raw_file(tmp_path)
    stats = series_stats(path, ["sku"], "units", "demand_label", batch_rows=100)
    a = stratified_sample(stats, 40, "demand_label", 3, floor=5, seed=42)
    b = stratified_sample(stats, 40, "demand_label", 3, floor=5, seed=42)
    c = stratified_sample(stats, 40, "demand_label", 3, floor=5, seed=7)
    pd.testing.assert_frame_equal(a, b)
    assert not a["series_id"].equals(c["series_id"])
    assert len(a) == 40
    # every stratum gets at least min(floor, size): the 3-series class keeps all 3
    assert (a["stratum"].str.startswith("event")).sum() == 3
    assert (a.groupby("stratum").size() >= 3).all()


def test_output_schema_matches_input_and_rows_complete(tmp_path):
    raw, path = _raw_file(tmp_path)
    out, man = tmp_path / "s.parquet", tmp_path / "m.csv"
    picked = run_sample(make_project(data_path=str(path)), 15, "demand_label", 2, 2, 1, out, man)
    got = pq.read_table(out)
    assert got.schema.names == list(raw.columns)
    df = got.to_pandas()
    assert set(df["sku"]) == set(picked["series_id"])
    assert len(df) == len(raw[raw["sku"].isin(picked["series_id"])])
    m = pd.read_csv(man)
    assert {"series_id", "stratum", "weight"} <= set(m.columns)


def test_csv_input(tmp_path):
    _, path = _raw_file(tmp_path, fmt="csv")
    picked = run_sample(make_project(data_path=str(path)), 10, None, 2, 1, 0,
                        tmp_path / "s.parquet", tmp_path / "m.csv")
    assert len(picked) == 10
