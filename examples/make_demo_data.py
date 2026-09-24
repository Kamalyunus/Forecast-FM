"""Synthetic ecommerce panel + plan snapshots for trying the pipeline end to
end: `python examples/make_demo_data.py` then
`python -m forecast_fm -p examples/demo_project.yaml audit`.

Plan snapshots are written at the demo project's fold cutoffs with some
noise versus realized prices (plans change after they are issued)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from forecast_fm.config import load_project  # noqa: E402
from forecast_fm.folds import fold_cutoffs  # noqa: E402

OUT = Path("data/demo")


def make(n_series: int = 300, days: int = 1100, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(raw sales rows, full daily grid). Intermittent series only have rows
    on sale days, like a typical order extract; plans cover every day."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-01", periods=days, freq="D")
    t = np.arange(days)
    event = np.zeros(days)
    for d in dates[(dates.month == 11) & (dates.day >= 24) & (dates.day <= 28)]:
        event[(dates == d)] = 2
    frames, full = [], []
    labels = rng.choice(["BAU", "seasonal", "intermittent", "promo", "event"], n_series,
                        p=[0.45, 0.2, 0.2, 0.1, 0.05])
    for i in range(n_series):
        lab = labels[i]
        start = int(rng.integers(0, 200))
        level = rng.lognormal(1.5, 0.8)
        weekly = 1 + 0.25 * np.sin(2 * np.pi * (t + i) / 7)
        yearly = 1 + (0.6 if lab == "seasonal" else 0.15) * np.sin(2 * np.pi * (t - 80) / 365.25)
        promo = (rng.random(days) < (0.08 if lab == "promo" else 0.02)).astype(int)
        promo = np.convolve(promo, np.ones(5), "same").clip(0, 1).astype(int)  # 5-day promos
        regular = round(float(rng.uniform(5, 60)), 2)
        discount = promo * rng.choice([0.1, 0.2, 0.3])
        price = np.round(regular * (1 - discount), 2)
        lift = (1 + 3 * discount) * (1 + (1.5 if lab == "event" else 0.4) * event)
        lam = level * weekly * yearly * lift
        y = rng.poisson(lam).astype(float)
        if lab == "intermittent":
            y = y * (rng.random(days) < 0.15)
        in_stock = (rng.random(days) > 0.02).astype(int)
        y = y * in_stock
        sessions = rng.poisson(lam * 20 + 50)
        df = pd.DataFrame({
            "date": dates, "sku": f"SKU{i:05d}", "units": y, "price": price,
            "discount_pct": discount, "promo_flag": promo,
            "promo_type": np.where(promo > 0, "pct_off", "none"), "sitewide_event": event,
            "sessions": sessions, "in_stock": in_stock,
            "category": f"cat{i % 8}", "brand": f"brand{i % 25}", "demand_label": lab,
        }).iloc[start:]
        full.append(df)
        frames.append(df[df["units"] > 0] if lab == "intermittent" else df)
    return pd.concat(frames, ignore_index=True), pd.concat(full, ignore_index=True)


def plans(panel: pd.DataFrame, cutoffs: list[pd.Timestamp], horizon: int, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = []
    cols = ["price", "discount_pct", "promo_flag", "promo_type"]
    for c in cutoffs:
        in_window = (panel["date"] > c) & (panel["date"] <= c + pd.Timedelta(days=horizon))
        w = panel.loc[in_window, ["date", "sku", *cols]].copy()
        changed = rng.random(len(w)) < 0.1  # 10% of plan rows differ from what happened
        w.loc[changed, "promo_flag"] = 0
        w.loc[changed, "discount_pct"] = 0.0
        w.loc[changed, "promo_type"] = "none"
        w.insert(0, "as_of", c)
        out.append(w)
    return pd.concat(out, ignore_index=True)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    project = load_project("examples/demo_project.yaml")
    raw, full = make()
    raw.to_parquet(OUT / "sales.parquet", index=False)
    first, last = raw["date"].min(), raw["date"].max()
    cut = fold_cutoffs(first, last, project) + fold_cutoffs(first, last, project, holdout=True)
    plans(full, cut, project.horizon).to_parquet(OUT / "plan_snapshots.parquet", index=False)
    print(f"{raw['sku'].nunique()} series, {len(raw):,} rows -> {OUT}/sales.parquet; "
          f"plan snapshots at {len(cut)} cutoffs -> {OUT}/plan_snapshots.parquet")
