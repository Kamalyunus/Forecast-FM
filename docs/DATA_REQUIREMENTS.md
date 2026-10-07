# Data requirements

What to export for Forecast-FM, and in what format. Hand this page to whoever
pulls the data. Example files with the right shape are in
[`docs/data_templates/`](data_templates/); a test loads them through the real
loaders, so they stay correct.

Column names below are the defaults in `project.yaml`. Yours can differ: map
them in `project.yaml` (`timestamp_col`, `target_col`, `series_id_cols`, the
covariate lists, `plan_columns`) instead of renaming them in the export.

## Overview

| # | File | Needed for | Required? |
|---|---|---|---|
| 1 | **Sales history** | everything | **yes** |
| 2 | **Plan snapshots** | backtests with `plan` policies; every production forecast | yes, unless every known covariate uses `carry_forward` |
| 3 | **New SKUs** | forecasts for SKUs launching after the last sales date | optional |
| 4 | Your current production forecast | "beats what we use today" comparisons | optional (not built yet) |

**Format, for all files:**
- Parquet (preferred) or CSV. A file, a folder of parquet files, a glob such as
  `data/raw/sales/part-*.parquet`, or a list of these.
- Dates as `YYYY-MM-DD`. Daily data only.
- One header row; column names exactly as declared in `project.yaml`.
- IDs as text. With more than one id column (`[sku, warehouse]`), the values
  must not contain `|`.
- Large exports: write many parquet part files rather than one huge CSV. Every
  command streams them.

---

## 1. Sales history (required)

**Grain:** one row per **series per day**. A series is one value of
`series_id_cols`, e.g. a SKU, or a SKU × warehouse.

**History:** at least **~1,185 days (about 3.3 years)** before the last date to
get all 4 validation folds plus the holdout with the default settings. With
less, the folds are reduced and a message says so. You have 4 years: fine.

**Days with no row are treated as zero sales** (`missing_target: zero`), so
you don't need to export zero days. If missing really means "unknown" in your
system, set `missing_target: nan`.

| Column (default name) | Type | Required | Meaning and rules |
|---|---|---|---|
| `date` | date | yes | The sales day. |
| `sku` (+ e.g. `warehouse`) | text | yes | The series key: the grain you order at. A (series, date) pair must be unique, or set `duplicates: sum`. |
| `units` | number | yes | Demand: units sold that day. Returns: net them out, or keep them separate and set `negative_target`. |
| `in_stock` | 0/1 | strongly recommended | **0 = stocked out that day.** Those days are hidden from the model and not scored, because sales were capped by stock rather than demand. Blank counts as in stock. Alternative: give stock on hand and set `stockout_expr: "stock_on_hand <= 0"`. |
| `price` | number | recommended | The selling price that day. |
| `discount_pct` | number 0–1 | recommended | `1 - price / regular_price`. Or export `regular_price` and set `derived_columns: {discount_pct: "1 - price / regular_price"}`. |
| `promo_flag` | 0/1 | recommended | A promotion was running. |
| `promo_type` | text | optional | E.g. `none`, `pct_off`, `bogo`. Text columns are treated as categories. |
| `sitewide_event` | number | optional | 0..N: Black Friday, paydays, holidays. The same value for every SKU on a day. |
| `sessions` | number | optional | Site or product-page sessions. Observed after the fact (a *past* covariate). |
| `category`, `brand` | text | recommended | Static attributes, one value per SKU. The first value seen is used. They drive metric slices, sharding (`shard_by`), cross-learning (`group_by`) and new-SKU profiles. |
| `demand_label` | text | optional | Your class per SKU: `intermittent`, `seasonal`, `BAU`, `event`, `promo`. If absent, classes are computed (smooth / erratic / intermittent / lumpy). Used for metric slices and `fine_tune.train_mix`. |

**Each extra column needs a class.** Declare it in `project.yaml` as one of:

| Class | Rule | Examples | What the model gets |
|---|---|---|---|
| **known** (`known_covariate_cols`) | You know the value for every future day at forecast time. | price, promo calendar, events | History, plus the 90 future days from the plan snapshot |
| **past** (`past_covariate_cols`) | Only known after the fact. | sessions, stock, **observed** weather | History only |
| **static** (`static_cols`) | One value per SKU. | category, brand, size | Used for groups and slices, not as a model input |

Every known covariate also needs a `covariate_eval_policy` entry: `plan`
(from snapshots), `actual` (only for fixed calendars), or `carry_forward`
(hold today's value flat). Leaving it out is an error.

Example: [`data_templates/sales_history.csv`](data_templates/sales_history.csv).

---

## 2. Plan snapshots (needed for honest backtests and for production)

What you had **planned**, as it stood on a given day, for the next 90 days:
prices, promos, the event calendar. Backtests must use what was known at the
time, not what happened later, so each snapshot is stamped with the date it
was issued.

**Grain:** one row per **snapshot date (`as_of`) × series × future day**.
Each snapshot covers `as_of + 1` through `as_of + 90`.

| Column | Type | Required | Meaning |
|---|---|---|---|
| `as_of` | date | yes | The day the plan was issued (the forecast date it is for). Rename with `plan_as_of_col`. |
| `date` | date | yes | The planned day, in `as_of + 1 .. as_of + 90`. Rename with `plan_timestamp_col`. |
| `sku` (+ other id cols) | text | yes | Same keys as the sales history. |
| one column per known covariate | as in sales | yes | `price`, `discount_pct`, `promo_flag`, `promo_type`, `sitewide_event`. Rename mismatched names with `plan_columns`, e.g. `{planned_price: price}`. |

**Which dates you need:**
- **Backtests:** only the dates printed by `python -m forecast_fm cutoffs` (4
  validation and 1 holdout), not every day.
- **Production:** one snapshot with `as_of` = the forecast date (the last
  sales date), covering **every known covariate**, including fixed calendars
  such as `sitewide_event`. Without one, the forecast stops with an error
  rather than guessing.

**Coverage:** a snapshot should hold every SKU you forecast, for all 90 days.
Below 99% per covariate a warning is printed (`min_plan_coverage`); missing
values are passed to the model as missing, never as 0.

**No archive of past plans?** Set `covariate_eval_policy` to `carry_forward`
for price and promo in backtests, and note the limitation in
`experiments/DATA_NOTES.md`. Production still needs today's plan.

**New SKUs in plans:** a SKU in today's snapshot with no sales history is
treated as a new launch, from its first planned day (`cold_start.from_plans`).

Example: [`data_templates/plan_snapshots.csv`](data_templates/plan_snapshots.csv).

---

## 3. New SKUs (optional, production)

SKUs launching after the last sales date get a **launch-profile forecast**:
the typical demand by days-since-launch of past launches in the same category.
Set `cold_start.new_series_path` to this file.

**Grain:** one row per upcoming SKU.

| Column | Type | Required | Meaning |
|---|---|---|---|
| `sku` (+ other id cols) | text | yes | Same keys as the sales history. |
| `launch_date` | date | recommended | First day on sale. Forecasts are 0 before it. If missing: the day after the forecast date. Rename with `cold_start.launch_date_col`. |
| `category` (the `profile_cols`) | text | recommended | Picks the analogs. Without it the SKU gets the all-SKU profile. |
| other statics (`brand`, ...) | text | optional | Used for sharding when listed in `shard_by`. |

Example: [`data_templates/new_skus.csv`](data_templates/new_skus.csv).

---

## 4. Your current production forecast (optional, recommended)

The forecast your company uses today, as it was issued at the backtest
cutoffs: `as_of, date, sku, forecast` (optionally quantiles). It is the
comparison management will ask for. The model that reads it isn't built yet;
export it now if it's easy, so the history exists.

---

## Before the first run: checklist

1. Put the files under `data/raw/` and fill in the `TODO`s in `project.yaml`.
2. `python -m forecast_fm config --check-data`: every declared column is present.
3. `python -m forecast_fm audit`: row counts, date range, zero share, stockout
   share, covariate types, demand classes, fold cutoffs. Review it, and write
   down what the columns mean in `experiments/DATA_NOTES.md`.
4. `python -m forecast_fm cutoffs`: export plan snapshots `as_of` these dates.
5. On the full catalog, draw the experiment sample:
   `python -m forecast_fm sample --src data/raw/sales_full.parquet --n 30000 --by demand_label --until <first cutoff>`.

## Questions to settle with the data owner

- **Grain:** SKU, or SKU × warehouse / channel? The order decision decides.
- **Demand:** gross units or net of returns? Are cancelled orders included?
- **Stockouts:** is there a daily in-stock flag or stock-on-hand? Without
  one, lost sales look like zero demand.
- **Zero days:** does a missing row mean 0 sales, or "no data"?
- **Plans:** are past price and promo plans archived by issue date? Since when?
- **Weather:** observed (a past covariate), or forecasts as issued (known, via plans)?
- **New SKUs:** where do launch dates and categories live before the first sale?
- **Volume:** how many rows and SKUs? This sets the number of shards
  (`--shards`) for the full 700k–3M run.

## What you get back

`python -m forecast_fm forecast <config> --shards N` writes a folder of parquet
parts (`pandas.read_parquet(folder)` reads all of them): one row per **SKU × day
× horizon step**.

| Column | Meaning |
|---|---|
| `origin` | The forecast date (the last sales date). |
| `series_id`, plus your id columns | The SKU (and warehouse, ...). |
| `ts`, `horizon` | The forecast day, and how many days ahead it is (1–90). |
| `y_pred` | Point forecast (Chronos-2: the median; new SKUs: the profile mean). |
| `q_0.1` … `q_0.95` | Quantiles (`quantiles` and `service_level` in `project.yaml`). |
| `lifecycle` | `established` (model), `short_history` or `new` (launch profile). |
| `model` | The experiment or checkpoint name that produced it. |
