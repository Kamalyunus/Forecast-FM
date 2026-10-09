# Forecast-FM

Demand forecasting for ecommerce replenishment with **Chronos-2**, a
time-series foundation model: zero-shot first, then **fine-tuned on your own
sales history**. Daily demand, 90-day horizon reported at the 35-day and 90-day
lead times, price and promo plans as known covariates, new-SKU forecasts, and
a leakage-safe backtest with an append-only experiment ledger. Built to run on
an **Apple Silicon Mac**; CUDA and CPU also work.

| Read this | When |
|---|---|
| this README | first hour: install, try the demo, point it at your data |
| [`docs/DATA_REQUIREMENTS.md`](docs/DATA_REQUIREMENTS.md) | exporting the data: files, columns, formats (templates in `docs/data_templates/`) |
| [`docs/FINE_TUNING.md`](docs/FINE_TUNING.md) | fine-tuning: what the knobs do, cost, millions of SKUs |
| [`docs/SPEC.md`](docs/SPEC.md) | what is built, what is not, in what order |
| [`CLAUDE.md`](CLAUDE.md) | the rules for working in this repo (pipeline, leakage, ledger) |

## 1. Install

```bash
# arm64 Python 3.10+ (an x86 Python under Rosetta cannot see the GPU)
python3 -c "import platform; print(platform.machine())"     # must print arm64 on a Mac

python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[foundation,dev]"       # torch + chronos-forecasting + peft + pytest + ruff
python -m forecast_fm env                # expect "mps": true, "auto_device": "mps"
pytest                                   # green (about 2 minutes on the first run)
```

The Chronos-2 weights (`amazon/chronos-2`, about 500 MB) download from Hugging
Face on first use. No internet on the work machine? Copy them once:

```bash
hf download amazon/chronos-2 --local-dir models/chronos-2      # `hf` ships with huggingface_hub
# then in project.yaml:  model_defaults: {chronos2: {model_id: models/chronos-2}}
```

`python -m forecast_fm --help` lists the commands in the order you will use
them, and `<command> --help` explains every flag.

## 2. Try it in five minutes (synthetic data)

```bash
python examples/make_demo_data.py                      # -> data/demo/ (288 SKUs, 3 years, plans, 12 upcoming SKUs)
export FORECAST_FM_PROJECT=examples/demo_project.yaml  # instead of -p ... on every command

python -m forecast_fm audit                            # what the data looks like -> reports/data_audit.md
python -m forecast_fm run configs/01_seasonal_naive.yaml --no-commit      # the baseline (seconds)
python -m forecast_fm run configs/03_chronos2_zero_shot.yaml --no-commit  # Chronos-2, no training
python -m forecast_fm forecast configs/03_chronos2_zero_shot.yaml         # the production file
```

`--no-commit` is a debug run: results go to `reports/scratch/`, nothing is
ledgered. Each run prints metrics by fold, horizon bucket (`h01-35`,
`h36-90`), lifecycle and demand class. `forecast` writes one row per SKU × day,
new SKUs included, under `reports/forecast_<date>/`.

## 3. Your own data

1. **Export the files** described in [`docs/DATA_REQUIREMENTS.md`](docs/DATA_REQUIREMENTS.md):
   sales history (required), plan snapshots (prices and promos as they were
   planned on each forecast date), and optionally upcoming SKUs. Put them under
   `data/raw/`, the full history as `data/raw/sales_full.parquet`. The default
   `data_path` is the experiment *sample* (step 5); `--profile full` switches
   every command to the full file.
2. **Fill in `project.yaml`.** Every `TODO` marks a column or policy that is
   yours to set: column names, which covariates are *known* in advance and
   which are only *observed*, the stockout rule, the service level. The file
   explains each option in place. Check what the code will use:
   ```bash
   unset FORECAST_FM_PROJECT                                # back to ./project.yaml
   python -m forecast_fm --profile full config --check-data # resolved settings; every declared column present?
   ```
3. **Audit** the data and read the report with whoever owns the data:
   ```bash
   python -m forecast_fm --profile full audit               # -> reports/data_audit.md
   ```
   Write down what the columns mean in `experiments/DATA_NOTES.md`.
4. **Get plan snapshots for the backtest dates.** Backtests must only see what
   was known at the time, so each fold needs the plan as it stood on its
   cutoff. `python -m forecast_fm --profile full cutoffs` lists the dates.
5. **Draw the experiment sample** from the full catalog (experiments run on
   20–50k series, the final forecast on everything):
   ```bash
   python -m forecast_fm --profile full sample --n 30000 \
       --by demand_label --until <first cutoff>   # -> data/raw/sales_sample.parquet (the default data_path)
   ```
6. **Run the baseline**, then the rest of [section 4](#4-experiments). Without
   `--profile full`, every command now uses the sample.

Mistakes stop early with one line and a hint (a wrong column name, a missing
file, a duplicated YAML key, a `based_on` that is not in the ledger). Set
`FORECAST_FM_DEBUG=1` for the full traceback.

## 4. Experiments

One hypothesis per config, each compared against the run it is `based_on`.
The configs in `configs/` are the planned sequence:

| config | tests |
|---|---|
| `01_seasonal_naive` | the bar every model must beat (the reference) |
| `03_chronos2_zero_shot` | Chronos-2 with all covariates, no training |
| `04_chronos2_group_by` | cross-learning within a category |
| `05_chronos2_lora` | LoRA fine-tuning on fold history |
| `06a/b/c_ft_*` | which SKUs to fine-tune on: continuous only / natural mix / capped intermittent |
| `07_chronos2_lora_rounds` | the full recipe: 50k steps, 10 rounds of fresh SKUs, fixed 2-year window |

```bash
python -m forecast_fm run configs/01_seasonal_naive.yaml            # ledgered: experiments/exp001-*, git commit
# configs/03 has based_on: exp001; check the id in `leaderboard` before each later run
python -m forecast_fm run configs/03_chronos2_zero_shot.yaml
python -m forecast_fm leaderboard
```

A ledgered run refuses to start while `forecast_fm/` or `project.yaml` has
uncommitted changes, so every result is reproducible. The verdict is
`improved` / `regressed` / `inconclusive` (the change must hold across folds)
or `incomparable` (different data, policy, horizon or cutoffs than the
reference). `experiments/` is append-only.

**Fine-tuned recipes are measured the same way:** a config with `fine_tune:`
is retrained inside every fold. Measure the time first:

```bash
python -m forecast_fm bench configs/07_chronos2_lora_rounds.yaml --catalog 3000000 --memory-gb 24
# seconds/step and series/s on this machine -> hours per fine-tune, backtest and production forecast
```

Default folds retrain every 91 days with one forecast per fold. Set
`origin_step_days: 7` to forecast weekly from each fold's model until the
next retrain, as production will: more evidence per fine-tune, plus a
`model_age` table (accuracy against weeks since retraining) that tells you how
often to retrain.

## 5. Fine-tune once, forecast on a schedule

When the backtest says the recipe wins, train it once on all history and save it:

```bash
python -m forecast_fm finetune configs/07_chronos2_lora_rounds.yaml          # -> models/chronos2-lora-rounds-<date>/
python -m forecast_fm forecast models/chronos2-lora-rounds-<date>/forecast_config.yaml --shards 16
```

The model directory holds the weights, a provenance manifest (base model,
recipe, training window, data fingerprint, code commit) and a ready
`forecast_config.yaml`. A saved checkpoint refuses to forecast from any date
before its training end, so it can never be scored on data it has seen.
Retrain with a new `--as-of`; existing checkpoints are never overwritten
without `--force`. Everything about the recipe, cost, training on millions of
SKUs, and the fixed training window: [`docs/FINE_TUNING.md`](docs/FINE_TUNING.md).

### The production file

`forecast` writes one row per series × day × horizon step for **every** series:

```
origin, series_id, <your id cols>, ts, horizon, y_pred, q_0.1, ..., q_0.95, lifecycle, model
```

`lifecycle` says how each series was forecast: `established` by the model,
`short_history` (less than `min_history_days`) and `new` (not launched yet) by a
launch profile from past launches in the same category. New SKUs come from
`cold_start.new_series_path` and from plan-snapshot SKUs with no history.

The output is a directory of parquet parts; `pandas.read_parquet(dir)` reads
it all. For a catalog that does not fit in memory:

```bash
python -m forecast_fm forecast <config> --shards 16                 # one shard in memory at a time
python -m forecast_fm forecast <config> --shards 16 --shard 0 &     # or parallel processes, one shard each
```

Shards stream only their own rows, finished parts are skipped on rerun
(`--force` redoes them), and `shard_by: [category]` keeps a category in one
shard so cross-learning and launch profiles see the whole group. `run --shards N`
gives metrics identical to an unsharded backtest.

## 6. Configuring `project.yaml`

Every option is documented in the file itself. Beyond the columns and covariate classes:

| need | option |
|---|---|
| several files / a glob / a parquet folder | `data_path: [a.parquet, "parts/*.csv"]` |
| compute a column (`1 - price/regular_price`) | `derived_columns` (also applied to plan files) |
| keep a subset (channel, country, date window) | `row_filter`, `start_date`, `end_date` |
| stockouts from hours out of stock or stock levels | `stockout_expr: "oos_hours >= 12"`, `in_stock_col` |
| new SKUs / short histories | `min_history_days`, `cold_start` |
| large catalogs | `shard_by`, `--shards N` |
| repeated (series, date) rows | `duplicates: sum \| mean \| max \| first \| last` |
| returns / missing days | `negative_target`, `missing_target` |
| plan files with other column names | `plan_as_of_col`, `plan_timestamp_col`, `plan_columns` |
| stale or incomplete plans | `plan_max_age_days`, `min_plan_coverage` |
| production-like backtest (retrain vs forecast dates) | `origin_step_days` |
| fixed backtest dates | `cutoffs`, `holdout_cutoffs` |
| decision quantile | `service_level` |
| metrics by category / brand | `slice_cols` |
| machine settings for every experiment | `model_defaults: {chronos2: {device: mps}}` |
| output locations | `reports_dir`, `models_dir` |

Layering: `extends: base.yaml`, named `profiles:` (`--profile full`),
command-line overrides (`--set horizon=35 --set covariate_eval_policy.price=carry_forward`)
and `${ENV_VAR:-default}` in any string. `-p` picks the project file
(`$FORECAST_FM_PROJECT` sets the default). Duplicate keys are an error rather
than a silent override.

## Mac notes

- `device: auto` picks CUDA, then MPS, then CPU. `PYTORCH_ENABLE_MPS_FALLBACK=1`
  is set automatically, so an op MPS lacks falls back to CPU instead of failing.
- `dtype: bfloat16` on MPS needs macOS 14+. Keep `float32` until a parity check
  on the sample shows bf16 matches it.
- Long runs: `caffeinate -i python -m forecast_fm run ...` stops the Mac
  sleeping mid-backtest.
- Unified memory is shared by the data and the model. A 30k-series sample fits
  easily in 32 GB; the full catalog does not fit as one table, so the full run
  is sharded (`--shards`, sized by `bench`).

## Layout

| path | what |
|---|---|
| `forecast_fm/cli.py` | the commands |
| `forecast_fm/config.py` | `project.yaml` and experiment configs: sections, profiles, overrides, validation |
| `forecast_fm/data.py` | load raw data, build the daily grid, demand classes, streaming and sharding |
| `forecast_fm/sample.py` | streaming stratified sampler |
| `forecast_fm/folds.py` | retrain dates (fold cutoffs), forecast origins, holdout |
| `forecast_fm/plans.py` | horizon covariates by policy: `plan` / `actual` / `carry_forward`; plan snapshots via `as_of` |
| `forecast_fm/backtest.py` | rolling-origin harness; the model never receives post-cutoff targets |
| `forecast_fm/metrics.py` | WAPE, bias, MASE, wQL, quantile coverage by fold / horizon bucket / class / model age |
| `forecast_fm/models/` | `naive`, `seasonal_naive`, `croston`, `chronos2` (zero-shot, fine-tuning, fixed window) |
| `forecast_fm/train_mix.py` | which series a fine-tune trains on: class filter, share caps, activity, rounds |
| `forecast_fm/cold_start.py` | new / short-history series: launch profiles from analogs |
| `forecast_fm/runner.py` | sharded backtest and forecast orchestration (the long daily file) |
| `forecast_fm/finetune.py` | production fine-tuning: a saved checkpoint + manifest + forecast config |
| `forecast_fm/bench.py` | time training and forecasting; estimate the full plan |
| `forecast_fm/ledger.py` | append-only `experiments/`, verdicts vs `based_on` |
| `forecast_fm/device.py` | CUDA / MPS / CPU and dtype selection |
| `configs/` | experiment configs (one hypothesis each) |
| `examples/` | synthetic demo data generator and its project file |
| `project.yaml` | the fixed project policy (columns marked TODO) |
