# Forecast-FM

Chronos-2 demand forecasting for ecommerce replenishment. Daily demand for
~700k series, 35-day and 90-day lead times, price/promo plans as known
covariates. The target setup is **an Apple Silicon Mac (MPS)**; CUDA and CPU
also work.

- `docs/SPEC.md`: what to build, in what order, with status
- `CLAUDE.md`: the rules for working in this repo (pipeline, leakage, ledger)

## Setup on a Mac (Apple Silicon)

```bash
# 1. arm64 Python 3.10+ (NOT an x86 Python under Rosetta: it cannot see MPS)
python3 -c "import platform; print(platform.machine())"   # must print arm64

# 2. environment
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[foundation,dev]"     # the default PyPI torch wheel has MPS

# 3. check
python -m forecast_fm env              # expect "mps": true, "auto_device": "mps"
pytest                                 # all green, including the tiny Chronos-2 tests

# 4. Chronos-2 weights (downloaded on first use; or pin a local copy)
huggingface-cli download amazon/chronos-2 --local-dir models/chronos-2
#    then set model_params.model_id: models/chronos-2
```

Mac notes:
- `device: auto` picks CUDA, then MPS, then CPU. `PYTORCH_ENABLE_MPS_FALLBACK=1` is
  set automatically, so an op MPS lacks falls back to CPU instead of failing.
- `dtype: bfloat16` on MPS needs macOS 14+. Keep `float32` until a parity
  check on the sample shows bf16 matches it.
- Long runs: `caffeinate -i python -m forecast_fm run ...` stops the Mac
  sleeping mid-backtest.
- Unified memory is shared by the panel and the model. A 30k-series sample fits
  easily in 32 GB. All 700k series x 4 years (~1B rows) do not fit as one
  panel, so the full run is sharded by series (see `docs/SPEC.md`, WP4).

## Configuring `project.yaml`

Every option is documented in `project.yaml`. Beyond the columns and covariate classes:

| need | option |
|---|---|
| several files / a glob / a parquet folder | `data_path: [a.parquet, "parts/*.csv"]` |
| compute a column (`1 - price/regular_price`) | `derived_columns` (also applied to plan files) |
| keep a subset (channel, country, date window) | `row_filter`, `start_date`, `end_date`, `min_history_days` |
| repeated (series, date) rows | `duplicates: sum \| mean \| max \| first \| last` |
| returns / missing days | `negative_target`, `missing_target` |
| stockouts from stock levels | `stockout_expr: "stock_on_hand <= 0"` |
| plan files with other column names | `plan_as_of_col`, `plan_timestamp_col`, `plan_columns` |
| stale or incomplete plans | `plan_max_age_days`, `min_plan_coverage` |
| fixed backtest dates | `cutoffs`, `holdout_cutoffs` |
| decision quantile | `service_level` |
| metrics by category / brand | `slice_cols` |
| machine settings for every experiment | `model_defaults: {chronos2: {device: mps}}` |
| output locations | `reports_dir`, `models_dir` |

Layering: `extends: base.yaml`, named `profiles:` (`--profile full`), command-line
overrides (`--set horizon=35 --set covariate_eval_policy.price=carry_forward`) and
`${ENV_VAR:-default}` in any string. Check the result with:

```bash
python -m forecast_fm --profile full config --check-data
```

## Workflow

```bash
python -m forecast_fm sample --src data/raw/sales_full.parquet --n 30000 \
    --by demand_label --volume-bins 10 --seed 42        # stratified sample + manifest
python -m forecast_fm audit                              # -> reports/data_audit.md
python -m forecast_fm cutoffs                            # export plan snapshots as_of these dates
python -m forecast_fm run configs/01_seasonal_naive.yaml --no-commit   # debug run
python -m forecast_fm run configs/01_seasonal_naive.yaml               # ledgered run (+ git commit)
python -m forecast_fm leaderboard
python -m forecast_fm forecast configs/03_chronos2_zero_shot.yaml      # production forecast
```

### Fine-tune once, forecast many times

After the backtest shows the recipe works (a config with `fine_tune:` is
retrained per fold in `run`), train it once on all history and save it:

```bash
python -m forecast_fm finetune configs/05_chronos2_lora.yaml            # -> models/chronos2-lora-<date>/
python -m forecast_fm forecast models/chronos2-lora-<date>/forecast_config.yaml   # every day/week
```

The output directory holds the weights (`finetuned-ckpt/`), a provenance
manifest (`forecast_fm_model.json`: base model, recipe, training window,
data fingerprint, code commit, environment) and a ready `forecast_config.yaml`.
Existing checkpoints are never overwritten unless you pass `--force`. Retrain on your own
schedule with a new `--as-of`. Each `forecast` reports the checkpoint's age
in days.

A saved checkpoint refuses to forecast from any date before its `train_end`,
so it can never be backtested on data it has already seen. To measure a
recipe's accuracy, use `run` with `fine_tune:` in the config.

### Try it on synthetic data

```bash
python examples/make_demo_data.py
python -m forecast_fm -p examples/demo_project.yaml audit
python -m forecast_fm -p examples/demo_project.yaml run configs/01_seasonal_naive.yaml --no-commit
python -m forecast_fm -p examples/demo_project.yaml run configs/03_chronos2_zero_shot.yaml --no-commit
```

## Layout

| path | what |
|---|---|
| `forecast_fm/data.py` | load raw data, build the daily grid with vectorized index arithmetic, demand classes |
| `forecast_fm/sample.py` | streaming stratified sampler (WP2) |
| `forecast_fm/folds.py` | fold cutoffs (validation + holdout, no overlap) |
| `forecast_fm/plans.py` | horizon covariates by policy: `plan` / `actual` / `carry_forward`; plan snapshots via `as_of` |
| `forecast_fm/backtest.py` | rolling-origin harness; the model never receives post-cutoff targets |
| `forecast_fm/metrics.py` | WAPE, bias, MASE, wQL, quantile coverage by fold / horizon bucket / demand class |
| `forecast_fm/models/` | `naive`, `seasonal_naive`, `croston`, `chronos2` |
| `forecast_fm/device.py` | CUDA / MPS / CPU and dtype selection |
| `forecast_fm/finetune.py` | production fine-tuning: a saved checkpoint + manifest + forecast config |
| `forecast_fm/ledger.py` | append-only `experiments/`, verdicts vs `based_on` |
| `configs/` | experiment configs (one hypothesis each) |
| `project.yaml` | the fixed project policy (columns marked TODO) |
