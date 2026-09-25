# Spec: time-series foundation model for ecommerce replenishment

What to build and in what order. `CLAUDE.md` holds the rules for how work
is done here, and it wins over this file when they conflict.

This repo is a **fresh, lean rebuild** of the original handoff spec (written
for the `forecasting_assistant` workbench in Forecasting_Agent). It targets
an **Apple Silicon Mac (32 GB+ unified memory)** instead of a CUDA box.

Status legend: ✅ done · 🟡 partly done · 🔲 to do · ❓ blocked on the user

---

## 1. Problem

| | |
|---|---|
| Decision | Replenishment ordering |
| Lead times | **35 days** and **90 days**: one run at horizon 90, reported as `h01-35` / `h36-90` |
| Data | Daily demand, ~4 years, **~700k series** (ecommerce) |
| Demand classes (user labels) | intermittent, seasonal, BAU, event, promo |
| Covariates | known: price, promo (+ planned additions); past: weather, sessions, stock; static: category attributes |
| Model | **Chronos-2** (Apache-2.0): zero-shot first, then fine-tuned. TimesFM was rejected because its weights are non-commercial only. |
| Experiment scale | stratified sample of 20-50k series; the winning recipe then runs on all 700k |
| Machine | Apple Silicon Mac, MPS backend, 32 GB+ unified memory |

## 2. What exists (✅)

`pytest`: 97 tests (87 run without torch; 10 need the tiny real Chronos-2).

- ✅ **Data** (`data.py`):
  - The daily grid is built by index arithmetic (series offset + days since the series starts): no per-series loops and no merge.
  - Duplicate (series, date) rows raise an error.
  - Target-mass invariant check.
  - Fill rules per covariate (`covariate_fill`).
  - Stockouts come from `in_stock_col`.
  - Dtypes: series_id and categoricals are `category`; target and covariates are float32.
- ✅ **Demand classes**: the user's label column; otherwise Syntetos-Boylan (ADI / CV²), computed on data up to the first cutoff.
- ✅ **Folds** (`folds.py`): validation plus holdout cutoffs, with a gap so validation horizons never overlap the holdout window.
- ✅ **Horizon covariates** (`plans.py`):
  - Policy per column: `plan` / `actual` / `carry_forward`.
  - Plan snapshots come from csv or parquet with an `as_of` column. Each fold uses the latest snapshot with `as_of <= cutoff`.
  - Coverage is printed per covariate, with a warning below 99%.
  - Missing values stay NaN.
  - In production, `actual` columns are read from the plan snapshot.
- ✅ **Backtest** (`backtest.py`):
  - The model gets `history` (<= cutoff) and the future frame (keys + known covariates) only.
  - Targets are joined back for scoring after predict returns.
- ✅ **Metrics** (`metrics.py`, vectorized): WAPE, bias, MASE (seasonal-naive scale per fold), wQL, and coverage per quantile. Broken down by fold, horizon bucket, demand class, and bucket × class. Stockout days are not scored.
- ✅ **Models**: `naive`, `seasonal_naive`, `croston` (SBA; the recursion loops over days, vectorized across series), and `chronos2`:
  - Known covariates → history + horizon; past covariates → history only; categoricals → strings; statics → `group_by` cross-learning groups of ≤ `group_size`. Stockouts → NaN in the context.
  - `device: auto` picks CUDA, then MPS, then CPU. `PYTORCH_ENABLE_MPS_FALLBACK=1` is set.
  - `dtype: float32 | bfloat16 | float16`.
  - **Streaming predict**: inputs are built per chunk (`chunk_series`) or per group, into a preallocated `(n_series, H, n_q)` float32 array.
  - **Fine-tune checkpoint cache**: `reports/chronos2_ft/<key>/`. The key covers model_id, dtype, the fine_tune params, the covariate lists, the policy, horizon, cutoff, and a hash of the fold history. A different cutoff always gives a different key.
  - Run stats per fold: series/s, variates, predict seconds, accelerator memory.
- ✅ **Sampler** (WP2, `sample.py`): see below.
- ✅ **`cutoffs` command** and parquet plan loading (WP3).
- ✅ **Ledger** (`ledger.py`):
  - `experiments/expNNN-*/` holds the config, metrics.json, and slice CSVs; `LEDGER.md` gets one row per run; git commits `expNNN [verdict] …`.
  - Verdict rule: `improved` or `regressed` needs both a change larger than `verdict_threshold` and a majority of folds moving the same way. Anything else is `inconclusive`.
  - A ledgered run refuses to start when the code has uncommitted changes. `--no-commit` writes to `reports/scratch/`.
- ✅ **Production fine-tuning** (`finetune.py`, `python -m forecast_fm finetune <config>`):
  - Trains once on history <= `--as-of` (default: the last date).
  - Saves `models/<name>-<date>/` with `finetuned-ckpt/`, a `forecast_fm_model.json` manifest (base model, recipe and config hash, training window, data hash, code commit, env), and a ready `forecast_config.yaml`.
  - The output is staged in a `.partial` directory and never overwritten without `--force`.
  - The chronos2 model refuses a saved checkpoint at any cutoff before its `train_end`, checked before loading. Forecasts report the checkpoint's age in days.
- ✅ **Configurable `project.yaml`** (`config.py`):
  - Loading: sections, `extends`, `profiles` (`--profile`, `$FORECAST_FM_PROFILE`), `--set` dotted overrides, `${ENV}` interpolation, did-you-mean errors, and all validation errors reported at once.
  - Data: multi-file, glob and list inputs; `derived_columns`; `row_filter`; a date window; `min_history_days`; `duplicates`, `negative_target` and `missing_target` rules; `stockout_expr`.
  - Plans: column mapping, `plan_max_age_days`, `min_plan_coverage`.
  - Backtest and evaluation: explicit `cutoffs` / `holdout_cutoffs`, `service_level`, `slice_cols`.
  - Models and paths: `model_defaults` per model family, `reports_dir`, `models_dir`.
  - `python -m forecast_fm config [--check-data]` prints the resolved config.
  - Not supported: frequencies other than daily (`freq: D`).
- ✅ **Training-series mix** (`train_mix.py`, `fine_tune.train_mix`):
  - Options: a class filter, `max_share` caps per class, a near-dead-series filter (`min_nonzero_days` over `lookback_days`), a `max_series` cap that keeps the shares, and a seed.
  - Computed from each fold's history only. The realized mix is recorded in the fold stats, the checkpoint cache and the `finetune` manifest.
  - `configs/06a-c` set up the study: continuous-only (A), natural mix (B), and intermittent capped at 25% (C).
- ✅ **New and short-history series** (`cold_start.py`):
  - Launch profiles from analogs (series whose launch is observed): the mean and quantiles by days-since-launch, pooled by `profile_cols` with fallback to coarser groups.
  - `min_history_days` is decided at each origin: series below it go to cold start, scaled by their own sales so far.
  - New SKUs: in backtests, series first seen within the horizon; in production, `new_series_path` and plan-snapshot SKUs with no history.
  - Scored as a `lifecycle` slice (established / short_history / new).
- ✅ **Review fixes** (`tests/test_fixes.py`):
  - The checkpoint guard reads the manifest one level up, and fine-tuned weights without a manifest are refused. Backtest cache checkpoints carry a manifest.
  - Croston no longer treats stockout days as zero demand.
  - `min_history_days` is decided per origin.
  - Known covariates must have an explicit policy.
  - Duplicate plan rows raise an error (or keep the last one).
  - Ledgered runs record the resolved config and invocation, the git check runs before the backtest, and verdicts across different evaluation setups are `incomparable`.
  - Fine-tuning gets full series, with `context_length` passed to the trainer.
  - The MPS fallback is set at import.
  - Categoricals are stored as codes.
  - Contexts are padded to the cutoff.
  - Missing predictions are counted.
  - Summing duplicate rows keeps missing values missing.
  - The sampler takes `--until` and uses per-day volume.
  - The fine-tune cache key includes a code fingerprint.
- ✅ **CLI**: `env, models, config, audit, sample, cutoffs, run, leaderboard, finetune, forecast`.
- ✅ **CI**: ubuntu core job without torch, plus a macos-14 (arm64) job with torch and chronos.

## 3. Invariants

See `CLAUDE.md`. In short: pipeline only, no leakage, one hypothesis per run, `experiments/` append-only, tests green, NaN-not-0.

## 4. Work packages

### WP1: Onboard the real data ❓
- 🔲 Replace every `TODO` in `project.yaml` with the real columns.
- 🔲 Classify every covariate as known / past / static, and write the user's reasons in `experiments/DATA_NOTES.md`.
  - Observed weather goes in `past_covariate_cols`. Weather forecasts as issued may be known, fed via `plan`.
- 🔲 Run `audit` and review `reports/data_audit.md` with the user.

**Acceptance:** `audit` runs clean, and its covariate table matches reality.

### WP2: Stratified sample ✅
`python -m forecast_fm sample --src <full> --n 30000 --by demand_label --volume-bins 10 --seed 42`
- Streams the id, target and `--by` columns through pyarrow.
- Strata = label × mean-volume bin. Each stratum's quota is proportional, with a floor of `min(floor, size)`.
- Writes the full rows of the chosen series, plus `data/raw/sample_manifest.csv` (series, stratum, population weight).
- Tests: determinism, floor, schema, csv input.

**Remaining:** run it on the real 700k file, and record the wall time and peak RSS on the Mac.

### WP3: Fold cutoffs + plan-snapshot contract 🟡
- ✅ `python -m forecast_fm cutoffs` prints the validation and holdout dates. The user exports plan snapshots `as_of` those dates.
- ✅ Contract: `as_of, <timestamp_col>, <series_id_cols...>, <known cols>`, with rows for `as_of+1 .. as_of+horizon`, in csv or parquet.
- 🔲 **Acceptance:** a real backtest with `plan` policies prints coverage ≥ 99% for every covariate.

### WP4: Scale the data path to 700k–3M series on a Mac 🟡
The grid, classes, metrics and model input assembly are vectorized. But one 700k × 1460 panel is ~1B rows (~40+ GB with covariates) and does **not** fit in 32 GB of unified memory. Chronos-2 forecasts are per series, or per `group_by` group, so **sharding by series is exact**:
- ✅ `--shards N` on `run` and `forecast`, and `--shard i` on `forecast` for parallel processes. Series are assigned by a stable hash of the series id, or of the `shard_by` statics so a group stays together.
- ✅ Streamed ETL per shard: the data and plan files are read in batches and only the shard's rows are kept. One streaming pass finds the global date span and series set, so every shard shares the cutoffs and grid end, and a new SKU is told apart from one whose history lives in another shard.
- ✅ Shard outputs: backtest metrics are combined from additive sums per shard, so they are identical to an unsharded run (tested). Forecast writes one parquet part per shard (the long daily file); it is resumable and writes a manifest.
- ✅ Categoricals are integer codes in the grid and model inputs; strings are built per series.
- 🔲 Benchmark on the Mac: rows/s and peak RSS per shard, and total wall time. Choose N so a shard fits the memory budget.
- Note: cold-start launch profiles pool analogs within a shard. Set `shard_by` = `cold_start.profile_cols` to make them identical to an unsharded run.

**Acceptance:** the full panel runs end to end within a memory budget agreed with the user (e.g. peak RSS < 24 GB on 32 GB).

### WP5: Chronos-2 at production scale 🟡
- ✅ `dtype` param, streaming predict, fine-tune checkpoint cache, NaN for missing horizon values, coverage warning.
- 🔲 **First real load:** record `model_context_length` (expect ≥ 1460) and `model_prediction_length` (must be ≥ 90, or the model unrolls autoregressively) in `DATA_NOTES.md`. Both are printed on load.
- 🔲 **bf16 parity on MPS:** run the same zero-shot config on the sample in float32 and in bfloat16. Adopt bf16 only if the metrics match within noise.
- 🔲 **Throughput on the Mac:** series/s for zero-shot with all covariates, with and without `group_by`, across `batch_size` ∈ {128, 256, 512}. Use the numbers to size WP4's full run.
- 🔲 **Fine-tuning on MPS:** LoRA first (`peft`). Record the wall time per fold at `num_steps` 1000.

**Acceptance:**
- A 30k-series zero-shot backtest (3-4 folds + holdout) completes on the Mac.
- Throughput and memory are reported.
- Tests are green.

### WP6: Covariate enrichment (data side) ❓
Each tier is one experiment, run only after the zero-shot reference exists. Columns must be real data columns, because Chronos-2 reads them in context.

| Tier | Column | Class | Mechanism |
|---|---|---|---|
| 1 | `discount_pct` = 1 − price/regular_price | known (plan) | elasticity: depth, not level |
| 1 | `promo_day`, `days_to_promo_end` | known (plan) | promo decay shape, post-promo dip, pre-sale wait |
| 1 | `sitewide_event` (0..N) | known (actual) | shared spikes: Black Friday, paydays, holidays |
| 1 | `shipping_cutoff` flag | known (actual) | gifting demand drop at the last-delivery date |
| 1 | `in_stock` / stock on hand | past | censored demand → stockout masking |
| 2 | planned email/push/banner/ad spend | known (plan) | marketing-driven demand |
| 2 | sessions, PDP views, add-to-carts | past | leading indicators, short horizons |
| 3 | weather forecast as issued | known (plan) | weather-sensitive categories only |
| 3 | days since launch | known | lifecycle ramp |

Compute scales with (1 + n_covariates) variates per series. Measure accuracy against compute for each tier.

### WP7: Experiment plan (ledgered, one run per user go-ahead) 🔲

| # | Model / axis | Hypothesis | based_on |
|---|---|---|---|
| 1 | `seasonal_naive` | sets the bar | — |
| 2 | `gbt` (LightGBM, lags/calendar/known covariates/statics) | global regression beats the baseline, especially on promo/event | 1 |
| 3 | `chronos2` zero-shot, all covariates | in-context covariates match or beat gbt without feature engineering | 2 |
| 4 | `chronos2` + `group_by: [category]` | cross-learning helps short and intermittent series | 3 |
| 5 | `chronos2` + LoRA `fine_tune` | domain adaptation cuts promo/event error | best of 3/4 |
| 6 | `router`: intermittent → croston or chronos2, the rest → best | the intermittent class needs its own method | best |
| 7+ | WP6 tiers, one per run | per the tier's mechanism | current best |

- 🔲 **`gbt` model family** is not built yet (needed for #2). Build a direct multi-horizon LightGBM with origin-only lag features, and add a leakage test.
- 🔲 **`router` model family** (needed for #6).
- Configs for #1, #3, #4 and #5 are in `configs/`.
- Always check `h01-35` against `h36-90`, and the demand-class slices. Watch the intermittent class through its quantiles and coverage.

### WP8: Ship 🔲
- Success criteria agreed with the user (`success_criteria`). The holdout is evaluated once, only for the promoted recipe (a `promote` command is still to build).
- Production flow: backtest the recipe with `run` → `finetune` on all history → `forecast` with the saved checkpoint → retrain on a schedule (monthly, say, or when drift shows).
- 🔲 **Retrain cadence evidence:** backtest a checkpoint trained at cutoff T and forecasting at T + 30/60/90 days (a forward shadow evaluation the guard allows). This shows how fast accuracy decays with checkpoint age.
- 🔲 **Fine-tuning on 700k series:** train on the stratified sample, or on all series in shards. Compare them on the sample's backtest before choosing.
- Full 700k production forecast: the promoted config through sharded `forecast` (depends on WP4 and WP5).

## 5. Open questions for the user ❓
1. Real column names and series grain (`sku`? `sku × warehouse`?).
2. Are daily plan snapshots archived? If not: `carry_forward` for price, and record the limitation.
3. Is weather observed or forecast as issued?
4. Service-level target (e.g. 95%): it sets the decision quantile.
5. Success criteria (WAPE/bias overall, the `h01-35` bucket, per class): set after the baselines.
6. Mac model and memory (M-series chip, 32/64/128 GB?) and the macOS version (bf16 on MPS needs 14+).
7. Network access to huggingface.co, or a local copy of `amazon/chronos-2`.
8. Memory budget for the full 700k run (WP4).

## 6. Environment

See `README.md` → "Setup on a Mac". Quick check:

```bash
python -m forecast_fm env     # "mps": true, "auto_device": "mps"
pytest                        # green
```
