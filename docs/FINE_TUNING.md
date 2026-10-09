# Fine-tuning Chronos-2 on your data

How `fine_tune:` works, what the knobs do, and how to run it on millions of
SKUs on a Mac. The short version is in the README; this page is the detail.

## The two places fine-tuning happens

| Command | Trains on | Purpose |
|---|---|---|
| `run <config with fine_tune:>` | each fold's history up to its cutoff, retrained per fold | **measure** the recipe: the verdict against `based_on` |
| `finetune <config>` | all history up to `--as-of` (or its last `train_window_days`) | **ship** the recipe: a saved checkpoint for `forecast` |

A checkpoint saved by `finetune` refuses to forecast from any date before its
`train_end`, so it can never be backtested on data it has already seen. Judge a
recipe with `run`, then `finetune` it once.

## The recipe

```yaml
model: chronos2
model_params:
  context_length: 730          # days the model reads per series, in training and at forecast time
  fine_tune:
    mode: lora                 # lora (adapters, cheap) | full (all weights)
    num_steps: 50000           # TOTAL steps, across rounds
    learning_rate: 1.0e-5
    batch_size: 64             # variates per step (target + covariates), ~8 SKUs at 8 variates
    train_window_days: 730     # learn from targets in the last 2 years only (fixed window)
    rounds: 10                 # draw a fresh set of SKUs 10 times (memory + variety)
    log_every: 100             # loss curve resolution in the manifest
    train_mix:                 # which SKUs may train
      max_series: 50000        # SKUs drawn (and held in memory) per round
      max_share: {intermittent: 0.25}
      min_nonzero_days: 4
      lookback_days: 365
      seed: 42
```

`configs/07_chronos2_lora_rounds.yaml` is this recipe. `configs/05` is the
minimal one (LoRA, 1000 steps, everything else default).

## What one training step is

The Chronos-2 trainer:

1. picks `batch_size` variates' worth of SKUs (about 8 SKUs at 8 variates),
   uniformly from the round's pool;
2. cuts each SKU's history at a random day: up to `context_length` days before
   it are the input (target, past covariates, known covariates), and the next 90
   days are the answer, with the known covariates' actual values for those days;
3. runs Chronos-2 on the batch (16-day patches, attention across time and across
   the SKU's variates) and predicts 21 quantiles for each of the 90 days;
4. scores them with quantile (pinball) loss against the real 90 days;
5. backpropagates, and AdamW updates the LoRA adapter (or all weights in `full`).

So 50,000 steps is about 400,000 SKU windows, and the time is steps × seconds
per step. **The cost is set by `num_steps` × `batch_size`, not by how many
SKUs you own.** More SKUs only change which windows the trainer can draw.

## Cost: measure before you commit hours

```bash
python -m forecast_fm bench configs/07_chronos2_lora_rounds.yaml --catalog 3000000 --memory-gb 24
```

`bench` times a few training steps and a zero-shot forecast on a random
subset of your data, then extrapolates: hours per fine-tune, hours for the
backtest (one fine-tune per fold, forecasts at every origin), hours for the
production forecast of the catalog, and the `--shards` that fit the memory
budget. Rerun it when the recipe or the covariate count changes. The numbers
are your machine's, not ours.

Levers, cheapest first: `mode: lora`, a smaller `context_length`, fewer
`num_steps`, a smaller `batch_size`.

## Choosing the training series (`train_mix`)

The trainer draws series uniformly, so the class mix of the training set is
the mix the model learns. With a catalog that is mostly intermittent, an
unconstrained draw teaches the model mostly zeros. `train_mix` limits that:

- `classes`: which demand classes may train (default: all).
- `max_share: {intermittent: 0.25}`: a cap on a class's share of the training
  series. The other classes fill the rest.
- `min_nonzero_days` over `lookback_days`: drop near-dead series.
- `max_series`: the total per round, keeping the shares.

Only training is affected: every series is still forecast and scored. The
selection uses data up to each fold's cutoff only, and the realized mix is
recorded in the run stats and the `finetune` manifest. `configs/06a-c` are
the continuous-only / natural-mix / capped-intermittent study.

## Many SKUs, bounded memory: rounds

`rounds: K` splits `num_steps` into K rounds. Each round draws
`train_mix.max_series` new SKUs, disjoint from earlier rounds and with the
same class mix, and loads only those. `finetune` profiles the whole catalog in
one streaming pass first (class, length, recent activity per SKU), so millions
of SKUs never sit in memory. LoRA adapters are merged after each round; the
learning rate steps down across rounds; the training loss per round is in the
manifest.

Without a per-round cap, a pool larger than `max_pool_series` (200k) is
refused rather than running out of memory.

## Fixed-window training (`train_window_days`)

By default a fine-tune learns from the whole history up to its cutoff, an
expanding window, so a later fold trains on more years than an earlier one and
none matches production. With `train_window_days: W`, every fine-tune learns
only from targets in the last W days before its cutoff, each with up to
`context_length` days of real history before it, as at inference. Every
fold's model and the production model then train on the same amount of recent
data. Young SKUs still train from their first possible day (padded at the
front with missing values, as Chronos-2 pads short histories when forecasting).

Choosing W is a trade-off: shorter tracks recent behaviour, longer sees more
yearly peaks. Keep W ≥ 1 year + the 90-day horizon so every season appears as
a target. Test 365, 730 and the full history as one experiment each.

## What `finetune` writes

```
models/<name>-<as_of>/
  finetuned-ckpt/          the weights (a merged model: loads without the base)
  forecast_fm_model.json   provenance: base model, recipe, train_end, train window,
                           data fingerprint, realized train mix, loss curve, code commit, env
  forecast_config.yaml     ready for:  python -m forecast_fm forecast models/<...>/forecast_config.yaml
```

Existing checkpoints are never overwritten unless you pass `--force`. Retrain
on your own schedule with a new `--as-of`; each `forecast` reports the
checkpoint's age in days.

## Is the fine-tune good? Judge it like any experiment

There is no inner validation set or early stopping inside a fine-tune. The
backtest folds are the judge:

- compare against the zero-shot run (`based_on`) on the primary metric, per fold;
- look at `h01-35` vs `h36-90`, and at the intermittent class's quantiles and
  coverage, not just the median;
- with `origin_step_days` set, read the `model_age` table: accuracy against
  weeks since retraining tells you how often to retrain.
