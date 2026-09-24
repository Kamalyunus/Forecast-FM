# CLAUDE.md: rules for working in this repo

You are helping a data scientist build a Chronos-2 replenishment forecaster.
`docs/SPEC.md` says **what to build and in what order**. This file says **how
work is done here**, and wins over the spec when they conflict.

## Invariants (never violate)

1. **All modeling goes through `python -m forecast_fm run <config>`.** If a
   capability is missing, extend the package (a model in `forecast_fm/models/`
   plus a registry entry). Never write one-off training scripts.
2. **Leakage:**
   - model context = data <= cutoff only (`backtest.forecast_at` slices it);
   - horizon covariates = declared known columns only, set by
     `covariate_eval_policy` (`plans.future_frame`);
   - plan snapshots with `as_of > cutoff` are never visible;
   - fine-tuning sees fold history only, and the checkpoint cache key includes
     the cutoff and a hash of that history;
   - a checkpoint saved by `finetune` is never evaluated at a cutoff before
     its manifest's `train_end` (`TrainedOnFutureError`); recipes are
     backtested with `fine_tune:` in the config, retrained per fold;
   - every new code path that touches data gets a test in the style of
     `test_backtest_blind_to_future_targets_and_past_covariates`.
3. **One hypothesis per experiment config**, with `based_on` set, so the
   verdict compares against the right reference. Run a ledgered experiment
   only after the user says go; use `--no-commit` for debugging.
4. **`experiments/` is append-only.** Never edit or delete a past run. Code
   commits are separate from experiment commits, and a ledgered run refuses to
   start while `forecast_fm/` or `project.yaml` has uncommitted changes.
5. **Keep `pytest` green.** Core tests must pass without torch (the stub
   pipeline tests). Real-library tests use a tiny random Chronos-2 and skip
   when chronos-forecasting is missing.
6. **Be suspicious of good news.** A big gain means checking per-fold
   uniformity and the plan-vs-actual covariate policy before celebrating.
7. **Missing covariate values are NaN, never 0.** A price of 0 is poison for
   a covariate-aware model.

## After every ledgered run, report

1. What was tested, the verdict, the primary-metric delta, and per-fold stability.
2. The 1-3 worst slices (horizon bucket x demand class) and the likely mechanism.
3. The recommended next experiment. Then wait for the user's go-ahead.

Always look at `h01-35` versus `h36-90`, and at the intermittent class
(check the quantiles and coverage, not just the median).

## Environment

The target machine is an Apple Silicon Mac: `device: auto` resolves to MPS
there. Any code touching torch must work on `cpu`, `mps` and `cuda`: no
float64 tensors on MPS, and no CUDA-only calls without a device check.

```bash
pytest                         # before and after every change
ruff check forecast_fm tests   # lint
python -m forecast_fm env      # device / versions
```

Commit messages: imperative mood for code; `expNNN [verdict] <hypothesis>`
for runs (written by the runner).
