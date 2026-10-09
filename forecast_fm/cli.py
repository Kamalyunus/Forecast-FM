"""Command line: `python -m forecast_fm <command>`.

The commands in the order a project uses them:

  env         check Python, torch and the device (expect "auto_device": "mps" on a Mac)
  config      show the resolved project.yaml; --check-data loads the data and checks columns
  audit       data audit report: counts, dates, zeros, stockouts, demand classes, cutoffs
  cutoffs     the retrain dates and forecast origins: export plan snapshots as_of these
  sample      a stratified series sample of a large catalog, for experiments
  run         backtest an experiment config (--no-commit to debug; ledgered otherwise)
  leaderboard every ledgered experiment, best first
  bench       time training and forecasting here; estimate the hours of the full plan
  finetune    fine-tune Chronos-2 on all history and save a checkpoint
  forecast    the production forecast: one row per series x day, new SKUs included

Try it on synthetic data first:

  python examples/make_demo_data.py
  python -m forecast_fm -p examples/demo_project.yaml run configs/01_seasonal_naive.yaml --no-commit

Set FORECAST_FM_PROJECT to avoid repeating -p, and FORECAST_FM_DEBUG=1 for full tracebacks.
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

import pandas as pd

from . import __version__

PROJECT_ENV = "FORECAST_FM_PROJECT"
DEBUG_ENV = "FORECAST_FM_DEBUG"
FOUNDATION_MODULES = ("torch", "chronos", "peft", "transformers", "accelerate")


def _project(args):
    from .config import load_project

    return load_project(args.project, profile=args.profile, overrides=args.set)


def _experiment(args, project):
    """Load the experiment and refuse bad params or a missing library now,
    before any data is loaded."""
    import importlib.util

    from .config import load_experiment
    from .models import create_model

    exp = load_experiment(args.config)
    create_model(exp.model, project.model_params(exp.model, exp.model_params))
    if exp.model == "chronos2" and importlib.util.find_spec("torch") is None:
        raise ModuleNotFoundError("torch is not installed", name="torch")
    return exp


def cmd_config(args):
    """Print the fully resolved project config (extends, sections, profile,
    --set and ${ENV} applied) and optionally check it against the data."""
    import yaml

    project = _project(args)
    print(yaml.safe_dump(project.to_dict(), sort_keys=False))
    if args.check_data:
        from .data import load_raw

        raw = load_raw(project, args.data)
        print(f"[config] OK: {len(raw):,} rows, {raw['series_id'].nunique():,} series, "
              f"{raw['ts'].min().date()} .. {raw['ts'].max().date()}; all declared columns present")
        print("[config] next: python -m forecast_fm audit")


def cmd_env(args):
    import json

    from .device import describe

    info = describe()
    print(json.dumps(info, indent=2))
    if info.get("torch") is None:
        print("[env] torch is not installed: baselines work; for chronos2 run  "
              "pip install -e \".[foundation,dev]\"")
    elif info.get("auto_device") == "cpu" and info.get("machine") == "x86_64" \
            and sys.platform == "darwin":
        print("[env] x86 Python on a Mac cannot see MPS: install an arm64 Python (see README)")


def cmd_models(args):
    import inspect

    from .models import REGISTRY

    if args.name:
        if args.name not in REGISTRY:
            raise ValueError(f"unknown model {args.name!r}; registered: {sorted(REGISTRY)}")
        print(f"{args.name}\n\n{inspect.getdoc(REGISTRY[args.name])}")
        return
    for name, cls in REGISTRY.items():
        print(f"{name:16s} params: {', '.join(sorted(cls.PARAM_KEYS)) or '—'}")
    print("\npython -m forecast_fm models <name> describes a family's params")


def cmd_audit(args):
    from .audit import audit
    from .data import build_panel, load_raw

    project = _project(args)
    raw = load_raw(project, args.data)
    panel = build_panel(raw, project)
    out = args.out or f"{project.reports_dir}/data_audit.md"
    print(audit(panel, len(raw), project, out))
    print(f"[audit] -> {out}")


def cmd_sample(args):
    from .sample import run_sample

    project = _project(args)
    run_sample(project, args.n, args.by, args.volume_bins, args.floor, args.seed, args.out,
               args.manifest, src=args.src, until=args.until)


def cmd_cutoffs(args):
    """Fold cutoffs (retrain dates) and every forecast origin: export plan
    snapshots as_of each origin."""
    from .data import date_span
    from .folds import fold_cutoffs, fold_origins, origin_limit

    project = _project(args)
    first, last = date_span(project, args.data)
    val = fold_cutoffs(first, last, project)
    for k, fold in enumerate(fold_origins(val, project, origin_limit(first, last, project))):
        for o in fold:
            print(f"validation\tfold {k}\tretrain {val[k].date()}\torigin {o.date()}")
    if not project.holdout_folds and not project.holdout_cutoffs:
        print("holdout\t\tnone (holdout_folds: 0)")
        return
    for c in fold_cutoffs(first, last, project, holdout=True):
        print(f"holdout\t\tretrain {c.date()}\torigin {c.date()}")


def cmd_run(args):
    from .ledger import check_reference, record, release, require_clean, reserve
    from .runner import run_backtest

    project = _project(args)
    exp = _experiment(args, project)
    # everything that can refuse the run does so here, before the backtest
    reserved = None
    if not args.no_commit:
        require_clean(args.project)
    check_reference(exp, commit=not args.no_commit)
    if not args.no_commit:
        reserved = reserve(exp)  # the id is taken now: parallel runs cannot collide
        print(f"[run] ledger entry {reserved.name} reserved")
    pred_dir = None
    if args.save_predictions:
        pred_dir = Path(project.reports_dir) / "predictions" / exp.name
    try:
        result, _ = run_backtest(project, exp, args.data, shards=args.shards, predictions_dir=pred_dir)
    except BaseException:
        if reserved is not None:
            release(reserved)
        raise
    context = {"project_file": args.project, "profile": args.profile, "overrides": args.set,
               "data": args.data, "shards": args.shards, "cutoffs": result["cutoffs"],
               "experiment_file": args.config}
    out = record(exp, project, result, commit=not args.no_commit, context=context, reserved=reserved)
    if pred_dir:
        print(f"[run] predictions -> {pred_dir}")
    for name, t in result["tables"].items():
        if name != "bucket_x_class":
            print(f"\n{name}\n{t.to_string(index=False, float_format=lambda v: f'{v:.4f}')}")
    print(f"\n[run] metrics and slice tables -> {out}/")


def cmd_leaderboard(args):
    from .ledger import leaderboard

    df = leaderboard(_project(args))
    print(df.to_string(index=False) if not df.empty else
          "no ledgered experiments yet (a `run` without --no-commit adds one)")


def cmd_finetune(args):
    from .finetune import finetune

    project = _project(args)
    exp = _experiment(args, project)
    # streams the catalog: only each round's training series are loaded
    finetune(project, exp, data=args.data, as_of=args.as_of, out=args.out,
             force=args.force, config_path=args.config)


def cmd_bench(args):
    from .bench import run

    project = _project(args)
    run(project, _experiment(args, project), args.out or f"{project.reports_dir}/bench.json",
        data=args.data, n_series=args.series, train_steps=args.steps, catalog=args.catalog,
        backtest_series=args.backtest_series, memory_gb=args.memory_gb)


def cmd_forecast(args):
    from .runner import run_forecast

    project = _project(args)
    exp = _experiment(args, project)
    only = None if args.shard is None else args.shard
    run_forecast(project, exp, args.data, shards=args.shards, only=only, out_dir=args.out,
                 force=args.force)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m forecast_fm", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version=f"forecast-fm {__version__}")
    ap.add_argument("-p", "--project", default=os.environ.get(PROJECT_ENV, "project.yaml"),
                    help=f"project file (default: ${PROJECT_ENV} or project.yaml)")
    ap.add_argument("--profile", help="apply a named profile from the project file "
                                      "(default: $FORECAST_FM_PROFILE)")
    ap.add_argument("-s", "--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a project setting, e.g. --set horizon=35 "
                         "--set covariate_eval_policy.price=carry_forward (repeatable)")
    sub = ap.add_subparsers(dest="cmd", required=True, metavar="<command>")
    data_help = "data file(s) to use instead of the project's data_path"

    sub.add_parser("env", help="Python / torch / device / chronos versions").set_defaults(fn=cmd_env)

    m = sub.add_parser("models", help="registered model families and their params")
    m.add_argument("name", nargs="?", help="describe this family's params")
    m.set_defaults(fn=cmd_models)

    k = sub.add_parser("config", help="print the resolved project config")
    k.add_argument("--check-data", action="store_true",
                   help="also load the data and check that every declared column is present")
    k.add_argument("--data", help=data_help)
    k.set_defaults(fn=cmd_config)

    a = sub.add_parser("audit", help="data audit -> reports/data_audit.md")
    a.add_argument("--data", help=data_help)
    a.add_argument("--out", help="report path (default: <reports_dir>/data_audit.md)")
    a.set_defaults(fn=cmd_audit)

    s = sub.add_parser("sample", help="stratified series sample of a large catalog, for experiments")
    s.add_argument("--n", type=int, default=30000, help="series to keep (default 30000)")
    s.add_argument("--by", default=None, help="stratum label column, e.g. demand_label")
    s.add_argument("--volume-bins", type=int, default=10,
                   help="volume quantile bins within each stratum (default 10)")
    s.add_argument("--floor", type=int, default=200,
                   help="minimum series per stratum, so rare classes are represented (default 200)")
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--src", default=None, help="full dataset (default: the project's data_path)")
    s.add_argument("--until", default=None,
                   help="measure volume only up to this date (e.g. the first validation cutoff), "
                        "so strata never use the backtest period")
    s.add_argument("--out", default="data/raw/sales_sample.parquet", help="sample file to write")
    s.add_argument("--manifest", default="data/raw/sample_manifest.csv",
                   help="per-stratum counts of the catalog and the sample")
    s.set_defaults(fn=cmd_sample)

    c = sub.add_parser("cutoffs", help="retrain dates and forecast origins (export plan "
                                       "snapshots as_of these)")
    c.add_argument("--data", help=data_help)
    c.set_defaults(fn=cmd_cutoffs)

    r = sub.add_parser("run", help="backtest an experiment config and ledger it")
    r.add_argument("config", help="experiment yaml, e.g. configs/01_seasonal_naive.yaml")
    r.add_argument("--data", help=data_help)
    r.add_argument("--no-commit", action="store_true",
                   help="debug run: results in <reports_dir>/scratch/, no ledger, no git commit")
    r.add_argument("--save-predictions", action="store_true",
                   help="write predictions to <reports_dir>/predictions/<name>/")
    r.add_argument("--shards", type=int, default=1,
                   help="process series in N shards (bounded memory; metrics are exact)")
    r.set_defaults(fn=cmd_run)

    sub.add_parser("leaderboard", help="ledgered experiments, best first").set_defaults(fn=cmd_leaderboard)

    t = sub.add_parser("finetune", help="fine-tune Chronos-2 on all history and save a checkpoint")
    t.add_argument("config", help="chronos2 config with a fine_tune: block")
    t.add_argument("--data", help=data_help)
    t.add_argument("--as-of", help="last training date (default: last date in the data)")
    t.add_argument("--out", help="output dir (default: <models_dir>/<name>-<as_of>)")
    t.add_argument("--force", action="store_true", help="replace an existing output dir")
    t.set_defaults(fn=cmd_finetune)

    b = sub.add_parser("bench", help="time fine-tuning and forecasting on this machine; "
                                     "estimate hours and shards for the full plan")
    b.add_argument("config", help="chronos2 config (with fine_tune: to time training)")
    b.add_argument("--data", help=data_help)
    b.add_argument("--series", type=int, default=2000, help="series to measure on (default 2000)")
    b.add_argument("--steps", type=int, default=50, help="training steps to time (default 50)")
    b.add_argument("--catalog", type=int, help="catalog size to extrapolate to (default: all in data)")
    b.add_argument("--backtest-series", type=int, help="series in the backtest (default: all in data)")
    b.add_argument("--memory-gb", type=float, default=24.0, help="memory budget for shard sizing")
    b.add_argument("--out", help="default: <reports_dir>/bench.json")
    b.set_defaults(fn=cmd_bench)

    f = sub.add_parser("forecast", help="production forecast from the last date: the long daily "
                                        "file for every series, new ones included")
    f.add_argument("config", help="experiment yaml, or a checkpoint's forecast_config.yaml")
    f.add_argument("--data", help=data_help)
    f.add_argument("--out", help="output directory (default: <reports_dir>/forecast_<origin>/)")
    f.add_argument("--shards", type=int, default=1, help="split the series into N shards")
    f.add_argument("--shard", type=int, action="append",
                   help="run only this shard (repeatable; run shards as parallel processes)")
    f.add_argument("--force", action="store_true", help="redo shards whose part already exists")
    f.set_defaults(fn=cmd_forecast)
    return ap


def main(argv=None):
    """Parse and run; errors propagate (tests and scripts want the exception)."""
    args = build_parser().parse_args(argv)
    pd.set_option("display.width", 160)
    args.fn(args)


def explain(e: BaseException, project_file: str) -> str:
    """One line a new user can act on, instead of a traceback."""
    lines = [f"error: {e}"]
    if isinstance(e, ModuleNotFoundError) and (e.name or "").split(".")[0] in FOUNDATION_MODULES:
        lines.append("hint: the chronos2 model needs the foundation extras:  "
                     "pip install -e \".[foundation,dev]\"  (baselines run without them)")
    elif isinstance(e, FileNotFoundError):
        lines.append(f"hint: paths are relative to the current directory; data_path and "
                     f"planned_covariates_path are set in {project_file} (docs/DATA_REQUIREMENTS.md "
                     f"says what the files must contain). To try the pipeline on synthetic data:  "
                     f"python examples/make_demo_data.py && python -m forecast_fm "
                     f"-p examples/demo_project.yaml audit")
    lines.append(f"({DEBUG_ENV}=1 shows the full traceback)")
    return "\n".join(lines)


def entry(argv=None) -> int:
    """`python -m forecast_fm`: user errors become one readable message."""
    argv = sys.argv[1:] if argv is None else list(argv)
    # library warnings (plan coverage, stale snapshots) as one line, not a traceback-like block
    warnings.formatwarning = lambda msg, cat, fn, ln, line=None: f"warning: {msg}\n"
    try:
        main(argv)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError, ModuleNotFoundError) as e:
        if os.environ.get(DEBUG_ENV):
            raise
        project_file = os.environ.get(PROJECT_ENV, "project.yaml")
        for flag in ("-p", "--project"):
            if flag in argv and argv.index(flag) + 1 < len(argv):
                project_file = argv[argv.index(flag) + 1]
        print(explain(e, project_file), file=sys.stderr)
        return 1
    return 0
