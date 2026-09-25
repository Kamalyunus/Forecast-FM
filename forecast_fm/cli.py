"""Command line: `python -m forecast_fm <command>`."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _project(args):
    from .config import load_project

    return load_project(args.project, profile=args.profile, overrides=args.set)


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


def cmd_env(args):
    from .device import describe

    print(json.dumps(describe(), indent=2))


def cmd_models(args):
    from .models import REGISTRY

    for name, cls in REGISTRY.items():
        print(f"{name:16s} params: {', '.join(sorted(cls.PARAM_KEYS)) or '—'}")


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
    from .data import date_span
    from .folds import fold_cutoffs

    project = _project(args)
    first, last = date_span(project, args.data)
    for kind, hold in (("validation", False), ("holdout", True)):
        for c in fold_cutoffs(first, last, project, holdout=hold):
            print(f"{kind}\t{c.date()}")


def cmd_run(args):
    from .config import load_experiment
    from .ledger import record, require_clean
    from .runner import run_backtest

    project = _project(args)
    exp = load_experiment(args.config)
    if not args.no_commit:
        require_clean(args.project)  # before the backtest, not after it
    pred_dir = None
    if args.save_predictions:
        pred_dir = Path(project.reports_dir) / "predictions" / exp.name
    result, _ = run_backtest(project, exp, args.data, shards=args.shards, predictions_dir=pred_dir)
    context = {"project_file": args.project, "profile": args.profile, "overrides": args.set,
               "data": args.data, "shards": args.shards, "cutoffs": result["cutoffs"],
               "experiment_file": args.config}
    record(exp, project, result, commit=not args.no_commit, context=context)
    if pred_dir:
        print(f"[run] predictions -> {pred_dir}")
    for name, t in result["tables"].items():
        if name != "bucket_x_class":
            print(f"\n{name}\n{t.to_string(index=False, float_format=lambda v: f'{v:.4f}')}")


def cmd_leaderboard(args):
    from .ledger import leaderboard

    df = leaderboard(_project(args))
    print(df.to_string(index=False) if not df.empty else "no ledgered experiments yet")


def cmd_finetune(args):
    from .config import load_experiment
    from .data import load_panel
    from .finetune import finetune

    project = _project(args)
    exp = load_experiment(args.config)
    finetune(project, exp, load_panel(project, args.data), as_of=args.as_of, out=args.out,
             force=args.force, config_path=args.config)


def cmd_forecast(args):
    from .config import load_experiment
    from .runner import run_forecast

    project = _project(args)
    exp = load_experiment(args.config)
    only = None if args.shard is None else args.shard
    run_forecast(project, exp, args.data, shards=args.shards, only=only, out_dir=args.out,
                 force=args.force)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="forecast_fm", description=__doc__)
    ap.add_argument("-p", "--project", default="project.yaml")
    ap.add_argument("--profile", help="apply a named profile from project.yaml "
                                      "(default: $FORECAST_FM_PROFILE)")
    ap.add_argument("-s", "--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a project setting, e.g. --set horizon=35 "
                         "--set covariate_eval_policy.price=carry_forward (repeatable)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("env", help="torch / device / chronos versions").set_defaults(fn=cmd_env)
    sub.add_parser("models", help="registered model families").set_defaults(fn=cmd_models)

    k = sub.add_parser("config", help="print the resolved project config")
    k.add_argument("--check-data", action="store_true", help="also load the data and check columns")
    k.add_argument("--data")
    k.set_defaults(fn=cmd_config)

    a = sub.add_parser("audit", help="data audit -> reports/data_audit.md")
    a.add_argument("--data")
    a.add_argument("--out", help="default: <reports_dir>/data_audit.md")
    a.set_defaults(fn=cmd_audit)

    s = sub.add_parser("sample", help="stratified series sample (before ETL)")
    s.add_argument("--n", type=int, default=30000)
    s.add_argument("--by", default=None, help="stratum label column, e.g. demand_label")
    s.add_argument("--volume-bins", type=int, default=10)
    s.add_argument("--floor", type=int, default=200)
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--src", default=None, help="full dataset (default: project data_path)")
    s.add_argument("--until", default=None,
                   help="measure volume only up to this date (e.g. the first validation cutoff), "
                        "so strata never use the backtest period")
    s.add_argument("--out", default="data/raw/sales_sample.parquet")
    s.add_argument("--manifest", default="data/raw/sample_manifest.csv")
    s.set_defaults(fn=cmd_sample)

    c = sub.add_parser("cutoffs", help="fold cutoff dates (export plan snapshots as_of these)")
    c.add_argument("--data")
    c.set_defaults(fn=cmd_cutoffs)

    r = sub.add_parser("run", help="backtest an experiment config and ledger it")
    r.add_argument("config")
    r.add_argument("--data")
    r.add_argument("--no-commit", action="store_true", help="debug run: no ledger, no git commit")
    r.add_argument("--save-predictions", action="store_true",
                   help="write predictions to <reports_dir>/predictions/<name>/")
    r.add_argument("--shards", type=int, default=1,
                   help="process series in N shards (bounded memory; metrics are exact)")
    r.set_defaults(fn=cmd_run)

    sub.add_parser("leaderboard").set_defaults(fn=cmd_leaderboard)

    t = sub.add_parser("finetune", help="fine-tune Chronos-2 on all history and save a checkpoint")
    t.add_argument("config", help="chronos2 config with a fine_tune: block")
    t.add_argument("--data")
    t.add_argument("--as-of", help="last training date (default: last date in the data)")
    t.add_argument("--out", help="output dir (default: models/<name>-<as_of>)")
    t.add_argument("--force", action="store_true", help="replace an existing output dir")
    t.set_defaults(fn=cmd_finetune)

    f = sub.add_parser("forecast", help="production forecast from the last date: the long daily "
                                        "file for every series, new ones included")
    f.add_argument("config")
    f.add_argument("--data")
    f.add_argument("--out", help="output directory (default: <reports_dir>/forecast_<origin>/)")
    f.add_argument("--shards", type=int, default=1, help="split the series into N shards")
    f.add_argument("--shard", type=int, action="append",
                   help="run only this shard (repeatable; run shards as parallel processes)")
    f.add_argument("--force", action="store_true", help="redo shards whose part already exists")
    f.set_defaults(fn=cmd_forecast)

    args = ap.parse_args(argv)
    pd.set_option("display.width", 160)
    args.fn(args)
