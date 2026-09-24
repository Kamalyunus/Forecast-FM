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
               args.manifest, src=args.src)


def cmd_cutoffs(args):
    from .data import TS, load_raw
    from .folds import fold_cutoffs

    project = _project(args)
    ts = load_raw(project, args.data)[TS]
    first, last = ts.min(), ts.max()
    for kind, hold in (("validation", False), ("holdout", True)):
        for c in fold_cutoffs(first, last, project, holdout=hold):
            print(f"{kind}\t{c.date()}")


def _run_backtest(project, exp, data=None):
    from .backtest import backtest
    from .data import SERIES, TS, demand_classes, load_panel
    from .device import describe
    from .folds import fold_cutoffs
    from .metrics import score
    from .plans import load_plans

    panel = load_panel(project, data)
    plans = load_plans(project)
    preds, stats = backtest(panel, project, exp, plans)
    first_cutoff = fold_cutoffs(panel[TS].min(), panel[TS].max(), project)[0]
    classes = demand_classes(panel, project, end=first_cutoff)
    statics = None
    if project.slice_cols:
        statics = panel.drop_duplicates(SERIES).set_index(SERIES)[project.slice_cols]
    result = score(preds, project, classes, statics)
    result.update(stats=stats, env=describe())
    return preds, result


def cmd_run(args):
    from .config import load_experiment
    from .ledger import record

    project = _project(args)
    exp = load_experiment(args.config)
    preds, result = _run_backtest(project, exp, args.data)
    out = record(exp, project, result, commit=not args.no_commit)
    if args.save_predictions:
        preds.to_parquet(out / "predictions.parquet", index=False)
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
    from .backtest import forecast_at
    from .config import load_experiment
    from .data import TS, load_panel
    from .plans import load_plans

    project = _project(args)
    exp = load_experiment(args.config)
    panel = load_panel(project, args.data)
    cutoff = panel[TS].max()
    out, stats = forecast_at(panel, cutoff, project, exp, load_plans(project), production=True)
    out.insert(0, "cutoff", cutoff)
    dest = args.out or f"{project.reports_dir}/forecast_{cutoff:%Y%m%d}.parquet"
    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(dest, index=False)
    print(f"[forecast] origin {cutoff.date()}: {out['series_id'].nunique():,} series -> {dest}")
    if stats:
        print(json.dumps(stats, default=str))


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
    r.add_argument("--save-predictions", action="store_true")
    r.set_defaults(fn=cmd_run)

    sub.add_parser("leaderboard").set_defaults(fn=cmd_leaderboard)

    t = sub.add_parser("finetune", help="fine-tune Chronos-2 on all history and save a checkpoint")
    t.add_argument("config", help="chronos2 config with a fine_tune: block")
    t.add_argument("--data")
    t.add_argument("--as-of", help="last training date (default: last date in the data)")
    t.add_argument("--out", help="output dir (default: models/<name>-<as_of>)")
    t.add_argument("--force", action="store_true", help="replace an existing output dir")
    t.set_defaults(fn=cmd_finetune)

    f = sub.add_parser("forecast", help="production forecast from the last date")
    f.add_argument("config")
    f.add_argument("--data")
    f.add_argument("--out", help="default: <reports_dir>/forecast_<origin>.parquet")
    f.set_defaults(fn=cmd_forecast)

    args = ap.parse_args(argv)
    pd.set_option("display.width", 160)
    args.fn(args)
