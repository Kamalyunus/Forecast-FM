"""The experiment ledger: `experiments/` is append-only.

Each ledgered run gets `experiments/expNNN/` (config, metrics, slice
tables, run stats) and one row in `experiments/LEDGER.md`, committed to git
as `expNNN [verdict] <hypothesis>`. Debug runs (`--no-commit`) write to
`reports/scratch/` and never touch the ledger.
"""

from __future__ import annotations

import json
import math
import re
import subprocess
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import yaml

from .config import ExperimentConfig, ProjectConfig
from .provenance import input_snapshot

EXP_DIR = Path("experiments")
LEDGER = EXP_DIR / "LEDGER.md"
HEADER = ("| id | model | hypothesis | based_on | primary | value | Δ vs ref | verdict |\n"
          "|---|---|---|---|---|---|---|---|\n")


def _git(*args: str, check: bool = False) -> str:
    r = subprocess.run(["git", *args], capture_output=True, text=True, check=False)
    if check and r.returncode != 0:
        raise RuntimeError(f"git {args[0]} failed: {(r.stderr or r.stdout).strip()}")
    return r.stdout.strip() if r.returncode == 0 else ""


def in_git_repo() -> bool:
    return _git("rev-parse", "--is-inside-work-tree") == "true"


def code_state(project_file: str | Path = "project.yaml") -> dict:
    """Commit, and whether the package or the project file actually used has
    uncommitted changes. Outside a git checkout: no commit, nothing dirty,
    `in_repo` False (a ledgered run refuses; a debug run proceeds)."""
    if not in_git_repo():
        return {"commit": None, "dirty_code": False, "in_repo": False}
    return {"commit": _git("rev-parse", "--short", "HEAD") or None, "in_repo": True,
            "dirty_code": bool(_git("status", "--porcelain", "--", "forecast_fm", str(project_file),
                                    check=True))}


def require_clean(project_file: str | Path) -> None:
    """Called BEFORE a ledgered run starts, not after hours of backtest."""
    state = code_state(project_file)
    if not state["in_repo"]:
        raise RuntimeError("not inside a git checkout: a ledgered run needs git to record the code "
                           "state (run from the repository root, or use --no-commit)")
    if state["dirty_code"]:
        raise RuntimeError(f"uncommitted changes in forecast_fm/ or {project_file}: commit them first "
                           "so the run is reproducible (or use --no-commit)")


# settings that do not change what is measured: excluded from the signature
# (warning thresholds and the class label only change tables and messages,
# never the primary metric)
_NOT_EVAL = {"name", "model_defaults", "reports_dir", "models_dir", "success_criteria",
             "verdict_threshold", "max_missing_share", "slice_cols", "primary_metric",
             "min_plan_coverage", "plan_max_age_days", "demand_label_col"}


def eval_signature(project: ProjectConfig, cutoffs: list, origins: list | None = None,
                   data=None, input_fingerprints: dict | None = None) -> str:
    """Two runs are comparable only if data, filters, covariate policy,
    horizon, quantiles, cutoffs and forecast origins are identical. `data`:
    a --data override of the project's data_path; `origins`: per-fold
    forecast origins (more data can add origins to the same cutoffs)."""
    import hashlib

    d = {k: v for k, v in project.to_dict().items() if k not in _NOT_EVAL}
    d["_input_fingerprints"] = (input_fingerprints if input_fingerprints is not None
                                else input_snapshot(project, data).fingerprints)
    d["_cutoffs"] = [str(pd.Timestamp(c).date()) for c in cutoffs]
    if origins:
        d["_origins"] = [[str(pd.Timestamp(o).date()) for o in fold] for fold in origins]
    if data is not None:
        d["_data"] = [str(x) for x in data] if isinstance(data, (list, tuple)) else str(data)
    return hashlib.sha256(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _slug(exp: ExperimentConfig) -> str:
    return re.sub(r"[^a-z0-9]+", "-", exp.name.lower()).strip("-")[:40]


def next_id() -> str:
    ids = [int(m.group(1)) for p in EXP_DIR.glob("exp*") if (m := re.match(r"exp(\d+)", p.name))]
    return f"exp{max(ids, default=0) + 1:03d}"


RUNNING = ".running"


def reserve(exp: ExperimentConfig) -> Path:
    """Claim the next id BEFORE the backtest, so two runs finishing in any
    order never share an id or overwrite each other's artifacts."""
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    out = EXP_DIR / f"{next_id()}-{_slug(exp)}"
    out.mkdir(exist_ok=False)
    (out / RUNNING).write_text(str(pd.Timestamp.now()))
    return out


def release(out: Path) -> None:
    """Undo a reservation whose run failed (only the marker is inside)."""
    out = Path(out)
    if out.is_dir() and {p.name for p in out.iterdir()} <= {RUNNING}:
        for p in out.iterdir():
            p.unlink()
        out.rmdir()


def _exp_dirs(exp_id: str) -> list[Path]:
    """exp001 matches exp001-<slug> and exp001, never exp0010-..."""
    return sorted(p for p in EXP_DIR.glob(f"{exp_id}*") if re.fullmatch(rf"{exp_id}(-.*)?", p.name))


def load_metrics(exp_id: str) -> dict:
    matches = [p / "metrics.json" for p in _exp_dirs(exp_id) if (p / "metrics.json").is_file()]
    if not matches:
        raise FileNotFoundError(f"no ledgered experiment {exp_id!r} under {EXP_DIR}/")
    return json.loads(matches[0].read_text())


def has_experiment(exp_id: str) -> bool:
    return any((p / "metrics.json").is_file() for p in _exp_dirs(exp_id))


def check_reference(exp: ExperimentConfig, commit: bool) -> None:
    """Called BEFORE a backtest: a ledgered run needs its `based_on` run in
    the ledger, or the verdict would fail after hours of compute. A debug run
    only warns and gets no verdict."""
    if not exp.based_on or has_experiment(exp.based_on):
        return
    msg = (f"based_on {exp.based_on!r} is not in {EXP_DIR}/ (python -m forecast_fm leaderboard lists "
           f"the ids): run the reference experiment first, or set based_on: null")
    if commit:
        raise ValueError(msg)
    print(f"[run] note: {msg}; this debug run gets no verdict")


def missing_share(metrics: dict) -> float:
    """Share of the scorable rows that got no forecast (1.0 when nothing was scored)."""
    overall = metrics["overall"]
    if "n" not in overall:
        return 0.0  # counts unknown (an older record): nothing to hold against it
    n, miss = overall.get("n") or 0, overall.get("n_missing_pred") or 0
    return miss / (n + miss) if n + miss else 1.0


def _trusted(metrics: dict, project: ProjectConfig, label: str) -> bool:
    """A run can take part in a verdict when it scored something, few rows lack a
    forecast (new SKUs without analogs, discontinued SKUs: visible, tolerated up
    to max_missing_share), and its inputs held still while it ran."""
    share = missing_share(metrics)
    if share > project.max_missing_share:
        print(f"[verdict] {label}: {share:.2%} of scorable rows have no forecast "
              f"(max_missing_share {project.max_missing_share:.0%}): inconclusive")
        return False
    if metrics.get("inputs_changed"):
        print(f"[verdict] {label}: input files changed during the run: inconclusive")
        return False
    return True


def verdict(metrics: dict, ref: dict | None, project: ProjectConfig) -> tuple[str, float | None]:
    """improved / regressed need the overall change beyond the threshold AND
    a majority of folds moving the same way; anything else is inconclusive.
    Runs on different data, cutoffs or origins (eval_signature) are
    incomparable; a run or reference with too many missing forecasts, or whose
    inputs changed while it ran, is inconclusive."""
    if not _trusted(metrics, project, "this run"):
        return "inconclusive", None
    if ref is None:
        return "reference", None
    if not _trusted(ref, project, "the reference"):
        return "inconclusive", None
    if ref.get("eval_signature") != metrics.get("eval_signature"):
        return "incomparable", None  # different data, policy, horizon, cutoffs or origins
    m = project.primary_metric
    new, old = metrics["overall"].get(m), ref["overall"].get(m)
    if new is None or not old or not math.isfinite(new) or not math.isfinite(old):
        return "inconclusive", None
    rel = (new - old) / abs(old)
    folds_new = {f["fold"]: f[m] for f in metrics["tables"]["fold"]}
    folds_old = {f["fold"]: f[m] for f in ref["tables"]["fold"]}
    # a fold without a finite value on either side says nothing: it is not "worse"
    common = [k for k in folds_new if k in folds_old
              and pd.notna(folds_new[k]) and pd.notna(folds_old[k])
              and math.isfinite(folds_new[k]) and math.isfinite(folds_old[k])]
    if not common:
        return "inconclusive", rel
    better = sum(folds_new[k] < folds_old[k] for k in common)
    if rel < -project.verdict_threshold and better > len(common) / 2:
        return "improved", rel
    if rel > project.verdict_threshold and better < len(common) / 2:
        return "regressed", rel
    return "inconclusive", rel


def record(exp: ExperimentConfig, project: ProjectConfig, result: dict, commit: bool,
           context: dict | None = None, reserved: Path | None = None) -> Path:
    """Write the run's artifacts; with commit=True, ledger it. `context`: how
    the run was invoked (project file, profile, --set, --data, shards,
    cutoffs); stored with the fully resolved project config. `reserved`: the
    directory `reserve` claimed before the run."""
    context = dict(context or {})
    project_file = context.get("project_file", "project.yaml")
    state = code_state(project_file)
    if commit:
        require_clean(project_file)
        out = Path(reserved) if reserved is not None else reserve(exp)
        exp_id = out.name.split("-")[0]
    else:
        exp_id = "scratch"
        out = Path(project.reports_dir) / "scratch" / re.sub(r"[^a-z0-9]+", "-", exp.name.lower())
    out.mkdir(parents=True, exist_ok=True)

    ref = load_metrics(exp.based_on) if exp.based_on and (commit or has_experiment(exp.based_on)) else None
    tables = {k: t.to_dict(orient="records") for k, t in result["tables"].items()}
    fingerprints = result.get("input_fingerprints")
    if fingerprints is None:
        fingerprints = input_snapshot(project, context.get("data")).fingerprints
    signature = eval_signature(project, context.get("cutoffs") or result.get("cutoffs") or [],
                               origins=result.get("origins"), data=context.get("data"),
                               input_fingerprints=fingerprints)
    v, rel = verdict({"overall": result["overall"], "tables": tables, "eval_signature": signature,
                      "inputs_changed": result.get("inputs_changed")}, ref, project)
    payload = {
        "id": exp_id, "verdict": v, "delta_rel": rel, "based_on": exp.based_on,
        "primary_metric": project.primary_metric, "code": state, "eval_signature": signature,
        "input_fingerprints": fingerprints, "inputs_changed": result.get("inputs_changed") or [],
        "missing_share": missing_share({"overall": result["overall"]}),
        "invocation": {k: v for k, v in context.items() if k != "cutoffs"},
        "cutoffs": [str(pd.Timestamp(c).date())
                    for c in (context.get("cutoffs") or result.get("cutoffs") or [])],
        "project": project.to_dict(),
        "overall": result["overall"],
        "tables": tables,
        "run_stats": result.get("stats", []), "env": result.get("env", {}),
    }
    (out / "config.yaml").write_text(yaml.safe_dump(asdict(exp), sort_keys=False))
    (out / "metrics.json").write_text(json.dumps(payload, indent=2, default=str))
    for k, t in result["tables"].items():
        t.to_csv(out / f"slices_{k}.csv", index=False)

    val = result["overall"].get(project.primary_metric)
    shown = "n/a" if val is None else f"{val:.4f}"
    print(f"[{exp_id}] {project.primary_metric}={shown} verdict={v}"
          + (f" ({rel:+.1%} vs {exp.based_on})" if rel is not None else ""))
    if commit:
        if not LEDGER.exists():
            LEDGER.write_text("# Experiment ledger\n\n" + HEADER)
        with LEDGER.open("a") as f:
            f.write(f"| {exp_id} | {exp.model} | {exp.hypothesis.replace('|', '/')} | "
                    f"{exp.based_on or '—'} | {project.primary_metric} | {shown} | "
                    f"{'—' if rel is None else f'{rel:+.1%}'} | {v} |\n")
        (out / RUNNING).unlink(missing_ok=True)
        _git("add", "--", str(out), str(LEDGER), check=True)
        # only these paths: never sweep whatever else sits in the index into the run's commit
        _git("commit", "-m", f"{exp_id} [{v}] {exp.hypothesis}", "--", str(out), str(LEDGER), check=True)
        print(f"[{exp_id}] committed {_git('rev-parse', '--short', 'HEAD')}")
    return out


def leaderboard(project: ProjectConfig) -> pd.DataFrame:
    rows = []
    for p in sorted(EXP_DIR.glob("exp*/metrics.json")):
        m = json.loads(p.read_text())
        rows.append({"id": m["id"], "verdict": m["verdict"], **m["overall"]})
    df = pd.DataFrame(rows)
    return df.sort_values(project.primary_metric) if not df.empty else df
