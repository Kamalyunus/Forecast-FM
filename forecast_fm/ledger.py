"""The experiment ledger: `experiments/` is append-only.

Each ledgered run gets `experiments/expNNN/` (config, metrics, slice
tables, run stats) and one row in `experiments/LEDGER.md`, committed to git
as `expNNN [verdict] <hypothesis>`. Debug runs (`--no-commit`) write to
`reports/scratch/` and never touch the ledger.
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import yaml

from .config import ExperimentConfig, ProjectConfig

EXP_DIR = Path("experiments")
LEDGER = EXP_DIR / "LEDGER.md"
HEADER = ("| id | model | hypothesis | based_on | primary | value | Δ vs ref | verdict |\n"
          "|---|---|---|---|---|---|---|---|\n")


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=False).stdout.strip()


def code_state(project_file: str | Path = "project.yaml") -> dict:
    """Commit, and whether the package or the project file actually used has
    uncommitted changes."""
    return {"commit": _git("rev-parse", "--short", "HEAD") or None,
            "dirty_code": bool(_git("status", "--porcelain", "--", "forecast_fm", str(project_file)))}


def require_clean(project_file: str | Path) -> None:
    """Called BEFORE a ledgered run starts, not after hours of backtest."""
    if code_state(project_file)["dirty_code"]:
        raise RuntimeError(f"uncommitted changes in forecast_fm/ or {project_file}: commit them first "
                           "so the run is reproducible (or use --no-commit)")


# settings that do not change what is measured: excluded from the signature
_NOT_EVAL = {"name", "model_defaults", "reports_dir", "models_dir", "success_criteria",
             "verdict_threshold", "slice_cols", "primary_metric"}


def eval_signature(project: ProjectConfig, cutoffs: list) -> str:
    """Two runs are comparable only if data, filters, covariate policy,
    horizon, quantiles and cutoffs are identical."""
    import hashlib

    d = {k: v for k, v in project.to_dict().items() if k not in _NOT_EVAL}
    d["_cutoffs"] = [str(pd.Timestamp(c).date()) for c in cutoffs]
    return hashlib.sha256(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:16]


def next_id() -> str:
    ids = [int(m.group(1)) for p in EXP_DIR.glob("exp*") if (m := re.match(r"exp(\d+)", p.name))]
    return f"exp{max(ids, default=0) + 1:03d}"


def load_metrics(exp_id: str) -> dict:
    matches = sorted(EXP_DIR.glob(f"{exp_id}*/metrics.json"))
    if not matches:
        raise FileNotFoundError(f"no ledgered experiment {exp_id!r} under {EXP_DIR}/")
    return json.loads(matches[0].read_text())


def verdict(metrics: dict, ref: dict | None, project: ProjectConfig) -> tuple[str, float | None]:
    """improved / regressed need the overall change beyond the threshold AND
    a majority of folds moving the same way; anything else is inconclusive."""
    if ref is None:
        return "reference", None
    if ref.get("eval_signature") and metrics.get("eval_signature") \
            and ref["eval_signature"] != metrics["eval_signature"]:
        return "incomparable", None  # different data, policy, horizon or cutoffs
    m = project.primary_metric
    new, old = metrics["overall"].get(m), ref["overall"].get(m)
    if new is None or not old:
        return "inconclusive", None
    rel = (new - old) / abs(old)
    folds_new = {f["fold"]: f[m] for f in metrics["tables"]["fold"]}
    folds_old = {f["fold"]: f[m] for f in ref["tables"]["fold"]}
    common = [k for k in folds_new if k in folds_old]
    better = sum(folds_new[k] < folds_old[k] for k in common)
    if rel < -project.verdict_threshold and better > len(common) / 2:
        return "improved", rel
    if rel > project.verdict_threshold and better < len(common) / 2:
        return "regressed", rel
    return "inconclusive", rel


def record(exp: ExperimentConfig, project: ProjectConfig, result: dict, commit: bool,
           context: dict | None = None) -> Path:
    """Write the run's artifacts; with commit=True, ledger it. `context`: how
    the run was invoked (project file, profile, --set, --data, shards,
    cutoffs); stored with the fully resolved project config."""
    context = dict(context or {})
    project_file = context.get("project_file", "project.yaml")
    state = code_state(project_file)
    if commit:
        if state["dirty_code"]:
            raise RuntimeError(f"uncommitted changes in forecast_fm/ or {project_file}: commit code "
                               "first so the run is reproducible (or use --no-commit)")
        exp_id = next_id()
        slug = re.sub(r"[^a-z0-9]+", "-", exp.name.lower()).strip("-")[:40]
        out = EXP_DIR / f"{exp_id}-{slug}"
    else:
        exp_id = "scratch"
        out = Path(project.reports_dir) / "scratch" / re.sub(r"[^a-z0-9]+", "-", exp.name.lower())
    out.mkdir(parents=True, exist_ok=True)

    ref = load_metrics(exp.based_on) if exp.based_on else None
    tables = {k: t.to_dict(orient="records") for k, t in result["tables"].items()}
    signature = eval_signature(project, context.get("cutoffs") or result.get("cutoffs") or [])
    v, rel = verdict({"overall": result["overall"], "tables": tables, "eval_signature": signature},
                     ref, project)
    payload = {
        "id": exp_id, "verdict": v, "delta_rel": rel, "based_on": exp.based_on,
        "primary_metric": project.primary_metric, "code": state, "eval_signature": signature,
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
        _git("add", str(out), str(LEDGER))
        _git("commit", "-m", f"{exp_id} [{v}] {exp.hypothesis}")
    return out


def leaderboard(project: ProjectConfig) -> pd.DataFrame:
    rows = []
    for p in sorted(EXP_DIR.glob("exp*/metrics.json")):
        m = json.loads(p.read_text())
        rows.append({"id": m["id"], "verdict": m["verdict"], **m["overall"]})
    df = pd.DataFrame(rows)
    return df.sort_values(project.primary_metric) if not df.empty else df
