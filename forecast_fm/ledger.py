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


def code_state() -> dict:
    return {"commit": _git("rev-parse", "--short", "HEAD") or None,
            "dirty_code": bool(_git("status", "--porcelain", "--", "forecast_fm", "project.yaml"))}


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


def record(exp: ExperimentConfig, project: ProjectConfig, result: dict, commit: bool) -> Path:
    """Write the run's artifacts; with commit=True, ledger it."""
    if commit:
        state = code_state()
        if state["dirty_code"]:
            raise RuntimeError("uncommitted changes in forecast_fm/ or project.yaml: commit code "
                               "first so the run is reproducible (or use --no-commit)")
        exp_id = next_id()
        slug = re.sub(r"[^a-z0-9]+", "-", exp.name.lower()).strip("-")[:40]
        out = EXP_DIR / f"{exp_id}-{slug}"
    else:
        state = code_state()
        exp_id = "scratch"
        out = Path("reports/scratch") / re.sub(r"[^a-z0-9]+", "-", exp.name.lower())
    out.mkdir(parents=True, exist_ok=True)

    ref = load_metrics(exp.based_on) if exp.based_on else None
    tables = {k: t.to_dict(orient="records") for k, t in result["tables"].items()}
    v, rel = verdict({"overall": result["overall"], "tables": tables}, ref, project)
    payload = {
        "id": exp_id, "verdict": v, "delta_rel": rel, "based_on": exp.based_on,
        "primary_metric": project.primary_metric, "code": state,
        "overall": result["overall"],
        "tables": tables,
        "run_stats": result.get("stats", []), "env": result.get("env", {}),
    }
    (out / "config.yaml").write_text(yaml.safe_dump(asdict(exp), sort_keys=False))
    (out / "metrics.json").write_text(json.dumps(payload, indent=2, default=str))
    for k, t in result["tables"].items():
        t.to_csv(out / f"slices_{k}.csv", index=False)

    val = result["overall"].get(project.primary_metric)
    print(f"[{exp_id}] {project.primary_metric}={val:.4f} verdict={v}"
          + (f" ({rel:+.1%} vs {exp.based_on})" if rel is not None else ""))
    if commit:
        if not LEDGER.exists():
            LEDGER.write_text("# Experiment ledger\n\n" + HEADER)
        with LEDGER.open("a") as f:
            f.write(f"| {exp_id} | {exp.model} | {exp.hypothesis.replace('|', '/')} | "
                    f"{exp.based_on or '—'} | {project.primary_metric} | {val:.4f} | "
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
