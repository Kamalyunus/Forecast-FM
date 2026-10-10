"""Content identities for local inputs, streamed once before a run.

Hashes are never cached by filename or mtime: replacing a file in place must
change its identity. Cheap stat checks reject inputs modified during a run.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import NamedTuple

from .config import ProjectConfig
from .data import _expand


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def _files(source) -> list[Path]:
    out = []
    for path in _expand(source):
        if path.is_dir():
            out.extend(sorted(p.resolve() for p in path.rglob("*") if p.is_file()))
        else:
            if not path.is_file():
                raise FileNotFoundError(f"input does not exist: {path}")
            out.append(path.resolve())
    if not out:
        raise ValueError(f"input contains no files: {source}")
    return out


class _Stamp(NamedTuple):
    """Inode, size and modification time: what changes when a file's bytes are
    replaced. Not ctime, which a chmod, chown or backup tool moves without
    touching the content."""

    ino: int
    size: int
    mtime_ns: int


def _stamp(path: Path) -> _Stamp:
    s = path.stat()
    return _Stamp(s.st_ino, s.st_size, s.st_mtime_ns)


class InputSnapshot:
    """Serializable fingerprints plus a guard against changing live inputs."""

    def __init__(self, sources: dict):
        self.sources = {k: v for k, v in sources.items() if v is not None}
        self.fingerprints: dict = {}
        self._stamps: dict = {}
        for name, source in self.sources.items():
            entries, stamps = [], []
            for path in _files(source):
                before = _stamp(path)
                h = hashlib.sha256()
                with path.open("rb") as f:
                    for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
                        h.update(chunk)
                if _stamp(path) != before:
                    raise ValueError(f"input changed while fingerprinting: {path}; retry with stable inputs")
                entries.append({"path": str(path), "size": before.size, "sha256": h.hexdigest()})
                stamps.append((str(path), before))
            self.fingerprints[name] = entries
            self._stamps[name] = stamps

    def changed(self) -> list[str]:
        """Names of the inputs whose files changed since the snapshot."""
        out = []
        for name, source in self.sources.items():
            try:
                current = [(str(p), _stamp(p)) for p in _files(source)]
            except (OSError, ValueError):
                out.append(name)
                continue
            if current != self._stamps[name]:
                out.append(name)
        return out

    def check(self) -> None:
        """Raise if any input changed: for a result about to be published."""
        changed = self.changed()
        if changed:
            raise ValueError(f"input changed during run: {', '.join(changed)}; retry with stable inputs")


def input_snapshot(project: ProjectConfig, data=None, *, production: bool = False,
                   model_params: dict | None = None) -> InputSnapshot:
    sources = {"sales": data if data is not None else project.data_path,
               "plans": project.planned_covariates_path}
    if production:
        sources["new_series"] = project.cold_start.get("new_series_path")
        model_id = (model_params or {}).get("model_id")
        if model_id and Path(str(model_id)).exists():
            sources["checkpoint"] = str(model_id)
            # The checkpoint guard also reads provenance one level above
            # a directly selected finetuned-ckpt directory.
            manifest = Path(str(model_id)).parent / "forecast_fm_model.json"
            if manifest.is_file():
                sources["checkpoint_manifest"] = str(manifest)
    return InputSnapshot(sources)


def code_digest() -> str:
    root = Path(__file__).parent
    return digest({str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in sorted(root.rglob("*.py"))})
