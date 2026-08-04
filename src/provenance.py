"""Run provenance — the record that makes a result trustworthy (../CLAUDE.md §4).

Captures, for every kept run: exact command, config path + resolved config, script
sha256(s), seed, GPU id(s) + timestamp, dictionary identity, and best-effort library
versions. Written as `run_manifest.json` alongside the run's outputs.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.configs import PROJECT_ROOT


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _lib_versions() -> dict[str, str]:
    """Best-effort versions of the libraries that affect results. Never raises."""
    versions: dict[str, str] = {"python": sys.version.split()[0], "platform": platform.platform()}
    for mod in ("torch", "transformers", "sae_lens", "transformer_lens", "numpy"):
        try:
            versions[mod] = __import__(mod).__version__
        except Exception:
            versions[mod] = "absent"
    return versions


@dataclass
class RunManifest:
    experiment: str
    run_id: str
    seed: int
    config_path: str
    config_resolved: dict[str, Any]
    model_hf_id: str | None = None
    model_revision: str | None = None            # requested (from configs/models.yaml)
    model_revision_resolved: str | None = None   # what HF actually loaded (config._commit_hash)
    dictionary: dict[str, Any] | None = None   # dictionary identity: repo + revision + type
    gpu_visible: str | None = field(default_factory=lambda: os.environ.get("CUDA_VISIBLE_DEVICES"))
    command: str = field(default_factory=lambda: " ".join(sys.argv))
    script_sha256: dict[str, str] = field(default_factory=dict)
    started_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    finished_utc: str | None = None
    libraries: dict[str, str] = field(default_factory=_lib_versions)
    extra: dict[str, Any] = field(default_factory=dict)

    def hash_scripts(self, paths: list[str | Path]) -> "RunManifest":
        """Hash scripts for provenance. Relative paths are anchored to PROJECT_ROOT so the
        hashes are recorded regardless of CWD; a missing script is an error, not a skip —
        an empty hash field would silently break §4 provenance."""
        for p in paths:
            p = Path(p)
            if not p.is_absolute():
                p = PROJECT_ROOT / p
            if not p.exists():
                raise FileNotFoundError(f"provenance: script to hash not found: {p}")
            self.script_sha256[str(p.relative_to(PROJECT_ROOT) if p.is_relative_to(PROJECT_ROOT) else p)] = sha256_file(p)
        return self

    def finalize(self) -> "RunManifest":
        self.finished_utc = datetime.now(timezone.utc).isoformat()
        return self

    def write(self, out_dir: str | Path) -> Path:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "run_manifest.json"
        path.write_text(json.dumps(asdict(self), indent=2, default=str))
        return path
