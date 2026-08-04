"""Config loading: YAML + `extends` merge + registry resolution.

An experiment config (configs/experiments/*.yaml) may set `extends: _base.yaml`; this
loader deep-merges base under it. It then resolves the string keys `model` and
`dictionary` against configs/models.yaml and configs/dictionaries.yaml, attaching the
resolved specs as `model_spec` / `dictionary_spec`. All hyperparameters therefore live in
version-controlled files, not CLI flags (../CLAUDE.md §5).
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

# Anchor for every project-relative path (data/, configs/, script hashing) so runs behave
# identically from any CWD — review finding: CWD-relative paths could write outside data/.
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f) or {}


_load_yaml = load_yaml  # backward-compat alias


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge `override` onto a copy of `base` (override wins on scalars/lists)."""
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_experiment(config_path: str | Path) -> dict[str, Any]:
    """Load an experiment config with `extends` applied and registries resolved."""
    config_path = Path(config_path)
    cfg = _load_yaml(config_path)

    # 1. apply `extends` (relative to the experiment file's dir)
    parent = config_path.parent
    if "extends" in cfg:
        base = _load_yaml(parent / cfg.pop("extends"))
        cfg = _deep_merge(base, cfg)

    # 2. resolve model / dictionary registry references (paths are relative to the config dir)
    def _registry(key: str, registry_file: str, section: str) -> None:
        name = cfg.get(key)
        if name is None:
            return
        reg = _load_yaml(parent / cfg[registry_file])
        entries = reg.get(section, {})
        if name not in entries:
            raise KeyError(f"{key}={name!r} not found in {cfg[registry_file]} [{section}]")
        cfg[f"{key}_spec"] = entries[name]

    _registry("model", "models_registry", "models")
    _registry("dictionary", "dictionaries_registry", "dictionaries")

    cfg["_config_path"] = str(config_path)
    return cfg


def resolve_layers(cfg: dict[str, Any], n_layers: int | None = None) -> list[int]:
    """Turn `layers: sweep | [ints]` into concrete **0-indexed transformer-block indices**.

    Convention (matches Llama Scope / Gemma Scope layer numbering): "layer l" means the
    residual stream AFTER 0-indexed block l (resid_post), i.e. hidden_states[l+1] in HF
    output (hidden_states[0] is the embeddings). Valid range: 0 .. n_layers-1.

    `sweep` = roughly {half, two-thirds} depth plus a mid point — a cheap first scan
    (the auditing-game note: reward features showed up nearer the midpoint than 2/3).
    Requires n_layers when `sweep` is used. Out-of-range explicit layers raise (silently
    dropping layers on a shallower model would corrupt panel-wide comparisons).
    """
    layers = cfg.get("layers", "sweep")
    if isinstance(layers, list):
        if n_layers is not None:
            bad = [l for l in layers if not (0 <= l <= n_layers - 1)]
            if bad:
                raise ValueError(f"layers {bad} out of range for a {n_layers}-block model "
                                 f"(valid: 0..{n_layers - 1}).")
        return layers
    if layers == "sweep":
        if n_layers is None:
            raise ValueError("layers='sweep' needs n_layers (read it from the loaded model).")
        picks = {n_layers // 2, (2 * n_layers) // 3, (n_layers // 2) + 2}
        return sorted(min(l, n_layers - 1) for l in picks)
    raise ValueError(f"Unrecognized layers spec: {layers!r}")
