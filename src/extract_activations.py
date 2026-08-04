"""Activation-extraction harness (Phase-0, post-review).

Collects residual-stream activations for obfuscated-code snippets and caches them under
data/activations/<run_id>/ with a manifest + provenance. This is the shared front-end for
E1/E2/E6/E7 (feature analysis consumes the cache in a later stage).

Design constraints (../CLAUDE.md):
  * §1  torch is imported LAZILY, after CUDA_VISIBLE_DEVICES is pinned.
  * §4  seed set + recorded; smoke path validates the plumbing on a tiny model first;
        NO silent fallbacks — misalignment, collisions, and save failures raise.
  * §2  all outputs anchored to PROJECT_ROOT/data, regardless of CWD.

Layer convention (matches Llama Scope / Gemma Scope): "layer l" = resid_post of 0-indexed
block l = hidden_states[l+1]; saved under key `resid_post_L{l}`. See configs.resolve_layers.

Position schema: character spans (see src/data.py), resolved to token indices per model at
runtime via fast-tokenizer offset mappings; run fails if the resolution rate drops below
`positions.min_resolution_rate` (default 0.95).

Run:
  # cheap smoke on a tiny model (CPU): exercises the identifier-span path end to end
  python -m src.extract_activations --config configs/experiments/e1_semantic_capture.yaml --smoke

  # real run — pick a free GPU and pin it first:
  python -m src.extract_activations --config configs/experiments/e1_semantic_capture.yaml --gpu auto

NOTE: feature extraction against a *pretrained SAE* is stubbed (`apply_dictionary`) until
the exact repo ids in configs/dictionaries.yaml are confirmed. Raw residual-stream capture
is complete and exercised by --smoke.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

# Torch-free imports only at module scope (so GPU can be pinned before torch loads).
from src import configs as configs_mod
from src import data as data_mod
from src import gpu as gpu_mod
from src import provenance as prov_mod
from src import seedutil
from src.configs import PROJECT_ROOT

SCRIPTS_FOR_PROVENANCE = [
    "src/extract_activations.py",
    "src/configs.py",
    "src/data.py",
]

DTYPES = {"bfloat16", "float16", "float32"}


def resolve_spans_to_positions(
    spans: list[list[int]],
    offsets,                      # tokenizer offset_mapping for the (batch-1) sequence: [(s,e), ...]
    tok,
    input_ids,
    code: str,
) -> tuple[list[int], int, int]:
    """Map character spans -> token indices via offset overlap.

    Returns (positions, n_spans_resolved, n_spans_total). A span is resolved iff ≥1 token
    overlaps it AND the concatenated decode of those tokens contains the span's text — the
    decode check catches BOS shifts / wrong-offset bugs that a range filter never would.
    Spans lost to truncation count as unresolved.
    """
    positions: list[int] = []
    resolved = 0
    for a, b in spans:
        hit = [
            i for i, (s, e) in enumerate(offsets)
            if e > s and s < b and e > a          # (0,0) special tokens excluded by e > s
        ]
        if not hit:
            continue
        text = tok.decode([int(input_ids[i]) for i in hit])
        if code[a:b] in text:
            positions.extend(hit)
            resolved += 1
    return sorted(set(positions)), resolved, len(spans)


def apply_dictionary(hidden, dictionary_spec: dict, layer: int, device: str = "cpu"):
    """Encode residual-stream activations into SAE/transcoder features.

    Thin wrapper over src.dictionaries.load_dictionary — the feature-analysis stage should
    load once via that module and reuse; this exists for one-off encodes.
    """
    from src.dictionaries import load_dictionary

    return load_dictionary(dictionary_spec, layer=layer, device=device).encode(hidden)


def _save_safetensors(out_dir: Path, key: str, tensors: dict) -> str:
    """Persist {name: tensor} as safetensors. No silent fallback — a broken safetensors
    install should fail the run, not quietly change the output format."""
    from safetensors.torch import save_file

    save_file({k: v.contiguous().cpu() for k, v in tensors.items()}, str(out_dir / f"{key}.safetensors"))
    return "safetensors"


def run(cfg: dict, *, smoke: bool, limit: int | None, device_pref: str) -> Path:
    # ---- resolve run parameters (smoke overrides) -------------------------------
    experiment = cfg.get("experiment", "extract")
    seed = seedutil.set_seed(int(cfg.get("seed", 20260724)))
    min_rate = float(cfg.get("positions", {}).get("min_resolution_rate", 0.95))

    if smoke:
        model_hf_id = cfg["smoke"]["model_hf_id"]
        model_key = "smokemodel"
        model_revision = None
        token_positions = cfg["smoke"].get("token_positions", "identifiers")
        want_layers = cfg["smoke"]["layers"]
        snippets = data_mod.toy_stimuli(cfg["smoke"].get("n_items", 4))
    else:
        spec = cfg.get("model_spec") or {}
        model_hf_id = spec.get("hf_id")
        model_key = cfg.get("model", "model")
        model_revision = spec.get("revision")
        token_positions = cfg.get("token_positions", "last")
        want_layers = cfg.get("layers", "sweep")
        parent = Path(cfg["_config_path"]).resolve().parent
        cfg["data_config_resolved"] = configs_mod.load_yaml(parent / cfg["data_config"])
        snippets = data_mod.load_dataset(cfg, cfg["dataset"], cfg.get("tiers"))

    if limit:
        snippets = snippets[:limit]
    if not model_hf_id:
        raise ValueError("No model hf_id resolved — check the config's `model` key / models.yaml.")

    # In span modes, refuse snippets with no spans at all — that is a data error, and a
    # warn-and-fallback here silently corrupted the measurement (review finding).
    span_field = {"identifiers": "identifier_spans", "dispatcher": "dispatcher_spans"}.get(token_positions)
    if span_field:
        missing = [f"{s.snippet_id}/{s.tier}" for s in snippets if not getattr(s, span_field)]
        if missing:
            raise ValueError(f"token_positions={token_positions!r} but {len(missing)} snippet(s) have "
                             f"no {span_field}: {missing[:5]}{'...' if len(missing) > 5 else ''}")

    # ---- run dir under PROJECT_ROOT/data — collision-proof ----------------------
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    tag = "smoke" if smoke else "run"
    run_id = f"{stamp}_{experiment}_{model_key}_{tag}_pid{os.getpid()}"
    out_dir = PROJECT_ROOT / cfg.get("output", {}).get("activations_dir", "data/activations") / run_id
    out_dir.mkdir(parents=True, exist_ok=False)   # a collision must raise, never merge runs

    manifest = prov_mod.RunManifest(
        experiment=experiment, run_id=run_id, seed=seed,
        config_path=cfg.get("_config_path", "?"), config_resolved=cfg,
        model_hf_id=model_hf_id, model_revision=model_revision,
        dictionary=(None if smoke else cfg.get("dictionary_spec")),
    ).hash_scripts(SCRIPTS_FOR_PROVENANCE)
    manifest.write(out_dir)   # provisional manifest NOW — a mid-run crash still leaves provenance

    # ---- lazy torch + model load (GPU already pinned by caller) ------------------
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    use_cuda = (device_pref != "cpu") and torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    dtype_name = str(cfg.get("dtype", "bfloat16"))
    if dtype_name not in DTYPES:
        raise ValueError(f"dtype {dtype_name!r} not in {DTYPES}")
    dtype = getattr(torch, dtype_name) if use_cuda else torch.float32  # CPU: fp32, recorded below

    print(f"[{run_id}] model={model_hf_id} device={device} dtype={dtype} "
          f"seed={seed} n_snippets={len(snippets)} positions={token_positions}")

    tok = AutoTokenizer.from_pretrained(model_hf_id, revision=model_revision)
    if span_field and not tok.is_fast:
        raise RuntimeError(f"{model_hf_id}: span resolution needs a fast tokenizer (offset mappings).")
    model = AutoModelForCausalLM.from_pretrained(
        model_hf_id, revision=model_revision, dtype=dtype
    ).to(device).eval()
    manifest.model_revision_resolved = getattr(model.config, "_commit_hash", None)
    manifest.extra["dtype_actual"] = str(dtype)

    n_layers = model.config.num_hidden_layers
    # 0-indexed block l -> resid_post = hidden_states[l+1] (hidden_states[0] = embeddings).
    layers = configs_mod.resolve_layers({"layers": want_layers}, n_layers=n_layers)

    # ---- extraction loop: incremental manifest, duplicate-key guard --------------
    max_len = cfg.get("batching", {}).get("max_seq_len", 2048)
    seen_keys: set[str] = set()
    n_rows = 0
    rate_sum = 0.0
    manifest_path = out_dir / "manifest.jsonl"
    with open(manifest_path, "a") as mf:
        for s in snippets:
            key = f"{s.snippet_id}__{s.language}__{s.tier}"
            if key in seen_keys:
                raise ValueError(f"duplicate output key {key!r} — would overwrite a prior item's tensors.")
            seen_keys.add(key)
            # snippet ids contain '/' (e.g. "JavaScript/63") — never let a key become a path
            fs_key = key.replace("/", "--")

            enc = tok(s.code, return_tensors="pt", truncation=True, max_length=max_len,
                      return_offsets_mapping=bool(span_field))
            offsets = enc.pop("offset_mapping")[0].tolist() if span_field else None
            enc = {k: v.to(device) for k, v in enc.items()}
            seq_len = enc["input_ids"].shape[1]

            if span_field:
                positions, n_res, n_tot = resolve_spans_to_positions(
                    getattr(s, span_field), offsets, tok, enc["input_ids"][0], s.code)
                rate = n_res / n_tot if n_tot else 0.0
                if not positions:
                    raise ValueError(f"{key}: 0/{n_tot} spans resolved to tokens — alignment is broken "
                                     f"for this tokenizer; refusing to continue silently.")
            elif token_positions == "last":
                positions, rate, n_res, n_tot = [seq_len - 1], 1.0, 1, 1
            elif token_positions in ("code", "all"):
                positions, rate, n_res, n_tot = list(range(seq_len)), 1.0, 1, 1
            else:
                raise ValueError(f"Unknown token_positions mode: {token_positions!r}")
            rate_sum += rate

            with torch.no_grad():
                hs = model(**enc, output_hidden_states=True).hidden_states  # len n_layers+1
            tensors = {f"resid_post_L{l}": hs[l + 1][0, positions, :].to(torch.float32) for l in layers}
            fmt = _save_safetensors(out_dir, fs_key, tensors)

            row = {"key": key, "file": f"{fs_key}.safetensors", "snippet_id": s.snippet_id, "tier": s.tier,
                   "language": s.language, "seq_len": seq_len, "positions": positions,
                   "spans_resolved": n_res, "spans_total": n_tot, "resolution_rate": round(rate, 4),
                   "layers": layers, "hidden": int(hs[0].shape[-1]), "format": fmt}
            mf.write(json.dumps(row) + "\n")
            mf.flush()
            n_rows += 1

    # ---- resolution-rate gate (span modes) ---------------------------------------
    mean_rate = rate_sum / n_rows if n_rows else 0.0
    if span_field and mean_rate < min_rate:
        raise ValueError(f"mean span-resolution rate {mean_rate:.3f} < threshold {min_rate} — "
                         f"activations at {out_dir} are suspect; fix the alignment before analysis.")

    # ---- finalize provenance ------------------------------------------------------
    manifest.extra.update({"n_items": n_rows, "mean_resolution_rate": round(mean_rate, 4),
                           "layer_convention": "resid_post_L{l} = hidden_states[l+1], l 0-indexed"})
    manifest.finalize().write(out_dir)
    print(f"[{run_id}] wrote {n_rows} items (mean span-resolution {mean_rate:.3f}) -> {out_dir}")
    return out_dir


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract residual-stream activations for obfuscated-code snippets.")
    ap.add_argument("--config", required=True, help="Path to a configs/experiments/*.yaml file.")
    ap.add_argument("--smoke", action="store_true", help="Tiny-model + toy-stimuli plumbing check (CPU by default).")
    ap.add_argument("--gpu", default=None, help="'auto' to pick a free GPU, an integer index, or 'cpu'.")
    ap.add_argument("--limit", type=int, default=None, help="Cap number of snippets (quick runs).")
    args = ap.parse_args()

    cfg = configs_mod.load_experiment(args.config)

    # Pin the GPU BEFORE torch is imported anywhere (../CLAUDE.md §1). Idle thresholds come
    # from configs/compute.yaml so the config knob actually governs selection.
    device_pref = "cpu"
    if args.gpu not in (None, "cpu", "none"):
        compute = configs_mod.load_yaml(Path(cfg["_config_path"]).resolve().parent / cfg["compute_config"])
        thresholds = compute.get("idle_thresholds", {})
        if args.gpu == "auto":
            ids = gpu_mod.pick_free_gpus(
                1,
                max_mem_used_mb=int(thresholds.get("max_mem_used_mb", 1000)),
                max_util_pct=int(thresholds.get("max_util_pct", 5)),
            )
        else:
            ids = [int(args.gpu)]
        gpu_mod.pin(ids)
        device_pref = "cuda"
        print(f"pinned CUDA_VISIBLE_DEVICES={','.join(map(str, ids))}")
    elif not args.smoke:
        print("[warn] no --gpu given for a real run; using CPU. Pass --gpu auto for the A6000s.")

    run(cfg, smoke=args.smoke, limit=args.limit, device_pref=device_pref)


if __name__ == "__main__":
    main()
