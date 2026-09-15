"""Text-only Gemma-3 checkpoint from a multimodal snapshot.

`google/gemma-3-4b-it` ships as `Gemma3ForConditionalGeneration` with keys under
`language_model.model.*` plus a SigLIP tower. `Gemma3ForCausalLM.from_pretrained` on that
checkpoint silently loads NOTHING (every key is reported unexpected/missing and the model is
random-init — verified 2026-09-08 on c-03-04: chat reply was eight newlines). The released 12B
NLA sidecars are text-only `Gemma3ForCausalLM` checkpoints (`model_type gemma3_text`) that load
through `AutoModelForCausalLM`, and the AV/AR pairs trained here are saved in that same layout.

`ensure_text_checkpoint` materialises the text-only checkpoint ONCE (remap
`language_model.model.X -> model.X`, drop the vision tower, `config.json` = the text sub-config
with `architectures=[Gemma3ForCausalLM]`, tokenizer copied). Everything downstream (host forward
passes, AV starting point, AR trunk) then uses plain `from_pretrained` on that directory, so
truncating the AR trunk to `K+1` layers is the ordinary `config=` override the vendored recipe
uses (`nla/vendor/nla-repo`, TRAINING_NOTES.md).
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoTokenizer, Gemma3Config, Gemma3ForCausalLM, Gemma3TextConfig

_PREFIX = "language_model.model."
_TOK_FILES = ("tokenizer.json", "tokenizer.model", "tokenizer_config.json",
              "special_tokens_map.json", "added_tokens.json", "chat_template.json",
              "generation_config.json")


def snapshot_dir(model_id: str) -> Path:
    """Resolve a hub id to its local snapshot dir (offline; HF_HOME must hold it)."""
    p = Path(model_id)
    if p.is_dir():
        return p
    home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    snaps = home / "hub" / ("models--" + model_id.replace("/", "--")) / "snapshots"
    cands = sorted(snaps.iterdir()) if snaps.is_dir() else []
    if not cands:
        raise FileNotFoundError(f"no local snapshot for {model_id} under {snaps}")
    return cands[-1]


def text_config_from_multimodal(snap: Path) -> Gemma3TextConfig:
    full = Gemma3Config.from_pretrained(snap)
    cfg = Gemma3TextConfig(**full.text_config.to_dict())
    cfg.architectures = ["Gemma3ForCausalLM"]
    for k in ("bos_token_id", "eos_token_id", "pad_token_id"):
        if getattr(cfg, k, None) is None:
            setattr(cfg, k, getattr(full, k, None))
    return cfg


def ensure_text_checkpoint(model_id: str, out_dir: Path, dtype: torch.dtype = torch.bfloat16) -> Path:
    """Write (once) a text-only `Gemma3ForCausalLM` checkpoint of `model_id` into `out_dir`."""
    out_dir = Path(out_dir)
    done = out_dir / "TEXT_ONLY_DONE"
    if done.exists():
        return out_dir
    snap = snapshot_dir(model_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    index = json.loads((snap / "model.safetensors.index.json").read_text())["weight_map"]
    by_shard: dict[str, list[str]] = {}
    for k, shard in index.items():
        if k.startswith(_PREFIX):
            by_shard.setdefault(shard, []).append(k)
    new_map: dict[str, str] = {}
    total = 0
    for i, (shard, keys) in enumerate(sorted(by_shard.items()), 1):
        tensors = {}
        with safe_open(snap / shard, framework="pt", device="cpu") as f:
            for k in keys:
                nk = "model." + k[len(_PREFIX):]
                tensors[nk] = f.get_tensor(k).to(dtype).contiguous()
                total += tensors[nk].numel() * tensors[nk].element_size()
        name = f"model-{i:05d}-of-{len(by_shard):05d}.safetensors"
        save_file(tensors, out_dir / name, metadata={"format": "pt"})
        new_map.update({k: name for k in tensors})
    (out_dir / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": total, "source": str(snap)}, "weight_map": new_map}, indent=1))
    cfg = text_config_from_multimodal(snap)
    cfg.dtype = str(dtype).replace("torch.", "")
    cfg.save_pretrained(out_dir)
    for fn in _TOK_FILES:
        if (snap / fn).exists():
            shutil.copy(snap / fn, out_dir / fn)
    done.write_text(f"source={snap}\nkeys={len(new_map)}\n")
    return out_dir


def load_gemma_text(ckpt_dir: Path, n_layers: int | None = None,
                    dtype: torch.dtype = torch.bfloat16, device: str = "cpu") -> Gemma3ForCausalLM:
    """`Gemma3ForCausalLM` from a text-only checkpoint, optionally truncated to `n_layers` blocks.

    Truncation sets `num_hidden_layers` (and slices `layer_types`) BEFORE `from_pretrained`, as
    the vendored critic does; the layers above are simply never instantiated. Layer weights below
    the cut are checked for presence so a wrong prefix cannot hand back a random-init trunk again.
    """
    ckpt_dir = Path(ckpt_dir)
    cfg = Gemma3TextConfig.from_pretrained(ckpt_dir)
    if n_layers is not None:
        assert 1 <= n_layers <= cfg.num_hidden_layers, n_layers
        cfg.num_hidden_layers = n_layers
        cfg.layer_types = list(cfg.layer_types[:n_layers])
    model = Gemma3ForCausalLM.from_pretrained(ckpt_dir, config=cfg, dtype=dtype)
    index = json.loads((ckpt_dir / "model.safetensors.index.json").read_text())["weight_map"] \
        if (ckpt_dir / "model.safetensors.index.json").exists() else None
    if index is not None:
        want = [n for n, _ in model.named_parameters() if n != "lm_head.weight"]
        absent = [n for n in want if n not in index]
        if absent:
            raise RuntimeError(f"{len(absent)} parameters not in checkpoint (e.g. {absent[:3]})")
    return model.to(device)


def load_tokenizer(ckpt_dir: Path):
    return AutoTokenizer.from_pretrained(Path(ckpt_dir))
