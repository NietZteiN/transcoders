"""Phase B — train NLA verbaliser (AV) + reconstructor (AR) pairs at every layer of a small host.

Plain-PyTorch re-implementation of the vendored recipe (`nla/vendor/nla-repo`: TRAINING_NOTES.md,
configs/actor_sft.sh, configs/critic_sft.sh, nla/datagen/*), minus Miles/SGLang/Ray/FSDP and minus
the RL stage. One process, one GPU, one layer per SFT invocation; the SLURM array in
`nla/scripts/nla_ml_layer.sh` fans the 34 layers out. Config: `nla/configs/nla_ml.yaml`.

Stages (all resumable; `--stage`):
  corpus    CPU. Ultra-FineWeb shard -> docs.parquet (token ids) + rows.parquet (doc, pos, split,
            holdout, text_truncated). Positions per doc from the recipe's sha256(seed|doc_id) RNG.
  extract   GPU. One forward per doc batch with a hook on EVERY decoder block; per-layer fp32
            memmaps acts/L{K}.npy [N_rows, d] + acts/norms.json (per-layer mean/median raw norm
            and the frozen injection_scale_K).
  explain   GPU. Local Gemma-3-12B-it writes the stage-2 explanation for each row's truncated
            text with the recipe's prompt/regex/cleaning. `--shard i --n-shards n`; `--join`
            merges shards into explain/explanations.parquet. Explanations describe the TEXT, so
            one corpus serves all 34 layers (the recipe's own layer-independence).
  sft_av    GPU. Full fine-tune of the host on [prompt=AV template with the vector injected at
            the embedding row of ㈜; response=<explanation>..</explanation>], CE on response
            tokens. Saves a text-only Gemma3ForCausalLM + nla_meta.yaml that `LocalAV` /
            `NLAClient` load unchanged.
  sft_ar    GPU. Truncated (K+1 layers) host, final norm & lm_head removed, identity-init
            value_head; loss = 2(1-cos) on the last token. Saves what `NLACritic` loads.
  check     GPU. Held-out controls: AV loss real vs permuted vectors, AR fve real vs shuffled
            targets, greedy reads (no-tag / CJK rate) and cycle fidelity cos(AR(AV(v)), v).

Layer convention: `layer_index K` = output of decoder block K = HF `hidden_states[K+1]`
(vendored extractors.py:64, our extract.py / steer.py). Captured by forward hooks, never by
`output_hidden_states` (its last entry is post-final-norm).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml

_HERE = Path(__file__).resolve().parent
_PROJ = _HERE.parent.parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_PROJ / "nla" / "vendor" / "nla-repo"))
from nla_inference import (EXPLANATION_RE, inject_at_marked_positions,  # noqa: E402
                           load_nla_config, normalize_activation)
from nla.schema import wrap_explanation  # noqa: E402
from fsdp_util import (DistCtx, all_gather_tensor, all_reduce_sum, clip_grad_norm, dist_cleanup,
                       dist_setup, gather_full_state_dict, rank_microbatches, reduce_replicated_grads,
                       replicated_params, scale_for_world, set_grad_sync, shard_model)
from gemma_text import ensure_text_checkpoint, load_gemma_text  # noqa: E402


def _load_stage2_defs() -> dict:
    """Pull the stage-2 prompt, response regex and cleaner out of the vendored module WITHOUT
    importing it (its import chain needs the `anthropic` client, which this env does not have).
    Only top-level assignments/defs of the listed names are executed, so the vendored text stays
    the single source of truth rather than a copy that can drift."""
    import ast
    src = (_PROJ / "nla/vendor/nla-repo/nla/datagen/stage2_api_explain.py").read_text()
    want = {"_DEFAULT_INSTRUCTION", "_DEFAULT_RESPONSE_PATTERN", "_MIN_FEATURES",
            "_LIST_PREFIX_RE", "_BOLD_WRAP_RE", "_extract_and_clean"}
    keep = []
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in want for t in node.targets):
            keep.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in want:
            keep.append(node)
    ns = {"re": re}
    exec(compile(ast.Module(body=keep, type_ignores=[]), "stage2_api_explain.py", "exec"), ns)
    assert want <= set(ns), want - set(ns)
    return ns


_S2 = _load_stage2_defs()
_DEFAULT_INSTRUCTION, _DEFAULT_RESPONSE_PATTERN = _S2["_DEFAULT_INSTRUCTION"], _S2["_DEFAULT_RESPONSE_PATTERN"]
_extract_and_clean = _S2["_extract_and_clean"]

# Byte-identical to the released kitft sidecars (checked 2026-09-08 against
# hf_home/hub/models--kitft--nla-gemma3-12b-L32-{av,ar}); `load_nla_config` re-verifies the
# injection token and its neighbours through the live tokenizer at every load.
AV_TEMPLATE = ("You are a meticulous AI researcher conducting an important investigation into "
               "activation vectors from a language model. Your overall task is to describe the "
               "semantic content of that activation vector.\n\nWe will pass the vector enclosed in "
               "<concept> tags into your context. You must then produce an explanation for the "
               "vector, enclosed within <explanation> tags. The explanation consists of 2-3 text "
               "snippets describing that vector.\n\nHere is the vector:\n\n<concept>{injection_char}"
               "</concept>\n\nPlease provide an explanation.")
AR_TEMPLATE = "Summary of the following text: <text>{explanation}</text> <summary>"
INJECTION_CHAR = "㈜"        # U+321C, Gemma-3 tokenizer id 246566 (shared 4B/12B)
CJK_RE = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯]")
EXPERIMENT = "ML_multilayer_nla"


def utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_PROJ, text=True).strip()
    except Exception:
        return "unknown"


def load_cfg(path: Path) -> dict:
    cfg = yaml.safe_load(path.read_text())
    cfg["_path"] = str(path)
    cfg["_sha256"] = sha256_file(path)
    return cfg


def root(cfg: dict) -> Path:
    return _PROJ / cfg["paths"]["root"]


def provenance(cfg: dict, extra: dict | None = None) -> dict:
    d = {"experiment": EXPERIMENT, "timestamp": utc(), "config": cfg["_path"],
         "config_sha256": cfg["_sha256"], "script_sha256": sha256_file(Path(__file__)),
         "git_head": git_head(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
         "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
         "node": os.environ.get("SLURMD_NODENAME"), "cuda_visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
         "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
         "seed": cfg["seed"]}
    if extra:
        d.update(extra)
    return d


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ceil_2sf(x: float) -> float:
    """Round UP to two significant figures (the frozen injection_scale rule)."""
    if x <= 0:
        raise ValueError(x)
    e = math.floor(math.log10(x)) - 1
    return math.ceil(x / 10 ** e) * 10 ** e


# ─── tokens block (sidecar) ─────────────────────────────────────────────────────────────────

def tokens_block(tok) -> dict:
    """Resolve the sidecar `tokens:` block through the LIVE tokenizer + chat template."""
    inj = tok.encode(INJECTION_CHAR, add_special_tokens=False)
    assert len(inj) == 1, inj
    ids = tok.apply_chat_template([{"role": "user", "content": AV_TEMPLATE.format(injection_char=INJECTION_CHAR)}],
                                  tokenize=True, add_generation_prompt=True, return_dict=False)
    sites = [i for i, t in enumerate(ids) if t == inj[0]]
    assert len(sites) == 1, sites
    p = sites[0]
    # critic_suffix_ids: the recipe stores the tail of `</text> <summary>` that must close every
    # AR prompt; the released 12B sidecar keeps 5 ids (drops the leading `>` merge token).
    suf = tok("</text> <summary>", add_special_tokens=False)["input_ids"]
    return {"injection_char": INJECTION_CHAR, "injection_token_id": inj[0],
            "injection_left_neighbor_id": ids[p - 1], "injection_right_neighbor_id": ids[p + 1],
            "critic_suffix_ids": suf[1:]}


def write_sidecar(out: Path, role: str, cfg: dict, layer: int, tok, injection_scale: float | None,
                  training: dict) -> None:
    d = cfg["host"]["d_model"]
    meta = {"kind": "nla_model", "schema_version": 2, "role": role, "stage": "sl",
            "d_model": d,
            "extraction": {"injection_scale": injection_scale, "mse_scale": math.sqrt(d)},
            "tokens": tokens_block(tok),
            "prompt_templates": {"av": AV_TEMPLATE, "ar": AR_TEMPLATE},
            "created_at": utc(), "created_by": "nla_train.py (transcoders Phase B, plain PyTorch SFT)",
            "training": training, "extraction_layer_index": layer,
            "host": {"model_id": cfg["host"]["model_id"], "n_layers": cfg["host"]["n_layers"]}}
    if role == "ar":
        meta["critic"] = {"extraction_layer_index": layer}
    (out / "nla_meta.yaml").write_text(yaml.safe_dump(meta, allow_unicode=True, sort_keys=False))


# ─── stage: corpus ──────────────────────────────────────────────────────────────────────────

def sample_positions(token_ids: list[int], n: int, special_ids: set[int], doc_id: str,
                     seed: int, min_position: int) -> list[int]:
    """Recipe `_sample_positions` verbatim (stage0_extract.py:54): per-doc sha256 RNG."""
    rng = random.Random(hashlib.sha256(f"{seed}|{doc_id}".encode()).digest())
    cands = [i for i, t in enumerate(token_ids) if i >= min_position and t not in special_ids]
    if not cands:
        return []
    return rng.sample(cands, k=min(n, len(cands)))


def stage_corpus(cfg: dict, args) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    c = cfg["corpus"]
    out = root(cfg) / "corpus"; out.mkdir(parents=True, exist_ok=True)
    if (out / "rows.parquet").exists() and not args.force:
        print("[corpus] exists; --force to rebuild"); return
    tok = AutoTokenizer.from_pretrained(_PROJ / cfg["host"]["text_ckpt"])
    special = set(tok.all_special_ids)
    n_docs = args.limit or c["n_docs"]
    shard = _PROJ / c["shard"]
    shard_name = shard.name
    pf = pq.ParquetFile(shard)
    docs, rows = [], []
    n_seen = n_short = 0
    for batch in pf.iter_batches(batch_size=2048, columns=[c["text_column"]]):
        texts = batch.column(0).to_pylist()
        enc = tok(texts, add_special_tokens=True, truncation=True, max_length=c["max_length"])["input_ids"]
        for text, ids in zip(texts, enc):
            doc_idx = n_seen; n_seen += 1
            if len(ids) < c["min_tokens"]:
                n_short += 1; continue
            doc_id = f"{c['source']}:{shard_name}:{doc_idx}"
            pos = sample_positions(ids, c["positions_per_doc"], special, doc_id, c["sample_seed"], c["min_position"])
            if len(pos) < c["positions_per_doc"]:
                n_short += 1; continue
            side = "av" if int(hashlib.sha256(f"{cfg['seed']}|split|{doc_id}".encode()).hexdigest(), 16) % 1000 < c["av_frac"] * 1000 else "ar"
            docs.append({"doc_idx": doc_idx, "doc_id": doc_id, "split": side, "n_tokens": len(ids),
                         "token_ids": ids, "content_sha256": hashlib.sha256(text.encode()).hexdigest()})
            for p in sorted(pos):
                rows.append({"doc_idx": doc_idx, "pos": p, "split": side, "n_raw_tokens": p + 1,
                             "text_truncated": tok.decode(ids[:p + 1], skip_special_tokens=True)})
            if len(docs) >= n_docs:
                break
        if len(docs) >= n_docs:
            break
    # hold-out = the LAST holdout_docs_per_side docs of each side, in corpus order.
    for side in ("av", "ar"):
        ids_side = [d["doc_idx"] for d in docs if d["split"] == side]
        hold = set(ids_side[-c["holdout_docs_per_side"]:]) if not args.limit else set(ids_side[-2:])
        for r in rows:
            if r["split"] == side:
                r["holdout"] = r["doc_idx"] in hold
    for i, r in enumerate(rows):
        r["row_id"] = i
    pq.write_table(pa.Table.from_pylist(docs), out / "docs.parquet")
    pq.write_table(pa.Table.from_pylist(rows), out / "rows.parquet")
    summ = {"n_docs": len(docs), "n_rows": len(rows), "n_seen": n_seen, "n_short_skipped": n_short,
            "n_av_rows": sum(r["split"] == "av" for r in rows), "n_ar_rows": sum(r["split"] == "ar" for r in rows),
            "n_holdout_rows": sum(r["holdout"] for r in rows), "shard": str(shard), "shard_sha256_head": None,
            "tokenizer": str(_PROJ / cfg["host"]["text_ckpt"]), **provenance(cfg)}
    (out / "corpus_summary.json").write_text(json.dumps(summ, indent=1))
    print("[corpus]", json.dumps({k: v for k, v in summ.items() if not k.startswith("_")}, indent=1))


# ─── stage: extract ─────────────────────────────────────────────────────────────────────────

class AllLayerHooks:
    """Forward hooks on every decoder block; `.gather(idx)` copies out the sampled positions."""

    def __init__(self, layers):
        self.out: dict[int, torch.Tensor] = {}
        self.h = [l.register_forward_hook(self._mk(i)) for i, l in enumerate(layers)]

    def _mk(self, i):
        def hook(_m, _i, o):
            self.out[i] = o[0] if isinstance(o, tuple) else o
        return hook

    def close(self):
        for h in self.h:
            h.remove()


def stage_extract(cfg: dict, args) -> None:
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    seed_all(cfg["seed"])
    ck = _PROJ / cfg["host"]["text_ckpt"]
    out = root(cfg) / "acts"; out.mkdir(parents=True, exist_ok=True)
    if (out / "norms.json").exists() and not args.force:
        print("[extract] exists; --force to redo"); return
    docs = pq.read_table(root(cfg) / "corpus/docs.parquet").to_pylist()
    rows = pq.read_table(root(cfg) / "corpus/rows.parquet", columns=["row_id", "doc_idx", "pos"]).to_pylist()
    by_doc: dict[int, list[tuple[int, int]]] = {}
    for r in rows:
        by_doc.setdefault(r["doc_idx"], []).append((r["row_id"], r["pos"]))
    N, d, L = len(rows), cfg["host"]["d_model"], cfg["host"]["n_layers"]
    tok = AutoTokenizer.from_pretrained(ck)
    model = load_gemma_text(ck, dtype=torch.bfloat16, device=args.device).eval()
    layers = model.model.layers
    assert len(layers) == L
    # fp32, not fp16: the smoke measured mean norms up to 78 000 at layer 33 (> fp16 max 65 504) with the
    # mass in a few dimensions, so fp16 would overflow to inf on late layers. 2 GB/layer, 70 GB total.
    mm = [np.lib.format.open_memmap(out / f"L{K}.npy", mode="w+", dtype=np.float32, shape=(N, d)) for K in range(L)]
    norms = np.zeros((L, N), dtype=np.float32)
    hooks = AllLayerHooks(layers)
    # sanity: the hook convention equals hidden_states[K+1] for K < L-1 on one doc
    x = torch.tensor([docs[0]["token_ids"][:64]], device=args.device)
    with torch.no_grad():
        o = model(input_ids=x, output_hidden_states=True, use_cache=False)
    for K in (0, L // 2, L - 2):
        assert torch.allclose(hooks.out[K].float(), o.hidden_states[K + 1].float()), K
    assert not torch.allclose(hooks.out[L - 1].float(), o.hidden_states[L].float()), "last hs is post-norm"
    docs.sort(key=lambda dd: -dd["n_tokens"])
    bs, t0, done = args.batch_size or 16, time.time(), 0
    pad = tok.pad_token_id
    for i in range(0, len(docs), bs):
        chunk = docs[i:i + bs]
        T = max(dd["n_tokens"] for dd in chunk)
        ids = torch.full((len(chunk), T), pad, dtype=torch.long)
        am = torch.zeros((len(chunk), T), dtype=torch.long)
        for b, dd in enumerate(chunk):
            ids[b, :dd["n_tokens"]] = torch.tensor(dd["token_ids"]); am[b, :dd["n_tokens"]] = 1
        with torch.no_grad():
            model(input_ids=ids.to(args.device), attention_mask=am.to(args.device), use_cache=False)
        bi = torch.tensor([b for b, dd in enumerate(chunk) for _ in by_doc[dd["doc_idx"]]], device=args.device)
        pi = torch.tensor([p for dd in chunk for _, p in by_doc[dd["doc_idx"]]], device=args.device)
        ri = np.array([r for dd in chunk for r, _ in by_doc[dd["doc_idx"]]])
        for K in range(L):
            v = hooks.out[K][bi, pi]                     # [n, d] bf16
            norms[K, ri] = v.float().norm(dim=-1).cpu().numpy()
            mm[K][ri] = v.float().cpu().numpy()
        done += len(chunk)
        if (i // bs) % 20 == 0:
            print(f"[extract] {done}/{len(docs)} docs · T={T} · {time.time() - t0:.0f}s · "
                  f"mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GB", flush=True)
    hooks.close()
    for m in mm:
        m.flush()
    split = pq.read_table(root(cfg) / "corpus/rows.parquet", columns=["row_id", "split", "holdout"]).to_pylist()
    av_train = np.array([r["row_id"] for r in split if r["split"] == "av" and not r["holdout"]])
    stats = {}
    for K in range(L):
        n = norms[K]
        mean_av = float(n[av_train].mean())
        stats[K] = {"mean_all": float(n.mean()), "median_all": float(np.median(n)),
                    "p1": float(np.percentile(n, 1)), "p99": float(np.percentile(n, 99)),
                    "max": float(n.max()), "mean_av_train": mean_av,
                    "injection_scale": ceil_2sf(mean_av), "max_abs_element": float(np.abs(mm[K][:min(N, 2000)]).max())}
    np.save(out / "norms.npy", norms)
    (out / "norms.json").write_text(json.dumps({"rule": cfg["av"]["injection_scale_rule"], "n_rows": N,
                                                 "layers": stats, **provenance(cfg)}, indent=1))
    print("[extract] injection scales:", {K: s["injection_scale"] for K, s in stats.items()})
    print("[extract] max |element| (first 2000 rows):", {K: round(s["max_abs_element"]) for K, s in stats.items()})


# ─── stage: explain ─────────────────────────────────────────────────────────────────────────

def clean_explanation(raw: str, min_features: int) -> str | None:
    """Recipe stage-2 post-processing: regex + list-marker stripping, drop < min_features."""
    cleaned = _extract_and_clean(raw, _DEFAULT_RESPONSE_PATTERN)
    if cleaned is None or len(cleaned.split("\n\n")) < min_features:
        return None
    return cleaned


def stage_explain(cfg: dict, args) -> None:
    import pyarrow.parquet as pq
    from transformers import AutoModelForCausalLM, AutoTokenizer
    e = cfg["explainer"]
    out = root(cfg) / "explain"; out.mkdir(parents=True, exist_ok=True)
    rows = pq.read_table(root(cfg) / "corpus/rows.parquet", columns=["row_id", "text_truncated"]).to_pylist()
    if args.join:
        import pyarrow as pa
        keep, n_raw, n_drop = [], 0, 0
        for f in sorted(out.glob("shard*.jsonl")):
            for line in f.read_text().splitlines():
                r = json.loads(line); n_raw += 1
                if r["explanation"] is None:
                    n_drop += 1; continue
                keep.append({"row_id": r["row_id"], "explanation": r["explanation"]})
        keep.sort(key=lambda r: r["row_id"])
        pq.write_table(pa.Table.from_pylist(keep), out / "explanations.parquet")
        summ = {"n_generated": n_raw, "n_dropped": n_drop, "n_kept": len(keep), "n_rows_total": len(rows),
                "drop_rate": n_drop / max(1, n_raw), **provenance(cfg)}
        (out / "explain_summary.json").write_text(json.dumps(summ, indent=1))
        print("[explain/join]", {k: v for k, v in summ.items() if not k.startswith("_")}); return
    n_sh = args.n_shards or e["n_shards"]
    mine = [r for r in rows if r["row_id"] % n_sh == args.shard]
    if args.limit:
        mine = mine[:args.limit]
    path = out / f"shard{args.shard:02d}_of{n_sh:02d}.jsonl"
    done = set()
    if path.exists():
        done = {json.loads(l)["row_id"] for l in path.read_text().splitlines() if l.strip()}
    todo = [r for r in mine if r["row_id"] not in done]
    print(f"[explain] shard {args.shard}/{n_sh}: {len(mine)} rows, {len(done)} done, {len(todo)} to do", flush=True)
    if not todo:
        return
    seed_all(cfg["seed"] + 1000 + args.shard)
    tok = AutoTokenizer.from_pretrained(e["model_id"]); tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(e["model_id"], dtype=torch.bfloat16, device_map=args.device).eval()
    prompts = []
    for r in todo:
        content = _DEFAULT_INSTRUCTION.format(text=r["text_truncated"])
        prompts.append(tok.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True))
    order = sorted(range(len(todo)), key=lambda i: -len(prompts[i]))   # length-sorted batches
    bs, t0, n_done, n_null = args.batch_size or e["batch_size"], time.time(), 0, 0
    with path.open("a") as fh:
        for s in range(0, len(order), bs):
            idx = order[s:s + bs]
            enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(args.device)
            with torch.no_grad():
                g = model.generate(**enc, do_sample=True, temperature=e["temperature"], top_p=e["top_p"], top_k=e["top_k"],
                                   max_new_tokens=e["max_new_tokens"], pad_token_id=tok.pad_token_id)
            outs = tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            for i, raw in zip(idx, outs):
                expl = clean_explanation(raw, e["min_features"])
                n_null += expl is None
                fh.write(json.dumps({"row_id": todo[i]["row_id"], "raw": raw, "explanation": expl}, ensure_ascii=False) + "\n")
            fh.flush(); n_done += len(idx)
            if (s // bs) % 10 == 0:
                el = time.time() - t0
                print(f"[explain] {n_done}/{len(todo)} · null {n_null} · {el:.0f}s · ETA {el / n_done * (len(todo) - n_done) / 60:.0f} min · "
                      f"mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GB", flush=True)
            if args.max_hours and (time.time() - t0) / 3600 > args.max_hours:
                print("[explain] max-hours reached; resumable"); break


# ─── training helpers ───────────────────────────────────────────────────────────────────────

def lr_at(step: int, total: int, lr: float, min_lr: float, warmup: int) -> float:
    if step < warmup:
        return lr * (step + 1) / warmup
    p = min(1.0, (step - warmup) / max(1, total - warmup))
    return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * p))


def load_rows_with_expl(cfg: dict, side: str) -> tuple[list[dict], list[dict]]:
    """(train rows, holdout rows) of one side that have an explanation; sorted by row_id."""
    import pyarrow.parquet as pq
    rows = pq.read_table(root(cfg) / "corpus/rows.parquet", columns=["row_id", "split", "holdout", "doc_idx", "pos"]).to_pylist()
    ex = pq.read_table(root(cfg) / "explain/explanations.parquet").to_pylist()
    emap = {r["row_id"]: r["explanation"] for r in ex}
    tr, ho = [], []
    for r in rows:
        if r["split"] != side or r["row_id"] not in emap:
            continue
        r["explanation"] = emap[r["row_id"]]
        (ho if r["holdout"] else tr).append(r)
    return tr, ho


def acts_memmap(cfg: dict, layer: int) -> np.memmap:
    return np.load(root(cfg) / f"acts/L{layer}.npy", mmap_mode="r")


def injection_scale_for(cfg: dict, layer: int) -> float:
    return float(json.loads((root(cfg) / "acts/norms.json").read_text())["layers"][str(layer)]["injection_scale"])


def gpu_mem_gb() -> float:
    return torch.cuda.get_device_properties(0).total_memory / 2**30 if torch.cuda.is_available() else 0.0


class TrainLog:
    def __init__(self, path: Path):
        self.fh = path.open("a")

    def __call__(self, **kw):
        kw["t"] = utc(); self.fh.write(json.dumps(kw) + "\n"); self.fh.flush()
        print("[train]", " ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in kw.items()), flush=True)


# ─── stage: sft_av ──────────────────────────────────────────────────────────────────────────

class AVBatcher:
    """Tokenised AV rows: shared prompt ids (identical for every row) + per-row response ids."""

    def __init__(self, tok, rows: list[dict], max_response_tokens: int):
        self.tok = tok
        self.prompt = tok.apply_chat_template([{"role": "user", "content": AV_TEMPLATE.format(injection_char=INJECTION_CHAR)}],
                                              tokenize=True, add_generation_prompt=True, return_dict=False)
        tb = tokens_block(tok)
        self.inj_id, self.left, self.right = tb["injection_token_id"], tb["injection_left_neighbor_id"], tb["injection_right_neighbor_id"]
        self.inj_pos = [i for i, t in enumerate(self.prompt) if t == self.inj_id]
        assert len(self.inj_pos) == 1
        self.eot = tok.convert_tokens_to_ids("<end_of_turn>")
        assert isinstance(self.eot, int) and self.eot >= 0
        self.rows = rows
        self.resp = [tok(wrap_explanation(r["explanation"]), add_special_tokens=False)["input_ids"][:max_response_tokens - 1] + [self.eot]
                     for r in rows]
        self.pad = tok.pad_token_id

    def collate(self, idx: list[int], vecs: torch.Tensor) -> dict:
        P = len(self.prompt)
        T = P + max(len(self.resp[i]) for i in idx)
        ids = torch.full((len(idx), T), self.pad, dtype=torch.long)
        lab = torch.full((len(idx), T), -100, dtype=torch.long)
        am = torch.zeros((len(idx), T), dtype=torch.long)
        for b, i in enumerate(idx):
            seq = self.prompt + self.resp[i]
            ids[b, :len(seq)] = torch.tensor(seq); am[b, :len(seq)] = 1
            lab[b, P:len(seq)] = torch.tensor(self.resp[i])
        return {"input_ids": ids, "attention_mask": am, "labels": lab, "vectors": vecs}


class EmbeddingInjector:
    """Forward hook on the (√d-scaled) input embedding: overwrite the ㈜ row with the vector.

    Mirrors `NLAClient._build_embeds`: embeds × embed_scale, THEN `inject_at_marked_positions`
    with `normalize_activation(v, injection_scale)`. Gemma's `Gemma3TextScaledWordEmbedding`
    applies the √d scale inside forward, so hooking its OUTPUT is exactly the post-scale row.
    """

    def __init__(self, embed, inj_id: int, left: int, right: int, scale: float):
        self.inj_id, self.left, self.right, self.scale = inj_id, left, right, scale
        self.vectors: torch.Tensor | None = None
        self.n_written = 0
        self.h = embed.register_forward_hook(self._hook)

    def _hook(self, _m, inputs, output):
        if self.vectors is None:
            return None
        ids = inputs[0]
        v = normalize_activation(self.vectors.to(output.device).float(), self.scale)
        out = inject_at_marked_positions(ids, output.float(), v, self.inj_id, self.left, self.right)
        self.n_written += ids.shape[0]
        return out.to(output.dtype)

    def close(self):
        self.h.remove()


def av_loss(model, batch: dict, device: str, chunk: int = 2048) -> tuple[torch.Tensor, int]:
    """CE summed over response tokens; lm_head applied only at label positions (262k vocab)."""
    ids, am, lab = (batch[k].to(device) for k in ("input_ids", "attention_mask", "labels"))
    h = model.model(input_ids=ids, attention_mask=am, use_cache=False).last_hidden_state
    # labels are next-token targets: position t predicts lab[t]; shift so h[t-1] -> lab[t]
    tgt = lab[:, 1:]; hs = h[:, :-1]
    sel = tgt != -100
    hsel, tsel = hs[sel], tgt[sel]
    total = torch.zeros((), device=device, dtype=torch.float32)
    for s in range(0, hsel.shape[0], chunk):
        logits = model.lm_head(hsel[s:s + chunk]).float()
        total = total + torch.nn.functional.cross_entropy(logits, tsel[s:s + chunk], reduction="sum")
    return total, int(sel.sum())


def subsample_train(tr: list, cfg: dict, K: int, frac: float | None, tag: str) -> list:
    """Take a seeded random FRACTION of the training rows, leaving the holdout untouched.

    H-C4 (dose-response) needs fidelity measured at several data volumes and compared against the
    BANKED 100 % run, so the evaluation must not move: `--limit` also truncates the holdout (to 32
    av / 64 ar), which would make `holdout_fve` / `holdout_gap` incomparable with the banked
    858/848-row numbers and silently turn a dose-response into a noise measurement.

    Random rather than a prefix: rows come out of the corpus grouped by document, so `tr[:N]` would
    subsample DOCUMENTS as well as rows and confound data volume with topic coverage. Seeded off the
    config seed, the layer and the side, so a given (layer, side, frac) is reproducible.
    """
    if frac is None or frac == 1.0:
        return tr
    if not 0.0 < frac <= 1.0:
        raise ValueError(f"--train-frac must lie in (0, 1], got {frac!r}")
    rng = random.Random(f"{cfg['seed']}|train_frac|{K}|{tag}")
    idx = list(range(len(tr)))
    rng.shuffle(idx)
    keep = sorted(idx[:max(1, round(frac * len(tr)))])
    return [tr[i] for i in keep]


def stage_sft_av(cfg: dict, args) -> None:
    from transformers import AutoTokenizer
    a = cfg["av"]; K = args.layer
    # H-C10 same-steps control: --epochs overrides the config so "half the rows, twice the passes"
    # can be run at EQUAL optimiser steps to the 100 % / 1-epoch run. `--train-frac` at a fixed
    # 1 epoch varies rows AND steps together, so on its own it cannot say which of the two the
    # dose effect is. Copied rather than mutated so cfg stays the on-disk config for provenance.
    if getattr(args, "epochs", None):
        a = {**a, "epochs": int(args.epochs)}
    out = root(cfg) / f"L{K}" / "av"; out.mkdir(parents=True, exist_ok=True)
    if (out / "nla_meta.yaml").exists() and not args.force:
        print(f"[sft_av] L{K} exists; --force to retrain"); return
    seed_all(cfg["seed"] + K)
    ck = _PROJ / cfg["host"]["text_ckpt"]
    tok = AutoTokenizer.from_pretrained(ck)
    tr, ho = load_rows_with_expl(cfg, "av")
    if args.limit:
        tr, ho = tr[:args.limit], ho[:min(len(ho), 32)]
    n_tr_full = len(tr)
    tr = subsample_train(tr, cfg, K, getattr(args, "train_frac", None), "av")
    scale = injection_scale_for(cfg, K)
    acts = acts_memmap(cfg, K)
    bat = AVBatcher(tok, tr, a["max_response_tokens"])
    hob = AVBatcher(tok, ho, a["max_response_tokens"])
    mb = args.micro_batch or a["micro_batch"]
    gb = min(a["global_batch"], len(tr)); accum = max(1, gb // mb)   # min() only bites in --limit smokes
    steps_per_epoch = len(tr) // gb
    total = steps_per_epoch * a["epochs"]
    if args.max_steps:
        total = min(total, args.max_steps)
    _dc = getattr(args, "dist", DistCtx())
    if _dc.is_main:
        print(f"[sft_av] L{K}: {len(tr)} train rows, {len(ho)} holdout, injection_scale {scale}, "
              f"micro {mb} x accum {accum} = {gb}, {total} steps, GPU {gpu_mem_gb():.0f} GB", flush=True)
    if _dc.enabled:
        # DIAGNOSTIC (2026-09-11): jobs 389416/389569/389666/389781 all ended with one rank in
        # dist_cleanup's barrier after step 1 with NO exception. Ranks disagreeing on the loop bounds
        # is the only remaining explanation, so every rank announces them.
        print(f"[rank{_dc.rank}/{_dc.world}] len(tr)={len(tr)} len(ho)={len(ho)} gb={gb} mb={mb} "
              f"accum={accum} steps_per_epoch={steps_per_epoch} total={total} epochs={a['epochs']}", flush=True)
    model = load_gemma_text(ck, dtype=torch.float32, device=args.device)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    DC: DistCtx = getattr(args, "dist", DistCtx())
    model = shard_model(model, DC)
    # Only the decoder blocks are sharded (av_loss enters model.model / model.lm_head directly, which
    # bypasses a root wrapper's hooks, and lm_head is tied to the embedding). The replicated
    # remainder is outside FSDP's reduction, so its grads are averaged by hand each step.
    REP = replicated_params(model, DC)
    # The injector is an OUTPUT-only forward hook on the embedding (never touches params), so it is
    # safe on a sharded module. fused AdamW does not support DTensor -> foreach under --fsdp (same
    # AdamW, not bit-identical; the equivalence test's tolerance accounts for it).
    inj = EmbeddingInjector(model.model.embed_tokens, bat.inj_id, bat.left, bat.right, scale)
    opt = torch.optim.AdamW(model.parameters(), lr=a["lr"], weight_decay=a["weight_decay"],
                            fused=not DC.enabled)
    log = TrainLog(out / "train_log.jsonl")
    rng = random.Random(cfg["seed"] + K)
    order = list(range(len(tr)))
    t0, step, tok_seen = time.time(), 0, 0
    for ep in range(a["epochs"]):
        rng.shuffle(order)
        for s in range(0, steps_per_epoch * gb, gb):
            if step >= total:
                break
            gidx = order[s:s + gb]
            n_tok_global = sum(len(bat.resp[i]) for i in gidx)
            lr = lr_at(step, total, a["lr"], a["min_lr"], a["warmup_steps"])
            for g in opt.param_groups:
                g["lr"] = lr
            loss_acc = 0.0
            starts = list(range(0, gb, mb))
            mine = rank_microbatches(len(starts), DC)
            for j, mi in enumerate(mine):
                # Reduce-scatter gradients ONLY on the last microbatch of the step: these nodes have
                # no NVLink (SYS, cross-socket), so per-microbatch syncing is 8x the traffic and blew
                # the NCCL watchdog in job 389416. Same arithmetic, 8x less communication.
                set_grad_sync(model, j == len(mine) - 1, DC)
                m = starts[mi]
                idx = gidx[m:m + mb]
                vecs = torch.from_numpy(np.stack([acts[tr[i]["row_id"]] for i in idx]).astype(np.float32))
                batch = bat.collate(idx, vecs)
                inj.vectors = batch["vectors"]; inj.n_written = 0
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    tot, n = av_loss(model, batch, args.device)
                assert inj.n_written == len(idx), (inj.n_written, len(idx))
                scale_for_world(tot / n_tok_global, DC).backward()
                loss_acc += float(tot) / n_tok_global
            inj.vectors = None
            reduce_replicated_grads(REP, DC)
            gn = clip_grad_norm(model.parameters(), a["grad_clip"], DC)
            opt.step(); opt.zero_grad(set_to_none=True)
            (loss_acc,) = all_reduce_sum(loss_acc, ctx=DC)   # ranks hold disjoint microbatches
            step += 1; tok_seen += n_tok_global
            if (step % 10 == 0 or step == 1 or step == total) and DC.is_main:
                el = time.time() - t0
                log(step=step, loss=loss_acc, lr=lr, grad_norm=float(gn), tok_per_s=tok_seen / el,
                    eta_min=el / step * (total - step) / 60, mem_gb=torch.cuda.max_memory_allocated() / 2**30)
    if DC.enabled:
        print(f"[rank{DC.rank}] TRAIN LOOP EXITED after step={step} of total={total}", flush=True)
    # ── held-out control: real vectors vs permuted (vector of another row) ──
    model.eval()
    ev = {}
    with torch.no_grad():
        for name, perm in (("real", False), ("permuted", True)):
            tot_l, tot_n = 0.0, 0
            hstarts = list(range(0, len(ho), mb))
            for mi in rank_microbatches(len(hstarts), DC):
                m = hstarts[mi]
                idx = list(range(m, min(len(ho), m + mb)))
                src = [ho[(i + 1) % len(ho)]["row_id"] if perm else ho[i]["row_id"] for i in idx]
                vecs = torch.from_numpy(np.stack([acts[r] for r in src]).astype(np.float32))
                batch = hob.collate(idx, vecs)
                inj.vectors = vecs
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    tot, n = av_loss(model, batch, args.device)
                tot_l += float(tot); tot_n += n
            tot_l, tot_n = all_reduce_sum(tot_l, tot_n, ctx=DC)
            ev[f"holdout_loss_{name}"] = tot_l / max(1, tot_n)
    inj.vectors = None; inj.close()
    ev["holdout_gap_permuted_minus_real"] = ev["holdout_loss_permuted"] - ev["holdout_loss_real"]
    if DC.is_main:
        print("[sft_av] holdout", ev, flush=True)
    training = {"lr": a["lr"], "min_lr": a["min_lr"], "warmup_steps": a["warmup_steps"], "global_batch_size": gb,
                "micro_batch": mb, "epochs": a["epochs"], "steps": step, "n_train_rows": len(tr),
                "train_frac": getattr(args, "train_frac", None),
                "n_train_rows_full": n_tr_full,
                "loss_type": "sft_loss", "weight_decay": a["weight_decay"], "grad_clip": a["grad_clip"],
                "precision": "fp32 master + bf16 autocast", "wall_s": time.time() - t0}
    sd = gather_full_state_dict(model, DC)
    if not DC.is_main:
        return
    if DC.enabled:
        model.save_pretrained(out, state_dict={k: v.to(torch.bfloat16) for k, v in sd.items()},
                              safe_serialization=True)
    else:
        model.to(torch.bfloat16).save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)
    write_sidecar(out, "av", cfg, K, tok, scale, training)
    (out / "eval.json").write_text(json.dumps({**ev, "n_holdout": len(ho)}, indent=1))
    (out / "provenance.json").write_text(json.dumps(provenance(cfg, {"layer": K, "training": training}), indent=1))
    load_nla_config(out, tok)      # the sidecar must round-trip through the vendored loader
    print(f"[sft_av] L{K} saved -> {out}")


# ─── stage: sft_ar ──────────────────────────────────────────────────────────────────────────

class ARBatcher:
    def __init__(self, tok, rows: list[dict], suffix_ids: list[int]):
        self.rows = rows
        self.ids = []
        n_bad = 0
        for r in rows:
            ids = tok(AR_TEMPLATE.format(explanation=r["explanation"]), add_special_tokens=True)["input_ids"]
            if ids[-len(suffix_ids):] != suffix_ids:
                n_bad += 1
            self.ids.append(ids)
        # recipe (stage3_build.py:145) asserts the suffix per row; here it is counted and reported
        # because a merged BPE boundary is a data defect, not a reason to stop a 34-layer array.
        self.n_bad_suffix = n_bad
        self.pad = tok.pad_token_id
        assert tok("x", add_special_tokens=True)["input_ids"][0] == tok.bos_token_id

    def collate(self, idx: list[int]) -> dict:
        T = max(len(self.ids[i]) for i in idx)
        ids = torch.full((len(idx), T), self.pad, dtype=torch.long)
        am = torch.zeros((len(idx), T), dtype=torch.long)
        last = torch.zeros(len(idx), dtype=torch.long)
        for b, i in enumerate(idx):
            n = len(self.ids[i]); ids[b, :n] = torch.tensor(self.ids[i]); am[b, :n] = 1; last[b] = n - 1
        return {"input_ids": ids, "attention_mask": am, "last": last}


def ar_loss(pred: torch.Tensor, gold: torch.Tensor, mse_scale: float) -> torch.Tensor:
    """Recipe critic loss: mse(normalize(pred, √d), normalize(gold, √d)).mean(-1) == 2(1-cos)."""
    p = normalize_activation(pred.float(), mse_scale); g = normalize_activation(gold.float(), mse_scale)
    return ((p - g) ** 2).mean(-1)


def fve(pred: torch.Tensor, gold: torch.Tensor, mse_scale: float) -> float:
    """1 - mean((pred_n-gold_n)^2)/mean((gold_n-μ)^2), μ = mean of the normalised gold."""
    p = normalize_activation(pred.float(), mse_scale); g = normalize_activation(gold.float(), mse_scale)
    mu = g.mean(0, keepdim=True)
    return float(1 - ((p - g) ** 2).mean() / ((g - mu) ** 2).mean())


def stage_sft_ar(cfg: dict, args) -> None:
    from safetensors.torch import save_file
    from transformers import AutoTokenizer
    a = cfg["ar"]; K = args.layer
    # H-C10 same-steps control: --epochs overrides the config so "half the rows, twice the passes"
    # can be run at EQUAL optimiser steps to the 100 % / 1-epoch run. `--train-frac` at a fixed
    # 1 epoch varies rows AND steps together, so on its own it cannot say which of the two the
    # dose effect is. Copied rather than mutated so cfg stays the on-disk config for provenance.
    if getattr(args, "epochs", None):
        a = {**a, "epochs": int(args.epochs)}
    out = root(cfg) / f"L{K}" / "ar"; out.mkdir(parents=True, exist_ok=True)
    if (out / "nla_meta.yaml").exists() and not args.force:
        print(f"[sft_ar] L{K} exists; --force to retrain"); return
    seed_all(cfg["seed"] + 100 + K)
    ck = _PROJ / cfg["host"]["text_ckpt"]
    tok = AutoTokenizer.from_pretrained(ck)
    d = cfg["host"]["d_model"]; ms = math.sqrt(d)
    tr, ho = load_rows_with_expl(cfg, "ar")
    if args.limit:
        tr, ho = tr[:args.limit], ho[:min(len(ho), 64)]
    n_tr_full = len(tr)
    tr = subsample_train(tr, cfg, K, getattr(args, "train_frac", None), "ar")
    acts = acts_memmap(cfg, K)
    suffix = tokens_block(tok)["critic_suffix_ids"]
    bat, hob = ARBatcher(tok, tr, suffix), ARBatcher(tok, ho, suffix)
    mb = args.micro_batch or a["micro_batch"]; gb = min(a["global_batch"], len(tr))
    total = (len(tr) // gb) * a["epochs"]
    if args.max_steps:
        total = min(total, args.max_steps)
    if getattr(args, "dist", DistCtx()).is_main:
        print(f"[sft_ar] L{K}: {len(tr)} train rows ({bat.n_bad_suffix} bad suffix), {len(ho)} holdout, "
              f"{K + 1}-layer trunk, {total} steps", flush=True)
    model = load_gemma_text(ck, n_layers=K + 1, dtype=torch.float32, device=args.device)
    model.model.norm = torch.nn.Identity()          # value head sees the raw block-K output
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    head = torch.nn.Linear(d, d, bias=False, dtype=torch.float32, device=args.device)
    with torch.no_grad():
        head.weight.copy_(torch.eye(d))
    DC: DistCtx = getattr(args, "dist", DistCtx())
    # Identity() replaces model.model.norm BEFORE sharding; the value head is sharded alongside the
    # trunk so its gradients are reduced by the same machinery rather than by hand. FSDP2 keeps one
    # DTensor per named parameter, which is why the lm_head exclusion below still works.
    model = shard_model(model, DC)
    params = [p for n, p in model.named_parameters() if not n.startswith("lm_head")] + list(head.parameters())
    # head stays replicated with the embedding/norm: it is 14.7 M params at d=3840 and sharding it
    # would buy nothing while adding another entry point that bypasses FSDP's hooks.
    REP = replicated_params(model, DC, extra_params=list(head.parameters()))
    opt = torch.optim.AdamW(params, lr=a["lr"], weight_decay=a["weight_decay"], fused=not DC.enabled)
    log = TrainLog(out / "train_log.jsonl")
    rng = random.Random(cfg["seed"] + 100 + K)
    order = list(range(len(tr)))
    t0, step = time.time(), 0

    def forward(b: dict, gold: torch.Tensor):
        ids, am, last = (b[k].to(args.device) for k in ("input_ids", "attention_mask", "last"))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = model.model(input_ids=ids, attention_mask=am, use_cache=False).last_hidden_state
        hl = h[torch.arange(len(last), device=args.device), last].float()
        return head(hl), gold.to(args.device)

    for ep in range(a["epochs"]):
        rng.shuffle(order)
        for s in range(0, (len(tr) // gb) * gb, gb):
            if step >= total:
                break
            gidx = order[s:s + gb]
            lr = lr_at(step, total, a["lr"], a["min_lr"], a["warmup_steps"])
            for g in opt.param_groups:
                g["lr"] = lr
            loss_acc = 0.0
            starts = list(range(0, gb, mb))
            mine = rank_microbatches(len(starts), DC)
            for j, mi in enumerate(mine):
                set_grad_sync(model, j == len(mine) - 1, DC)   # see the AV stage / fsdp_util
                m = starts[mi]
                idx = gidx[m:m + mb]
                gold = torch.from_numpy(np.stack([acts[tr[i]["row_id"]] for i in idx]).astype(np.float32))
                pred, gold = forward(bat.collate(idx), gold)
                loss = ar_loss(pred, gold, ms).sum() / gb
                scale_for_world(loss, DC).backward(); loss_acc += float(loss)
            reduce_replicated_grads(REP, DC)
            gn = clip_grad_norm(params, a["grad_clip"], DC)
            opt.step(); opt.zero_grad(set_to_none=True); step += 1
            (loss_acc,) = all_reduce_sum(loss_acc, ctx=DC)
            if (step % 10 == 0 or step == 1 or step == total) and DC.is_main:
                el = time.time() - t0
                log(step=step, loss=loss_acc, lr=lr, grad_norm=float(gn), eta_min=el / step * (total - step) / 60,
                    mem_gb=torch.cuda.max_memory_allocated() / 2**30)
    # ── held-out: fve real, fve vs shuffled targets, raw-mean baseline ──
    model.eval(); head.eval()
    preds, golds, rows = [], [], []
    hstarts = list(range(0, len(ho), mb))
    with torch.no_grad():
        for mi in rank_microbatches(len(hstarts), DC):
            m = hstarts[mi]
            idx = list(range(m, min(len(ho), m + mb)))
            gold = torch.from_numpy(np.stack([acts[ho[i]["row_id"]] for i in idx]).astype(np.float32))
            p, g = forward(hob.collate(idx), gold); preds.append(p.cpu()); golds.append(g.cpu())
            rows.append(torch.tensor(idx))
    P, G, R = torch.cat(preds), torch.cat(golds), torch.cat(rows)
    if DC.enabled:
        P, G, R = (all_gather_tensor(x, DC) for x in (P, G, R))
        order = torch.argsort(R)                    # restore single-GPU row order
        P, G = P[order], G[order]
    perm = torch.randperm(len(G), generator=torch.Generator().manual_seed(cfg["seed"]))
    tr_mean = torch.from_numpy(np.stack([acts[tr[i]["row_id"]] for i in range(0, len(tr), max(1, len(tr) // 2000))]).astype(np.float32)).mean(0, keepdim=True)
    ev = {"holdout_fve": fve(P, G, ms), "holdout_fve_shuffled": fve(P, G[perm], ms),
          "holdout_fve_train_mean_baseline": fve(tr_mean.expand_as(G), G, ms),
          "holdout_cos_mean": float(torch.nn.functional.cosine_similarity(P, G, dim=-1).mean()),
          "holdout_cos_shuffled_mean": float(torch.nn.functional.cosine_similarity(P, G[perm], dim=-1).mean()),
          "holdout_loss": float(ar_loss(P, G, ms).mean()), "n_holdout": len(ho), "n_bad_suffix_train": bat.n_bad_suffix}
    if DC.is_main:
        print("[sft_ar] holdout", ev, flush=True)
    training = {"lr": a["lr"], "min_lr": a["min_lr"], "warmup_steps": a["warmup_steps"], "global_batch_size": gb,
                "micro_batch": mb, "epochs": a["epochs"], "steps": step, "n_train_rows": len(tr),
                "train_frac": getattr(args, "train_frac", None),
                "n_train_rows_full": n_tr_full,
                "loss_type": "custom_loss(2(1-cos))", "weight_decay": a["weight_decay"], "grad_clip": a["grad_clip"],
                "precision": "fp32 master + bf16 autocast", "wall_s": time.time() - t0, "trunk_layers": K + 1}
    sd = gather_full_state_dict(model, DC)
    hw = head.weight.detach()          # replicated, identical on every rank
    if not DC.is_main:
        return
    if DC.enabled:
        model.save_pretrained(out, state_dict={k: v.to(torch.bfloat16) for k, v in sd.items()},
                              safe_serialization=True)
    else:
        model.to(torch.bfloat16).save_pretrained(out, safe_serialization=True)
    save_file({"weight": hw.to(torch.bfloat16).cpu().contiguous()}, out / "value_head.safetensors")
    tok.save_pretrained(out)
    write_sidecar(out, "ar", cfg, K, tok, None, training)
    (out / "eval.json").write_text(json.dumps(ev, indent=1))
    (out / "provenance.json").write_text(json.dumps(provenance(cfg, {"layer": K, "training": training}), indent=1))
    print(f"[sft_ar] L{K} saved -> {out}")


# ─── stage: check ───────────────────────────────────────────────────────────────────────────

def stage_check(cfg: dict, args) -> None:
    """Greedy reads of held-out AV vectors through the trained pair: tag/CJK rates and cycle cos."""
    from local_av import LocalAV
    from nla_inference import NLACritic
    K = args.layer; c = cfg["check"]
    ldir = root(cfg) / f"L{K}"
    av = LocalAV(ldir / "av", device=args.device)
    ar = NLACritic(ldir / "ar", device=args.device)
    _, ho = load_rows_with_expl(cfg, "av")
    acts = acts_memmap(cfg, K)
    rng = random.Random(cfg["seed"]); rng.shuffle(ho)
    ho = ho[:args.limit or c["n_reads"]]
    reads, t0 = [], time.time()
    for i, r in enumerate(ho):
        v = torch.from_numpy(np.asarray(acts[r["row_id"]], dtype=np.float32))
        text = av.generate(v, max_new_tokens=c["max_new_tokens"], temperature=0.0, extract_explanation=False)
        m = EXPLANATION_RE.search(text)
        body = m.group(1).strip() if m else text.strip()
        rec = ar.reconstruct(body)
        j = (i + 1) % len(ho)
        v_other = torch.from_numpy(np.asarray(acts[ho[j]["row_id"]], dtype=np.float32))
        reads.append({"row_id": r["row_id"], "read": text, "has_tags": m is not None,
                      "cjk": bool(CJK_RE.search(text)), "n_words": len(body.split()),
                      "cos_cycle": float(torch.nn.functional.cosine_similarity(rec, v, dim=0)),
                      "cos_other": float(torch.nn.functional.cosine_similarity(rec, v_other, dim=0)),
                      "cos_gold_expl": float(torch.nn.functional.cosine_similarity(ar.reconstruct(r["explanation"]), v, dim=0))})
    n = len(reads)
    summ = {"layer": K, "n_reads": n, "n_no_tags": sum(not r["has_tags"] for r in reads),
            "cjk_rate": sum(r["cjk"] for r in reads) / n, "mean_words": float(np.mean([r["n_words"] for r in reads])),
            "cos_cycle_mean": float(np.mean([r["cos_cycle"] for r in reads])),
            "cos_cycle_median": float(np.median([r["cos_cycle"] for r in reads])),
            "cos_other_mean": float(np.mean([r["cos_other"] for r in reads])),
            "cos_gold_expl_mean": float(np.mean([r["cos_gold_expl"] for r in reads])),
            "n_duplicate_reads": n - len({r["read"] for r in reads}), "wall_s": time.time() - t0,
            **provenance(cfg, {"av_eval": json.loads((ldir / "av/eval.json").read_text()),
                               "ar_eval": json.loads((ldir / "ar/eval.json").read_text())})}
    (ldir / "check.json").write_text(json.dumps(summ, indent=1))
    (ldir / "check_reads.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in reads))
    print("[check]", json.dumps({k: v for k, v in summ.items() if k not in ("av_eval", "ar_eval") and not k.startswith("_")}, indent=1))
    for r in reads[:3]:
        print("  read:", r["read"][:300].replace("\n", " | "))


# ─── main ───────────────────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["corpus", "extract", "explain", "sft_av", "sft_ar", "check", "host_ckpt"])
    ap.add_argument("--config", default=str(_PROJ / "nla/configs/nla_ml.yaml"))
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=None, help="smoke: rows/docs cap")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None,
                    help="H-C10: override av/ar epochs (config default 1) for the same-steps control")
    ap.add_argument("--train-frac", type=float, default=None,
                    help="H-C4 dose-response: seeded random fraction of the TRAIN rows; the "
                         "holdout is left intact so evals stay comparable with the banked run")
    ap.add_argument("--micro-batch", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None, help="extract: docs/forward (default 16); explain: rows/generate (config)")
    ap.add_argument("--root", default=None, help="override paths.root (smoke runs write elsewhere)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=None)
    ap.add_argument("--join", action="store_true")
    ap.add_argument("--max-hours", type=float, default=None)
    ap.add_argument("--force", action="store_true")
    # --fsdp: shard the SFT model over the GPUs of ONE node with FSDP2 (launch via torchrun).
    # Needed only for hosts whose fp32 weights+grads+Adam exceed one GPU (12B: ~206 GB vs 143).
    # Absent => the proven single-GPU path is untouched, byte for byte. See nla/src/fsdp_util.py.
    ap.add_argument("--fsdp", action="store_true")
    args = ap.parse_args()
    cfg = load_cfg(Path(args.config))
    if args.root:
        cfg["paths"]["root"] = args.root
    if "qwen" in cfg["host"]["model_id"].lower() or "qwen" in cfg["explainer"]["model_id"].lower():
        print("REFUSED: Chinese models are not run in this project."); return 2
    if args.stage in ("sft_av", "sft_ar", "check"):
        assert args.layer is not None and 0 <= args.layer < cfg["host"]["n_layers"], args.layer
    if args.stage == "host_ckpt":
        p = ensure_text_checkpoint(cfg["host"]["model_id"], _PROJ / cfg["host"]["text_ckpt"]); print("[host_ckpt]", p); return 0
    # Join the process group BEFORE materialising the host checkpoint: under torchrun every rank
    # runs main(), and two ranks writing the same text-only checkpoint concurrently would corrupt
    # it. Rank 0 writes, the others wait on the barrier (no-op when --fsdp is absent).
    args.dist = dist_setup(args.fsdp and args.stage in ("sft_av", "sft_ar"))
    if args.dist.enabled:
        args.device = args.dist.device
    if args.dist.is_main:
        ensure_text_checkpoint(cfg["host"]["model_id"], _PROJ / cfg["host"]["text_ckpt"])
    if args.dist.enabled:
        import torch.distributed as _dist
        _dist.barrier()
    try:
        {"corpus": stage_corpus, "extract": stage_extract, "explain": stage_explain,
         "sft_av": stage_sft_av, "sft_ar": stage_sft_ar, "check": stage_check}[args.stage](cfg, args)
    except BaseException:
        # Print the traceback HERE, before any collective. Python emits it only after `finally`
        # completes, and `finally` calls dist_cleanup() whose barrier hangs forever when the other
        # rank is still training -- so the watchdog SIGABRTs the process and the traceback is never
        # generated at all. Four 12B attempts (389416/389569/389666/389781/389786) reported only the
        # hung barrier for exactly this reason; PYTHONUNBUFFERED cannot help with a traceback that
        # was never produced.
        import traceback
        print(f"[rank{args.dist.rank}] EXCEPTION in stage {args.stage}:", flush=True)
        traceback.print_exc()
        sys.stdout.flush(); sys.stderr.flush()
        raise
    finally:
        dist_cleanup(args.dist)
    return 0


if __name__ == "__main__":
    sys.exit(main())
