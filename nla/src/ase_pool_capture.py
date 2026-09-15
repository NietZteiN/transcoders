"""H-R6 pool capture: layer-K span states + type/role tags for EVERY aligned renamed snippet, test and
fit-pool alike, in one forward pass per prompt (nla/configs/ase_vectors.yaml, prereg
log/nla-harness/2026-09-14_better-vector-prereg.md).

Reuses `ase_steer_run.prepare_residual` unchanged -- their `SteeredCausalLM` prompt builder, `align()`,
the measured special-token offset -- so the states here are the states the bake-off arms write into.
The split into TEST (the 50 bake-off snippets) and POOL is NOT made here; `ase_vectors.py` makes it
from `paths.test_subset`, so this file holds nothing that depends on which snippets are tested.

Output (`paths.pool`): torch.save of
  {"layer": K, "model": id, "spans": [ {snippet, span_idx, name, decoy, n_ren_tok, n_orig_tok,
                                        h0_mean, h1b_mean, h0_tok, h1b_tok, type, role, kind, usage} ],
   "excluded": {snippet: reason}, "tags_missing": [snippet...]}
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ase_roles                                        # noqa: E402
from ase_steer_run import TAG as _T, TEMP, TOP_P, prepare_residual   # noqa: E402

TAG = "[POOL]"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs" / "ase_vectors.yaml"))
    ap.add_argument("--artifact", default="/scratch/juno/jvl210002/ase2026/LLM-Attention-Fixation_submission")
    ap.add_argument("--model-id", default="codellama/CodeLlama-7b-Instruct-hf")
    ap.add_argument("--limit", type=int, default=0, help="smoke: first N renamed packs only")
    ap.add_argument("--out", default=None, help="override paths.pool (smoke)")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    low = args.model_id.lower()
    if any(v in low for v in ("qwen", "deepseek", "yi-", "glm", "internlm", "baichuan")):
        print(f"{TAG} REFUSED: {args.model_id} is a Chinese model."); return 2

    sys.path.insert(0, args.artifact)
    from models import SteeredCausalLM
    K = int(cfg["layer"])
    packs = [json.loads(l) for l in open(cfg["paths"]["packs_renamed"]) if l.strip()]
    packs = [p for p in packs if "pack" in p]
    if args.limit:
        packs = packs[: args.limit]
    man = {json.loads(l)["snippet"]: json.loads(l) for l in open(cfg["paths"]["manifest"]) if l.strip()}
    print(f"{TAG} {len(packs)} renamed packs · K={K} · model={args.model_id}", flush=True)

    cache_dir = os.path.join(os.environ["HF_HOME"], "hub") if os.environ.get("HF_HOME") else None
    lm = SteeredCausalLM()
    lm.config(model_name=args.model_id, max_new_tokens=8, temperature=TEMP, top_p=TOP_P,
              key_scope="prompt", cache_dir=cache_dir)
    lm.build()
    t0 = time.time()
    prep, excluded = prepare_residual(lm, packs, cfg["paths"]["packs_orig"], cfg["paths"]["manifest"], K, TAG)
    if not prep:
        print(f"{TAG} FATAL: nothing aligned"); return 3

    spans, tags_missing = [], []
    for rec in packs:
        sid = rec["snippet"]
        if sid not in prep:
            continue
        try:
            tags = ase_roles.tag_snippet(rec["java_path"], man[sid]["rename_map"])
        except Exception as e:                       # a parse failure is recorded, never guessed around
            print(f"{TAG} tags failed on {sid}: {type(e).__name__}: {e}", flush=True)
            tags_missing.append(sid); tags = {}
        for j, sp in enumerate(prep[sid]["spans"]):
            t = tags.get(sp["decoy"])
            spans.append({"snippet": sid, "span_idx": j, "name": sp["name"], "decoy": sp["decoy"],
                          "n_ren_tok": len(sp["ren_pos"]), "n_orig_tok": int(sp["h0_tok"].shape[0]),
                          "h0_mean": sp["h0_mean"].half(), "h1b_mean": sp["h1b_mean"].half(),
                          "h0_tok": sp["h0_tok"].half(), "h1b_tok": sp["h1b_tok"].half(),
                          "type": t.type if t else "?", "role": t.role if t else "?",
                          "kind": t.kind if t else "?", "usage": list(t.usage) if t else []})
    out = Path(args.out or cfg["paths"]["pool"]); out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"layer": K, "model": args.model_id, "config_sha": _sha(args.config), "spans": spans,
                "excluded": excluded, "tags_missing": tags_missing,
                "n_snippets": len(prep)}, out)
    from collections import Counter
    roles = Counter(s["role"] for s in spans)
    print(f"{TAG} saved {len(spans)} spans / {len(prep)} snippets to {out} · {len(excluded)} excluded · "
          f"tags missing {len(tags_missing)} · roles {dict(roles)} · {(time.time()-t0)/60:.1f} min", flush=True)
    return 0


def _sha(p: str) -> str:
    import hashlib
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]


if __name__ == "__main__":
    raise SystemExit(main())
