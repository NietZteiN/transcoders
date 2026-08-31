"""The obfuscation ladder, read-side: does the residual stream ever beat a token count?

WHAT THIS SETTLES. `last_prompt` ~ reply length has been shown on L0 and L1b (+0.0431 / +0.0615
over each tier's own baseline, 2026-08-30). Those are one point of the ladder each, and both sit on
the ATOM-level route. The study's frame is Schulte's Block Model with two documented failure
routes — atom-level interference (L1/L1b) and relational overload (L2/L3) — and the relational half
has never been probed at all. Two outcomes are worth having:

  * the residual stream ties length at every tier  -> "last_prompt carries nothing item-level" is
    general, and the programme's whole ledger of item-level nulls has one explanation.
  * it beats length on L2/L3 but not L1/L1b        -> the two routes differ in what they leave
    readable, which is a genuine mechanistic result and the first positive available.

UNIFORM REGIME ON PURPOSE. All five tiers are generated fresh here at the same budget with the same
draw count, rather than reusing the banked L0/L1b draws. Mixing generation regimes is exactly what
produced the retracted tier positive on 2026-08-30, where L0's rho was compared against a baseline
collected under different conditions.

DRAWS. Decoding is greedy, so repeated identical runs differ only through the nondeterminism floor
measured on 2026-08-29 (0.85-0.90 per-item agreement). That is precisely what makes k/K a
lower-variance label than any single grade, and it is why K > 1 here.

Activations are extracted ONCE per tier: reads are bit-exact (2026-08-30), so repeating them
would buy nothing.

Env `nla-mi`, one GPU per tier.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics as st
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_NLA_ROOT = _HERE.parent
_PROJ = _NLA_ROOT.parent
sys.path.insert(0, str(_NLA_ROOT / "vendor" / "nla-repo"))
sys.path.insert(0, str(_HERE))

SEED = 20260724
MAX_NEW_GEN = 2048


def load_tier(tier: str, rng: random.Random) -> list[dict]:
    """Snippets having both L0 and `tier`, with ground truth and a usable call on the tier side."""
    from task_bank import build_call
    rows = []
    for ds in ("dataset_a", "dataset_b"):
        p = _PROJ / "data" / "stimuli" / ds / f"{ds}.jsonl"
        if p.exists():
            rows += [json.loads(l) for l in open(p) if l.strip()]
    by: dict[str, dict] = {}
    for r in rows:
        by.setdefault(r["snippet_id"], {})[r["tier"]] = r
    out = []
    for sid, tiers in sorted(by.items()):
        a, b = tiers.get("L0"), tiers.get(tier)
        if not a or not b or not b.get("expected_output"):
            continue
        call = build_call(b)
        if not call:
            continue
        out.append({"snippet_id": sid, "code": b["code"], "call": call,
                    "truth": b["expected_output"], "language": b.get("language")})
    rng.shuffle(out)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", required=True, choices=["L0", "L1", "L1b", "L2", "L3"])
    ap.add_argument("--draws", type=int, default=5)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", default=str(_PROJ / "data/nla/p0/p1b/ladder"))
    args = ap.parse_args()

    import torch
    from layer_rotation import all_layer_acts
    from steer_run import TARGET_MODEL, build_user, graded
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(SEED)
    out = Path(args.out_dir) / args.tier
    out.mkdir(parents=True, exist_ok=True)
    items = load_tier(args.tier, random.Random(SEED))
    if args.limit:
        items = items[:args.limit]
    tokz = AutoTokenizer.from_pretrained(TARGET_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        TARGET_MODEL, torch_dtype=torch.bfloat16, device_map=args.device).eval()
    print(f"[ladder:{args.tier}] {len(items)} items · {args.draws} draws", flush=True)

    draws_p = out / "draws.jsonl"
    done = set()
    if draws_p.exists():
        for l in open(draws_p):
            if l.strip():
                r = json.loads(l)
                done.add((r["draw"], r["snippet_id"]))
    sink = open(draws_p, "a")

    for d in range(args.draws):
        for i, it in enumerate(items):
            if (d, it["snippet_id"]) in done:
                continue
            user = build_user(it["code"], it["call"])
            ids = tokz.apply_chat_template([{"role": "user", "content": user}], tokenize=True,
                                           add_generation_prompt=True, return_dict=False)
            with torch.no_grad():
                o = model.generate(torch.tensor([ids], device=model.device),
                                   max_new_tokens=MAX_NEW_GEN, do_sample=False,
                                   pad_token_id=tokz.eos_token_id)
            text = tokz.decode(o[0][len(ids):], skip_special_tokens=True)
            got, ok = graded(text, it["truth"])
            sink.write(json.dumps({"draw": d, "snippet_id": it["snippet_id"], "tier": args.tier,
                                   "correct": bool(ok), "parsed": got is not None,
                                   "answer": got, "reply_chars": len(text),
                                   "n_gen": int(o.shape[1]) - len(ids)}) + "\n")
            sink.flush()
        print(f"[ladder:{args.tier}] draw {d+1}/{args.draws} done", flush=True)
    sink.close()

    # Activations once: reads are bit-exact, so repeats buy nothing.
    acts_p = out / "acts.npy"
    if not acts_p.exists():
        A = np.stack([all_layer_acts(model, tokz, build_user(it["code"], it["call"]))
                      for it in items])
        np.save(acts_p, A)
        (out / "items.json").write_text(json.dumps([it["snippet_id"] for it in items], indent=2))
        print(f"[ladder:{args.tier}] acts {A.shape} -> {acts_p}", flush=True)

    rows = [json.loads(l) for l in open(draws_p) if l.strip()]
    by_item: dict[str, list[int]] = {}
    for r in rows:
        by_item.setdefault(r["snippet_id"], []).append(int(r["correct"]))
    (out / "manifest.json").write_text(json.dumps({
        "tier": args.tier, "seed": SEED, "max_new_gen": MAX_NEW_GEN, "draws": args.draws,
        "n_items": len(items), "n_rows": len(rows),
        "mean_correct": round(st.mean(r["correct"] for r in rows), 4),
        "mean_reply_chars": round(st.mean(r["reply_chars"] for r in rows), 1),
        "mean_parse_rate": round(st.mean(r["parsed"] for r in rows), 4),
        "n_unanimous": sum(1 for v in by_item.values() if len(set(v)) == 1),
        "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2))
    print(f"[ladder:{args.tier}] acc {st.mean(r['correct'] for r in rows):.4f} · "
          f"parse {st.mean(r['parsed'] for r in rows):.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
