"""Count the items a host gets right clean and wrong obfuscated — the denominator every accuracy
claim in this programme divides by.

On Gemma-3-12B-it that count is **6 of 60**, so a *perfect* rescue is worth +0.100 accuracy and the
standing power gate (>= 9 flippable at n = 60, twice the 0.85-0.90 greedy reproducibility floor)
fails. Every causal result in the W family therefore rests on a likelihood readout, not behaviour.

The obvious fix -- a bigger corpus -- is unavailable and it took reading the loader to find out:
`load_pairs` already reads BOTH stimulus files, so Dataset A (20 snippets) and Dataset B (50) are
the same 70 that become 60 after call-construction filtering. There is no untapped corpus.

What is left is the host. Gemma's adversarial-renaming penalty is around 10 points against the
21.25% collapse the behavioural papers report, so it may simply be a weak trap host. Llama-3.1-8B
is permitted, cached, ~16 GB in bf16, already wired as HOSTS["llama8b"], and is the one panel model
with pretrained SAEs -- so if its flippable count clears the gate, the accuracy question becomes
askable for the first time AND the charter's feature-level work has a host.

No steering, no NLA, no vectors: two greedy generations per item, graded with the same parser every
banked run used, so the number is directly comparable to Gemma's 6/60.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_PROJ = _HERE.parent.parent
sys.path.insert(0, str(_HERE))

from steer_run import HOSTS, MAX_NEW_GEN, SEED, build_user, graded, load_pairs  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama8b", choices=sorted(HOSTS))
    ap.add_argument("--allow-banked-host", action="store_true")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-hours", type=float, default=4.0)
    args = ap.parse_args()

    if args.model == "qwen7b" and not args.allow_banked_host:
        print("[FC] REFUSED: qwen7b is a Chinese model; pass --allow-banked-host.")
        return 2

    model_name, _ = HOSTS[args.model]
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokz = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, device_map=args.device).eval()
    print(f"[FC] {args.model} = {model_name}", flush=True)

    def gen(user: str) -> str:
        # apply_chat_template returns a BatchEncoding, not a bare tensor, in this transformers
        # version -- generate() then fails on `inputs_tensor.shape`. Ask for ids explicitly and
        # accept either shape, so a library change cannot silently break this again.
        enc = tokz.apply_chat_template([{"role": "user", "content": user}],
                                       tokenize=True, add_generation_prompt=True,
                                       return_tensors="pt", return_dict=True)
        ids = enc["input_ids"] if hasattr(enc, "keys") else enc
        ids = ids.to(model.device)
        with torch.no_grad():
            o = model.generate(ids, max_new_tokens=MAX_NEW_GEN, do_sample=False,
                               pad_token_id=tokz.pad_token_id or tokz.eos_token_id)
        return tokz.decode(o[0, ids.shape[1]:], skip_special_tokens=True)

    pairs = load_pairs(args.limit, random.Random(SEED))
    sink = open(out / "census_rows.jsonl", "w")
    rows = []
    for i, p in enumerate(pairs):
        if time.time() - t0 > args.max_hours * 3600:
            print("[FC] wall-clock stop", flush=True); break
        _, ok0 = graded(gen(build_user(p["code_l0"], p["call_l0"])), p["truth"])
        got1, ok1 = graded(gen(build_user(p["code_l1b"], p["call_l1b"])), p["truth"])
        r = {"snippet_id": p["snippet_id"], "language": p.get("language"),
             "l0_correct": bool(ok0), "l1b_correct": bool(ok1),
             "l1b_parsed": got1 is not None,
             "flippable": bool(ok0 and not ok1)}
        rows.append(r); sink.write(json.dumps(r) + "\n"); sink.flush()
        print(f"[FC] {i+1}/{len(pairs)} {p['snippet_id']} L0={ok0} L1b={ok1}"
              f"{'  <-- FLIPPABLE' if r['flippable'] else ''}", flush=True)
    sink.close()

    n = len(rows)
    a0 = sum(r["l0_correct"] for r in rows); a1 = sum(r["l1b_correct"] for r in rows)
    fl = sum(r["flippable"] for r in rows)
    gate = 9 * n / 60 if n else 0        # the standing gate, scaled to whatever n completed
    st = {"host": args.model, "model": model_name, "n": n,
          "l0_accuracy": a0 / n if n else None, "l1b_accuracy": a1 / n if n else None,
          "l1b_penalty_points": (a0 - a1) / n * 100 if n else None,
          "flippable": fl, "flippable_frac": fl / n if n else None,
          "power_gate_threshold": gate, "clears_power_gate": bool(fl >= gate),
          "gemma_reference": {"n": 60, "flippable": 6, "perfect_rescue": 0.100},
          "seed": SEED}
    json.dump(st, open(out / "census_stats.json", "w"), indent=2)
    print("\n" + json.dumps(st, indent=2), flush=True)
    print(f"\n[FC] {args.model}: L0 {a0}/{n}, L1b {a1}/{n}, penalty "
          f"{st['l1b_penalty_points']:.1f} pts, FLIPPABLE {fl}/{n} "
          f"(gate {gate:.1f}) -> {'CLEARS' if st['clears_power_gate'] else 'FAILS'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
