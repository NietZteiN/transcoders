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

# Host registry for the model-constraint port. The ladder is host-agnostic — it generates and
# extracts, and neither depends on the NLA — so pointing it at Gemma needs only the model id and
# the layer its released pair was trained at. Default preserves the banked Qwen behaviour.
HOSTS = {
    "qwen7b":   ("Qwen/Qwen2.5-7B-Instruct", 20),
    "gemma12b": ("google/gemma-3-12b-it", 32),
    # Instrument 3's host under the model constraint. It has no NLA pair, but the ladder needs
    # none — it generates and extracts — so this yields the dense-probe baseline that E1/E2 must
    # beat, on the only panel model with both pretrained SAEs and transcoders. Layer 16 of 32 is
    # the mid-depth analogue of Qwen's 20/28; nothing here depends on it being NLA-instrumented.
    "llama8b":  ("meta-llama/Llama-3.1-8B-Instruct", 16),
}


# Budgets used across the ladder runs. Rows written before 2026-09-02 do not carry
# `max_new_gen`, so the cap is inferred from these; newer rows record it directly.
KNOWN_CAPS = frozenset({1100, 2048, 4096, 8192})


def terminated(row: dict) -> bool:
    """Did this generation finish, or did it run into the wall?

    A row whose n_gen equals its generation cap never terminated — the model was still going.
    That matters for two separate reasons, and conflating them is what inflated the reply-length
    baseline: its ANSWER is missing or truncated (so scoring it wrong is a guess, not a
    measurement), and its LENGTH is censored (the true length is unknown, only bounded below).
    Both make it unusable as a correctness label and as a length datum.

    Note this is not the same as `parsed`. A handful of rows emit an `Output:` line and then keep
    generating to the cap — answered, but still non-terminating — and two Llama rows terminated
    normally while emitting no parsable answer, which is a genuine wrong answer rather than a
    truncation.
    """
    cap = row.get("max_new_gen")
    return row["n_gen"] < cap if cap else row["n_gen"] not in KNOWN_CAPS


def split_terminated(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """(terminated, did_not_terminate) — callers report the second count rather than hiding it."""
    t = [r for r in rows if terminated(r)]
    return t, [r for r in rows if not terminated(r)]


def repetition_features(code: str) -> list[float]:
    """How repetitive is this text, independent of any structure it encodes?

    A dispatcher object repeats a lexical pattern N times, so a probe that appears to decode
    structure may only be counting repeats. On Qwen these five counts alone reached rho = +0.8346
    against the residual stream's +0.8842 — the step that turned a structural claim into a
    surface-statistical one. Lives here rather than in either scorer because both need it and
    importing across them is circular: p1b_span_probe already takes `stimuli` from
    p1b_l2_mechanism.
    """
    from collections import Counter
    toks = code.split()
    lines = [l.strip() for l in code.splitlines() if l.strip()]
    tc, lc = Counter(toks), Counter(lines)
    return [
        float(tc.most_common(1)[0][1]) if tc else 0.0,   # max token frequency
        float(len(tc)),                                   # distinct tokens
        float(len(toks)),                                 # total tokens
        float(sum(v for v in lc.values() if v > 1)),      # duplicated lines
        float(lc.most_common(1)[0][1]) if lc else 0.0,    # max line frequency
    ]


def read_draws(tier_dir) -> list[dict]:
    """Every draw for one tier, across shards.

    Draw-sharding writes draws.jsonl (shard 0) plus draws_dN.jsonl per later shard. A reader that
    opens only draws.jsonl silently scores HALF the draws and returns a plausible number — the
    exact shape of failure this project has twice mistaken for a result. Globbing removes the
    dependency on remembering to merge; the merge script remains for anyone who wants a single
    file, but nothing requires it.
    """
    import glob
    rows = []
    for f in sorted(glob.glob(str(Path(tier_dir) / "draws*.jsonl"))):
        for line in open(f):
            if line.strip():
                rows.append(json.loads(line))
    return rows


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
    # Draw-sharding: a tier's draws split across jobs so more GPUs run at once. Shards are
    # DISJOINT draw ranges writing to their own draws_dN.jsonl, so there is no append race and
    # no resume ambiguity; `nla/scripts/merge_ladder_shards.sh` concatenates them into the
    # draws.jsonl every scorer already reads, leaving every loader untouched.
    ap.add_argument("--draw-start", type=int, default=0,
                    help="first draw index for this shard (default 0)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-new-gen", type=int, default=MAX_NEW_GEN,
                    help="generation budget. 100%% of unparsed replies sit exactly at the cap, "
                         "so this is the knob that controls censoring.")
    ap.add_argument("--snippets", default=None,
                    help="comma-separated snippet_ids to restrict to (for targeted re-runs)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--host", choices=sorted(HOSTS), default="qwen7b")
    ap.add_argument("--out-dir", default=None,
                    help="defaults to data/nla/p0/p1b/ladder for qwen7b, "
                         "…/ladder_<host> otherwise, so a port cannot overwrite the bank")
    args = ap.parse_args()

    import torch
    from layer_rotation import all_layer_acts
    from steer_run import build_user, graded

    model_id, layer = HOSTS[args.host]
    if args.out_dir is None:
        args.out_dir = str(_PROJ / "data/nla/p0/p1b"
                           / ("ladder" if args.host == "qwen7b" else f"ladder_{args.host}"))
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(SEED)
    out = Path(args.out_dir) / args.tier
    out.mkdir(parents=True, exist_ok=True)
    items = load_tier(args.tier, random.Random(SEED))
    if args.snippets:
        want = {x.strip() for x in args.snippets.split(",") if x.strip()}
        items = [i for i in items if i["snippet_id"] in want]
    if args.limit:
        items = items[:args.limit]
    tokz = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map=args.device).eval()
    # build_call was validated on Qwen's tokenizer. A tier whose calls do not survive a new
    # tokenizer would generate 600 answers to malformed questions — the failure mode that once
    # gave clean-code accuracy 1/16 — so the first item is echoed for eyeballing before the run.
    print(f"[ladder:{args.tier}] host {args.host} ({model_id}, L{layer}) · "
          f"{len(items)} items · {args.draws} draws", flush=True)
    if items:
        print(f"[ladder:{args.tier}] sample call: {items[0]['call']!r} "
              f"truth={items[0]['truth']!r}", flush=True)

    draws_p = out / ("draws.jsonl" if args.draw_start == 0
                     else f"draws_d{args.draw_start}.jsonl")
    done = set()
    if draws_p.exists():
        for l in open(draws_p):
            if l.strip():
                r = json.loads(l)
                done.add((r["draw"], r["snippet_id"]))
    sink = open(draws_p, "a")

    for d in range(args.draw_start, args.draw_start + args.draws):
        for i, it in enumerate(items):
            if (d, it["snippet_id"]) in done:
                continue
            user = build_user(it["code"], it["call"])
            ids = tokz.apply_chat_template([{"role": "user", "content": user}], tokenize=True,
                                           add_generation_prompt=True, return_dict=False)
            with torch.no_grad():
                o = model.generate(torch.tensor([ids], device=model.device),
                                   max_new_tokens=args.max_new_gen, do_sample=False,
                                   pad_token_id=tokz.eos_token_id)
            text = tokz.decode(o[0][len(ids):], skip_special_tokens=True)
            got, ok = graded(text, it["truth"])
            sink.write(json.dumps({"draw": d, "snippet_id": it["snippet_id"], "tier": args.tier,
                                   "correct": bool(ok), "parsed": got is not None,
                                   "answer": got, "reply_chars": len(text),
                                   "n_gen": int(o.shape[1]) - len(ids),
                                   "max_new_gen": args.max_new_gen}) + "\n")
            sink.flush()
        print(f"[ladder:{args.tier}] draw {d} done "
              f"({d - args.draw_start + 1}/{args.draws} in this shard)", flush=True)
    sink.close()

    # Activations once: reads are bit-exact, so repeats buy nothing.
    acts_p = out / "acts.npy"
    if args.draw_start == 0 and not acts_p.exists():
        A = np.stack([all_layer_acts(model, tokz, build_user(it["code"], it["call"]))
                      for it in items])
        np.save(acts_p, A)
        (out / "items.json").write_text(json.dumps([it["snippet_id"] for it in items], indent=2))
        print(f"[ladder:{args.tier}] acts {A.shape} -> {acts_p}", flush=True)

    rows = [json.loads(l) for l in open(draws_p) if l.strip()]
    by_item: dict[str, list[int]] = {}
    for r in rows:
        by_item.setdefault(r["snippet_id"], []).append(int(r["correct"]))
    man = out / ("manifest.json" if args.draw_start == 0
                 else f"manifest_d{args.draw_start}.json")
    man.write_text(json.dumps({
        "tier": args.tier, "host": args.host, "model": model_id, "layer": layer,
        "seed": SEED, "max_new_gen": args.max_new_gen, "draws": args.draws,
        "draw_start": args.draw_start,
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
