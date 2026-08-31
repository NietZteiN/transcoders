"""N13 Stage 1 — measure whether the model is TORN, not whether it is wrong.

WHY CORRECTNESS IS THE WRONG LABEL. HT12 asked whether round-trip faithfulness predicts
correctness and refuted it. But `correct = False` merges two opposite internal states:

    confidently wrong   one reading, firmly held, mistaken
    torn                two incompatible readings, unresolved   <- the target

A person reading "a plus b squared" is torn between `a + b^2` and `(a+b)^2`. A person who
confidently computes the wrong one is not torn at all. Both score zero. If any internal signal
tracks torn-ness, correctness is precisely the label that hides it.

**Nothing in this programme can currently see torn-ness, because every capture is a single
greedy run.** One sample cannot distinguish a model that would answer identically ten times from
one that would answer three different ways. This module supplies the missing axis: sample K
answers per item at temperature > 0 and measure the disagreement.

    n_distinct      how many different answers came back
    entropy         normalized Shannon entropy over the answer distribution (0 = certain)
    modal_share     fraction of samples agreeing with the plurality answer
    modal_correct   is the plurality answer right

giving the 2x2 that correctness alone cannot produce.

REUSE, NOT RE-DERIVATION. The prompt, the task list, and the grader are imported verbatim from
`overnight_capture` — the same functions that produced the banked answers. That is what makes
the greedy sanity check meaningful: regenerate a handful at temperature 0 and they must reproduce
the banked `model_answer`. Re-deriving prompt construction cost four bugs in the B4 runner, one
of which asked a program about a function it did not define.

Batching is safe here: these are subject-model generations, not AV reads. (The read path must
stay sequential — batching was measured to break AV determinism, 6.3x faster but only 15%
byte-identical.) Batched sampling uses LEFT padding; never copy that setting into a capture
script, where positions must line up with token indices.

Env `nla-mi`, one GPU.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_NLA_ROOT = _HERE.parent
_PROJ = _NLA_ROOT.parent
sys.path.insert(0, str(_NLA_ROOT / "vendor" / "nla-repo"))
sys.path.insert(0, str(_HERE))

from overnight_capture import (  # noqa: E402
    MAX_NEW_GEN, SEED, TARGET_MODEL, build_tasks, build_user, grade, task_key, truth_of,
)

BANKED = _PROJ / "data/nla/overnight/2026-08-04/captures.jsonl"


def _norm(s) -> str:
    """The harness's answer normalization, so 'distinct' means what grading means."""
    return re.sub(r"[\s'\"`]", "", str(s)).lower()


def entropy(counts: list[int]) -> float:
    """Shannon entropy normalized to [0,1] by log(K).

    Normalizing by log(K) rather than log(n_distinct) keeps items comparable: an item split 4
    ways is more torn than one split 2 ways, and dividing by the observed number of outcomes
    would erase exactly that difference.
    """
    n = sum(counts)
    if n <= 1:
        return 0.0
    h = -sum((c / n) * math.log(c / n) for c in counts if c)
    return h / math.log(n) if n > 1 else 0.0


def summarize(answers: list[str | None], t: dict) -> dict:
    """Per-item torn-ness from K sampled answers. `None` (unparsed) is its own outcome."""
    keys = [_norm(a) if a is not None else "<UNPARSED>" for a in answers]
    c = Counter(keys)
    modal, modal_n = c.most_common(1)[0]
    truth = _norm(truth_of(t))
    return {
        "k": len(answers),
        "n_distinct": len(c),
        "entropy": round(entropy(list(c.values())), 4),
        "modal_answer": modal,
        "modal_share": round(modal_n / len(answers), 4),
        "modal_correct": bool(modal == truth),
        "any_correct": bool(truth in c),
        "frac_correct": round(c.get(truth, 0) / len(answers), 4),
        "n_unparsed": c.get("<UNPARSED>", 0),
        "answer_counts": dict(c),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--items-per-batch", type=int, default=4,
                    help="items x k sequences are generated together; 4x8=32 at 1100 tokens")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--greedy-check", type=int, default=8,
                    help="items to regenerate at temperature 0 and compare with the banked "
                         "answer — the control-condition tripwire")
    ap.add_argument("--out", default=str(_PROJ / "data/nla/n13/answer_entropy.jsonl"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-hours", type=float, default=6.0)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from capture_core import JsonlSink, WallGuard

    torch.manual_seed(SEED)

    banked = {}
    for line in open(BANKED):
        r = json.loads(line)
        if "error" not in r and r.get("correct") is not None:
            banked[r["task_key"]] = r
    tasks, _ = build_tasks()
    graded = [t for t in tasks if task_key(t) in banked]
    if args.limit:
        graded = graded[: args.limit]
    print(f"[n13] {len(graded)} graded tasks rejoined to banked captures", flush=True)

    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    sink = JsonlSink(out_p, key_field="task_key")
    done = sink.done_keys()
    todo = [t for t in graded if task_key(t) not in done]
    print(f"[n13] {len(todo)} to go ({len(done)} done)", flush=True)

    tok = AutoTokenizer.from_pretrained(TARGET_MODEL)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"          # batched generation only — see module docstring
    model = AutoModelForCausalLM.from_pretrained(
        TARGET_MODEL, dtype=torch.bfloat16, device_map=args.device).eval()
    wall = WallGuard(args.max_hours)

    # ── the tripwire: greedy must reproduce the banked answer ─────────────
    if args.greedy_check and todo:
        print(f"[n13] greedy sanity on {args.greedy_check} items …", flush=True)
        agree = checked = 0
        for t in graded[: args.greedy_check]:
            prompt = tok.apply_chat_template([{"role": "user", "content": build_user(t)}],
                                             tokenize=False, add_generation_prompt=True)
            enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
            with torch.no_grad():
                o = model.generate(**enc, max_new_tokens=MAX_NEW_GEN, do_sample=False,
                                   pad_token_id=tok.eos_token_id)
            rep = tok.decode(o[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
            ans, _ = grade(t, rep)
            want = banked[task_key(t)]["model_answer"]
            checked += 1
            agree += int(_norm(ans) == _norm(want))
        print(f"[n13] greedy reproduces banked answer on {agree}/{checked}", flush=True)
        if agree < checked * 0.75:
            print("[n13] ABORT: greedy regeneration disagrees with the banked corpus — the "
                  "prompt or grader has drifted, so entropy would not be comparable.",
                  file=sys.stderr)
            return 2

    # ── K-sampling ────────────────────────────────────────────────────────
    t0, n_done = time.time(), 0
    for i in range(0, len(todo), args.items_per_batch):
        if wall.expired():
            print("[n13] wall budget reached — stopping cleanly", flush=True)
            break
        chunk = todo[i: i + args.items_per_batch]
        prompts = [tok.apply_chat_template([{"role": "user", "content": build_user(t)}],
                                           tokenize=False, add_generation_prompt=True)
                   for t in chunk]
        enc = tok(prompts, return_tensors="pt", padding=True,
                  add_special_tokens=False).to(model.device)
        try:
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=MAX_NEW_GEN, do_sample=True,
                                     temperature=args.temperature, top_p=args.top_p,
                                     num_return_sequences=args.k,
                                     pad_token_id=tok.eos_token_id)
            plen = enc["input_ids"].shape[1]
            texts = tok.batch_decode(out[:, plen:], skip_special_tokens=True)
        except Exception as e:
            for t in chunk:
                sink.append({"task_key": task_key(t), "error": repr(e)[:200]})
            print(f"[n13] ERROR on batch {i}: {e!r}", flush=True)
            continue

        # generate() returns k consecutive sequences per input, in input order
        for j, t in enumerate(chunk):
            reps = texts[j * args.k: (j + 1) * args.k]
            answers = [grade(t, rep)[0] for rep in reps]
            b = banked[task_key(t)]
            sink.append({
                "task_key": task_key(t), "kind": t["kind"], "dataset": t.get("dataset"),
                "snippet_id": t.get("snippet_id"), "tier": t.get("tier"),
                "language": t.get("language"), "truth": truth_of(t),
                "banked_greedy_answer": b["model_answer"], "banked_correct": b["correct"],
                "banked_reply_tokens": b.get("reply_tokens"),
                "temperature": args.temperature, "top_p": args.top_p,
                "reply_chars": [len(r) for r in reps],
                **summarize(answers, t),
            })
        n_done += len(chunk)
        rate = n_done / (time.time() - t0)
        print(f"[n13] {n_done}/{len(todo)} items · {rate*60:.1f}/min · "
              f"ETA {(len(todo)-n_done)/max(rate,1e-9)/60:.0f} min", flush=True)

    sink.close()
    rows = [json.loads(l) for l in open(out_p) if l.strip()]
    ok = [r for r in rows if "error" not in r]
    ent = [r["entropy"] for r in ok]
    (out_p.parent / "answer_entropy_manifest.json").write_text(json.dumps({
        "experiment": "n13_stage1_answer_entropy", "seed": SEED, "argv": sys.argv,
        "model": TARGET_MODEL, "k": args.k, "temperature": args.temperature,
        "top_p": args.top_p, "max_new_gen": MAX_NEW_GEN,
        "n_items": len(ok), "n_errors": len(rows) - len(ok),
        "frac_zero_entropy": round(sum(e == 0 for e in ent) / max(len(ent), 1), 4),
        "elapsed_hours": round(wall.elapsed_h(), 3),
        "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2))
    print(f"[n13] done · {len(ok)} items · "
          f"{sum(e == 0 for e in ent)}/{len(ent)} perfectly consistent", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
