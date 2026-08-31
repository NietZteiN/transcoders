"""N4-D3 — judge fallback for traces no value-checker can localize.

⚠️ THIS MODULE SEES GROUND TRUTH. It is deliberately separate from N7's alignment judge
(`judge_align.py`), which must stay answer-blind. Separate module, separate process, separate
output file. Never import one from the other; never merge their output.

Two jobs:
  1. localize   — for the 76 cases D0/D2 could not localize (mostly JS locals, and the L2/L3
                  dispatcher mis-decodes where the model computes a different program correctly)
  2. validate   — re-localize the cases D2-state already claimed, so we can measure agreement.
                  D2-state runs at confidence 0.60 on occurrence-index alignment and several of
                  its hits look like alignment artifacts; this is the check that decides whether
                  they may target N5's bursts.

Sampling: temperature 0.7 × 3 seeds, take the MEDIAN sentence index. A single temp-0 pick gives
no uncertainty estimate; the spread across seeds is the uncertainty. Spread ≤±1 → conf 0.60,
else 0.35.

Run (nla-mi env, judge server on its own GPU):
    python -m src.first_error_judge --judge-url http://localhost:30010 \
        --n4 data/nla/n4/2026-08-06/first_errors.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import urllib.request
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

PROJ = _HERE.parent.parent
SEED = 20260724
SENT_SPLIT = re.compile(r"(?<=[.!?:])\s+|\n+")

SYSTEM = (
    "You are analysing a chain of thought that a language model wrote while solving a code "
    "comprehension problem. The final answer it reached was WRONG. You are given the code, the "
    "correct answer, and the model's numbered reasoning sentences.\n\n"
    "Identify the FIRST sentence at which the reasoning goes wrong — the earliest point where it "
    "states, assumes or derives something that is not true of the code. Earlier sentences that are "
    "merely restating the problem are not errors. A sentence that correctly derives a consequence "
    "of an earlier mistake is NOT the first error; the earlier mistake is.\n\n"
    "Reply with a single JSON object and nothing else."
)

SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["sentence_index", "claimed", "truth", "kind", "confidence"],
    "properties": {
        "sentence_index": {"type": "integer", "minimum": 1},
        "claimed": {"type": "string", "maxLength": 120},
        "truth": {"type": "string", "maxLength": 120},
        "kind": {"type": "string", "enum": ["arithmetic", "state", "control_flow",
                                            "misread_code", "assumption", "none"]},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
    },
}


def sentences(reply: str) -> list[str]:
    return [s.strip() for s in SENT_SPLIT.split(reply) if s.strip()]


def build_prompt(case: dict) -> list[dict]:
    sents = sentences(case["model_reply"])
    numbered = "\n".join(f"{i+1}. {s}" for i, s in enumerate(sents[:120]))
    user = (
        f"[CODE]\n{case.get('code','')}\n\n"
        f"[QUESTION] What is the output of `{case.get('call') or case.get('target')}`?\n"
        f"[CORRECT ANSWER] {case.get('truth')}\n"
        f"[MODEL'S WRONG ANSWER] {case.get('model_answer')}\n\n"
        f"[MODEL'S REASONING, numbered]\n{numbered}\n\n"
        'Reply exactly: {"sentence_index": <int>, "claimed": "<what it said>", '
        '"truth": "<what is actually true>", "kind": "<arithmetic|state|control_flow|'
        'misread_code|assumption|none>", "confidence": "<high|medium|low>"}'
    )
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


_JSON_OBJ = re.compile(r"\{[^{}]*\}", re.S)


def parse_verdict(txt: str) -> dict | None:
    """Lenient JSON extraction. We do NOT use sglang's constrained `response_format`:
    the grammar backend SEGFAULTED the scheduler (exit -11) on this schema, taking the whole
    server down mid-run. Prompt for JSON, then parse the first object out of the reply and
    validate the fields ourselves — a malformed reply costs one sample, not the server."""
    if not txt:
        return None
    for m in _JSON_OBJ.finditer(txt):
        try:
            d = json.loads(m.group(0))
        except Exception:
            continue
        idx = d.get("sentence_index")
        if isinstance(idx, str) and idx.strip().lstrip("-").isdigit():
            idx = int(idx)
        if isinstance(idx, int) and idx >= 1:
            d["sentence_index"] = idx
            if d.get("kind") not in {"arithmetic", "state", "control_flow",
                                     "misread_code", "assumption", "none"}:
                d["kind"] = "assumption"
            return d
    return None


def judge_once(url: str, messages: list[dict], seed: int, temperature: float = 0.7) -> dict | None:
    body = json.dumps({
        "model": "judge", "messages": messages, "temperature": temperature, "seed": seed,
        "max_tokens": 220,
    }).encode()
    req = urllib.request.Request(f"{url}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            txt = json.loads(r.read())["choices"][0]["message"]["content"]
        return parse_verdict(txt)
    except Exception:
        return None


def char_of_sentence(reply: str, idx: int) -> int:
    sents = sentences(reply)
    if not (1 <= idx <= len(sents)):
        return int(0.5 * len(reply))
    pos, cur = 0, 0
    for i, s in enumerate(sents):
        j = reply.find(s, cur)
        if j < 0:
            j = cur
        if i == idx - 1:
            return j
        cur = j + len(s)
    return pos


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge-url", default="http://localhost:30010")
    ap.add_argument("--n4", default=str(PROJ / "data/nla/n4/2026-08-06/first_errors.jsonl"))
    ap.add_argument("--src", default=str(PROJ / "data/nla/overnight/2026-08-04/enriched.json"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    n4 = [json.loads(l) for l in open(args.n4)]
    cases = {c["task_key"]: c for c in json.load(open(args.src))}
    out_path = Path(args.out or (Path(args.n4).parent / "first_error_judge.jsonl"))

    # every non-slice case: unlocalized ones need D3; D2-state ones need validation
    todo = [r for r in n4 if r["kind"] != "slice_prediction"
            and (r["first_error"] is None or r["first_error"]["detector"] == "D2-state")]
    if args.limit:
        todo = todo[: args.limit]
    print(f"[D3] {len(todo)} cases "
          f"({sum(1 for r in todo if r['first_error'] is None)} unlocalized, "
          f"{sum(1 for r in todo if r['first_error'])} D2-state validations) × {args.seeds} seeds")

    n_ok = 0
    with open(out_path, "w") as fh:
        for i, r in enumerate(todo):
            case = cases[r["task_key"]]
            msgs = build_prompt(case)
            picks = []
            for s in range(args.seeds):
                v = judge_once(args.judge_url, msgs, seed=SEED + s)
                if v and isinstance(v.get("sentence_index"), int):
                    picks.append(v)
            if not picks:
                fh.write(json.dumps({"task_key": r["task_key"], "error": "no valid verdict"}) + "\n")
                continue
            idxs = [p["sentence_index"] for p in picks]
            med = int(statistics.median(idxs))
            spread = max(idxs) - min(idxs)
            chosen = min(picks, key=lambda p: abs(p["sentence_index"] - med))
            reply = case["model_reply"]
            pos = char_of_sentence(reply, med)
            n_ok += 1
            fh.write(json.dumps({
                "task_key": r["task_key"], "tier": r["tier"], "language": r["language"],
                "role": "validate" if r["first_error"] else "localize",
                "d3": {"position_char": pos, "u_rel": pos / max(len(reply), 1),
                       "sentence_index": med, "spread": spread,
                       "kind": chosen.get("kind"), "claimed": chosen.get("claimed"),
                       "truth": chosen.get("truth"),
                       "detector": "D3", "confidence": 0.60 if spread <= 1 else 0.35},
                "seeds": idxs,
                "d2_state": r["first_error"],
                "seed": SEED,
            }) + "\n")
            fh.flush()
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(todo)} · {n_ok} verdicts", flush=True)
    print(f"[D3] wrote {out_path} ({n_ok}/{len(todo)} verdicts)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
