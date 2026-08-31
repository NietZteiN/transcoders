"""N7a — build the judge-facing pack, and hold the ground truth on THIS side of the boundary.

BLINDING IS ENFORCED BY DATAFLOW, NOT BY DISCIPLINE. This module is the only one that opens the
graded corpus and the first-error file. It emits two files:

  pack.jsonl   {item_id, read, window}          <- the ONLY file judge_align.py is allowed to open
  keys.jsonl   item_id -> case/role/position/u_rel/condition/correct/tier/...

`ALLOWLIST` is asserted against every emitted row, so a future edit that adds `correct` (or `tier`,
or the answer) to the pack fails loudly instead of silently unblinding the judge.

WHAT BLINDING CAN AND CANNOT DO HERE — stated plainly, because the honest limit matters:
the window is the *model's own reply text*, so a determined judge could in principle infer that a
trace is going wrong and mark it down, without ever seeing a label. That leak cannot be removed
without destroying the data. It is neutralised three ways instead:
  1. HT13's claim is a **within-case interaction** (before-error vs after-error windows come from the
     SAME trace), so any judge bias that applies to a whole trace cancels in the contrast;
  2. the **distant null** (same case, |Δu| > 0.4) catches a judge that is scoring mere topicality;
  3. the **shuffled null** (different case, same tier) catches a judge that is scoring nothing at all.

Conditions emitted per read:
  real      window = +/-1 sentence around the read's own token position
  distant   same case, a window at |Δu| > MIN_DISTANT_U   -> topicality control
  shuffled  a window from a DIFFERENT case, same tier     -> floor control

Runs in `nla-mi` (transformers 5.12.1 — the version that produced the corpus, so
`apply_chat_template` renders byte-identically and `reply_start` lands in the banked space).
**Tokenizer only, no GPU, no model.** Gate: every reconstructed char span must equal the `tok`
string the dense run recorded.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "vendor" / "nla-repo"))

from capture_core import align_reply, load_captures          # noqa: E402
from overnight_capture import build_tasks, build_user, task_key  # noqa: E402

PROJ = _HERE.parent.parent
SEED = 20260724
ALLOWLIST = {"item_id", "read", "window"}      # the judge sees EXACTLY these
MIN_DISTANT_U = 0.4
N_NULL = 500                                   # per null condition
WINDOW_SENTENCES = 1                           # +/- this many sentences
MAX_WINDOW_CHARS = 1200

# Sentence splitting for reasoning text. CoT here is mostly line-structured (numbered steps,
# markdown bullets, code lines), so newlines are the primary boundary; within a line we split on
# terminal punctuation followed by whitespace + an uppercase/digit start. The negative lookbehind
# keeps decimals ("0.5"), ellipses and common abbreviations from splitting mid-number.
_SENT_RE = re.compile(r"(?<![0-9])(?<=[.!?])\s+(?=[A-Z(\[`])")


def split_sentences(text: str) -> list[tuple[int, int]]:
    """Char spans of 'sentences'. Never returns empty for non-empty text."""
    spans: list[tuple[int, int]] = []
    pos = 0
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if stripped:
            base = pos + (len(line) - len(line.lstrip()))
            body = line.strip()
            last = 0
            for m in _SENT_RE.finditer(body):
                spans.append((base + last, base + m.start()))
                last = m.end()
            spans.append((base + last, base + len(body)))
        pos += len(line)
    return [(s, e) for s, e in spans if e > s] or [(0, len(text))]


def window_for(text: str, spans: list[tuple[int, int]], char_pos: int,
               k: int = WINDOW_SENTENCES) -> tuple[str, int]:
    """The +/-k sentence window containing `char_pos`. Returns (text, sentence index)."""
    idx = 0
    for i, (s, e) in enumerate(spans):
        if s <= char_pos < e:
            idx = i
            break
        if s > char_pos:
            idx = max(0, i - 1)
            break
    else:
        idx = len(spans) - 1
    lo, hi = max(0, idx - k), min(len(spans) - 1, idx + k)
    out = text[spans[lo][0]:spans[hi][1]].strip()
    return out[:MAX_WINDOW_CHARS], idx


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dense", default=str(PROJ / "data/nla/n5/2026-08-06/dense_reads.jsonl"))
    ap.add_argument("--captures", default=str(PROJ / "data/nla/overnight/2026-08-04/captures.jsonl"))
    ap.add_argument("--first-errors", default=str(PROJ / "data/nla/n4/2026-08-06/first_errors.jsonl"))
    ap.add_argument("--out-dir", default=str(PROJ / "data/nla/n7" / datetime.now().strftime("%Y-%m-%d")))
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)

    caps = load_captures(args.captures)
    tasks = {task_key(t): t for t in build_tasks()[0]}
    u_err = {}
    for line in open(args.first_errors):
        r = json.loads(line)
        fe = r.get("first_error")
        if fe and fe.get("u_rel") is not None:
            u_err[r["task_key"]] = (float(fe["u_rel"]), fe["detector"])

    from transformers import AutoTokenizer
    tokz = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

    # ---- pass 1: rebuild geometry, verify it, collect every real (read, window) ----
    items, keys = [], []
    by_tier: dict[str, list[int]] = defaultdict(list)     # tier -> item indices (for the shuffled null)
    by_case: dict[str, list[int]] = defaultdict(list)
    n_tok_checked = n_tok_bad = 0

    for line in open(args.dense):
        row = json.loads(line)
        if "error" in row:
            continue
        for side in ("wrong", "control"):
            s = row[side]
            tk = s["task_key"]
            cap, t = caps[tk], tasks[tk]
            reply = cap.get("model_reply") or ""
            a = align_reply(tokz, build_user(t), reply)
            spans = split_sentences(reply)
            base = len(a.templ)
            # controls mirror the WRONG case's relative error; row["wrong"] is the side payload,
            # so the task_key comes from the pair record
            ue = u_err.get(row["pair"]["wrong"])
            for rd in s["reads"]:
                p = rd["position"]
                if p >= len(a.offsets):
                    continue
                cs, ce = a.offsets[p]
                if rd.get("tok") is not None:     # gate: geometry must reproduce the recorded token
                    n_tok_checked += 1
                    n_tok_bad += (a.full[cs:ce] != rd["tok"])
                win, sent_i = window_for(reply, spans, cs - base)
                if not win.strip() or not rd["read"].strip():
                    continue
                i = len(items)
                items.append({"item_id": f"r{i:06d}", "read": rd["read"], "window": win})
                keys.append({"item_id": f"r{i:06d}", "condition": "real", "case": tk,
                             "pair": row["task_key"], "role": side, "position": p,
                             "u_rel": rd.get("u_rel"), "kind": rd["kind"], "rt_cos": rd["rt_cos"],
                             "sent_i": sent_i, "correct": int(side == "control"),
                             "tier": s["tier"] or "slice", "u_err": ue[0] if ue else None,
                             "err_detector": ue[1] if ue else None,
                             "after_error": None if not ue or rd.get("u_rel") is None
                                            else int(rd["u_rel"] >= ue[0])})
                by_tier[s["tier"] or "slice"].append(i)
                by_case[tk].append(i)

    print(f"[N7a] real items: {len(items)}")
    print(f"[N7a] geometry gate: {n_tok_checked - n_tok_bad}/{n_tok_checked} token spans reproduced"
          f"{'  ** MISMATCH **' if n_tok_bad else '  OK'}")
    if n_tok_bad:
        print("[N7a] ABORT — char spans disagree with the dense run; windows would be misplaced.")
        return 1

    # ---- pass 2: the two nulls. Same read text, a window it should NOT match. ----
    def add_null(cond: str, src_i: int, win_i: int) -> None:
        i = len(items)
        items.append({"item_id": f"r{i:06d}", "read": items[src_i]["read"],
                      "window": items[win_i]["window"]})
        k = dict(keys[src_i])
        k.update({"item_id": f"r{i:06d}", "condition": cond, "src_item": items[src_i]["item_id"],
                  "window_from": items[win_i]["item_id"], "after_error": None})
        keys.append(k)

    real_n = len(items)
    # distant: same case, |Δu| > MIN_DISTANT_U -> a judge scoring topicality alone still scores high
    cands = [i for i in range(real_n) if keys[i]["u_rel"] is not None
             and any(abs((keys[j]["u_rel"] or 0) - keys[i]["u_rel"]) > MIN_DISTANT_U
                     for j in by_case[keys[i]["case"]])]
    rng.shuffle(cands)
    n_d = 0
    for i in cands:
        far = [j for j in by_case[keys[i]["case"]]
               if keys[j]["u_rel"] is not None
               and abs(keys[j]["u_rel"] - keys[i]["u_rel"]) > MIN_DISTANT_U]
        if far:
            add_null("distant", i, rng.choice(far))
            n_d += 1
        if n_d >= N_NULL:
            break
    # shuffled: different case, same tier -> the floor
    pool = list(range(real_n))
    rng.shuffle(pool)
    n_s = 0
    for i in pool:
        others = [j for j in by_tier[keys[i]["tier"]] if keys[j]["case"] != keys[i]["case"]]
        if others:
            add_null("shuffled", i, rng.choice(others))
            n_s += 1
        if n_s >= N_NULL:
            break
    print(f"[N7a] nulls: distant {n_d} · shuffled {n_s} · total items {len(items)}")

    # ---- the boundary assertion ----
    for it in items:
        assert set(it) == ALLOWLIST, f"pack row leaks columns: {set(it) - ALLOWLIST}"
    blob = json.dumps(items)
    for banned in ("correct", "expected_output", "tier", "task_key", "ground_truth"):
        assert f'"{banned}"' not in blob, f"pack contains a banned key: {banned}"

    # order is randomised so the judge cannot infer condition from position in the file
    order = list(range(len(items)))
    rng.shuffle(order)
    with open(out / "pack.jsonl", "w") as f:
        for i in order:
            f.write(json.dumps(items[i]) + "\n")
    with open(out / "keys.jsonl", "w") as f:
        for k in keys:
            f.write(json.dumps(k) + "\n")

    (out / "sanitize_manifest.json").write_text(json.dumps({
        "experiment": "n7a_sanitize_windows", "seed": SEED, "argv": sys.argv,
        "n_real": real_n, "n_distant": n_d, "n_shuffled": n_s,
        "allowlist": sorted(ALLOWLIST), "min_distant_u": MIN_DISTANT_U,
        "window_sentences": WINDOW_SENTENCES, "geometry_gate": f"{n_tok_checked - n_tok_bad}/{n_tok_checked}",
        "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=1))
    print(f"[N7a] wrote {out}/pack.jsonl ({len(items)} items) + keys.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
