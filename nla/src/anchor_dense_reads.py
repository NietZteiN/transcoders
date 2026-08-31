"""Give the N5 dense reads the same char-span anchors the banked reads carry.

The dense corpus records token `position` and `u_rel` but no character offsets, because the capture
runner never needed them. The results browser does: it highlights each read's exact token inside the
transcript. Rather than let the page recompute geometry (a second copy of the alignment arithmetic is
exactly what `capture_core.align_reply` exists to prevent), this recomputes it once, verifies it
against the token strings the dense run recorded, and writes anchored reads to disk.

Also classifies each read as reasoning vs answer-line, so the browser labels dense reads the same way
it labels banked ones: the answer line is located in the reply text ("Output:" / "Lines:") and any
read whose span starts at or after it becomes `ans@`.

Runs in `nla-mi` (transformers 5.12.1 — the version that produced the corpus). Tokenizer only, no GPU.
Gate: every reconstructed span must equal the `tok` string the dense run recorded.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "vendor" / "nla-repo"))

from capture_core import align_reply, load_captures                  # noqa: E402
from overnight_capture import build_tasks, build_user, task_key      # noqa: E402

PROJ = _HERE.parent.parent
_ANSWER_LINE = re.compile(r"^\s*(Output|Lines)\s*:", re.M)
CTX = 40                      # chars of context either side, same as the banked capture
_WORDCH = re.compile(r"[A-Za-z0-9_]")


def _word_at(full: str, s: int, e: int) -> str:
    """The whole identifier a sub-word token belongs to ('' when it is not word-like).

    Banked reads carry this so a card can say "piece of `_lastNSecs`"; without it a dense read of a
    fragment reads as gibberish with no clue what it belongs to.
    """
    i, j = s, e
    # Trim to the word characters INSIDE the token first. A token like " by" starts on a space;
    # expanding from that index walks left into the previous word and reports "step by".
    while i < j and not _WORDCH.match(full[i]):
        i += 1
    while j > i and not _WORDCH.match(full[j - 1]):
        j -= 1
    if i >= j:
        return ""
    inner = full[i:j]                       # the token's own word characters
    while i > 0 and _WORDCH.match(full[i - 1]):
        i -= 1
    while j < len(full) and _WORDCH.match(full[j]):
        j += 1
    word = full[i:j]
    # Only claim a parent when the token is a STRICT sub-word. Otherwise ".push" would render as
    # "piece of `push`" and " by" as "piece of `by`" -- true but useless, and it buries the ~700
    # cases where the badge means something ("tNSec" -> "_lastNSecs").
    return word if word != inner else ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dense", default=str(PROJ / "data/nla/n5/2026-08-06/dense_reads.jsonl"))
    ap.add_argument("--captures", default=str(PROJ / "data/nla/overnight/2026-08-04/captures.jsonl"))
    ap.add_argument("--out", default=str(PROJ / "data/nla/n5/2026-08-06/dense_reads_anchored.json"))
    args = ap.parse_args()

    caps = load_captures(args.captures)
    tasks = {task_key(t): t for t in build_tasks()[0]}

    from transformers import AutoTokenizer
    tokz = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

    out: dict[str, list[dict]] = defaultdict(list)
    n_ok = n_bad = n_skip = 0
    for line in open(args.dense):
        row = json.loads(line)
        if "error" in row:
            continue
        for side in ("wrong", "control"):
            s = row[side]
            tk = s["task_key"]
            if tk in out:                     # a control can appear in more than one pair
                continue
            cap, t = caps[tk], tasks[tk]
            reply = cap.get("model_reply") or ""
            a = align_reply(tokz, build_user(t), reply)
            base = len(a.templ)
            m = _ANSWER_LINE.search(reply)
            ans_at = (base + m.start()) if m else None
            for rd in s["reads"]:
                p = rd["position"]
                if p >= len(a.offsets):
                    n_skip += 1
                    continue
                cs, ce = a.offsets[p]
                if rd.get("tok") is not None:
                    if a.full[cs:ce] == rd["tok"]:
                        n_ok += 1
                    else:
                        n_bad += 1
                        continue              # never emit a span we cannot verify
                where = ("ans@%d" % p) if (ans_at is not None and cs >= ans_at) else ("cot@%d" % p)
                # Match the banked anchor schema exactly (s/e/rs/re/tok/word/before/after), or the
                # browser silently drops the token-context strip that is the whole point of a card.
                tok = a.full[cs:ce]
                out[tk].append({
                    "position": p, "read": rd["read"], "rt_cos": rd["rt_cos"],
                    "u_rel": rd.get("u_rel"), "role": rd["kind"], "where": where,
                    "src": "reused" if rd.get("reused") else "dense",
                    "anchor": {"s": cs, "e": ce, "rs": cs - base, "re": ce - base,
                               "in_reply": cs >= base, "tok": tok, "word": _word_at(a.full, cs, ce),
                               "before": a.full[max(0, cs - CTX):cs],
                               "after": a.full[ce:ce + CTX]},
                })
    for tk in out:
        out[tk].sort(key=lambda r: r["position"])

    print(f"[anchor] cases {len(out)} · reads {sum(len(v) for v in out.values())}")
    print(f"[anchor] token gate: {n_ok} verified, {n_bad} mismatched, {n_skip} out of range"
          f"{'  ** MISMATCH **' if n_bad else '  OK'}")
    if n_bad:
        print("[anchor] ABORT — spans disagree with the dense run")
        return 1
    Path(args.out).write_text(json.dumps({
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "n_cases": len(out), "n_reads": sum(len(v) for v in out.values()),
        "token_gate": f"{n_ok}/{n_ok + n_bad}", "reads": out}, separators=(",", ":")))
    print(f"[anchor] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
