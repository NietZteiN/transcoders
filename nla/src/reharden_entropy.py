"""N13 robustness — recompute per-item answer entropy under a hardened normalizer.

WHY. Stage 1 graded sampled answers with the harness normalizer
(`re.sub(r"[\s'\"`]", "", s).lower()`), which is exactly right for reproducing the banked greedy
labels but was never designed to count DISTINCT answers. Two failure modes inflate entropy:

  markdown emphasis   "**[]" and "[]" are the same answer; 36.4% of items contain a "*" somewhere
  dict key order      "{1:none,3:none,2:none}" and "{1:none,2:none,3:none}" are the same mapping

Both manufacture spurious distinct answers, which inflates entropy — the OUTCOME variable of the
whole experiment. Noise in an outcome attenuates real associations, so a null result measured on
a noisy label could be an artifact of the label rather than a fact about the model. That makes
this re-analysis mandatory before any conclusion is reported, not an optional extra.

The banked Stage-1 output is left untouched as the record; this writes a parallel file so both
analyses can be run and compared.

Env `nla-mi` or `transcoders-mi`. CPU, seconds.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

_PROJ = Path(__file__).resolve().parent.parent.parent


def harden(s) -> str:
    """Harness normalization plus the two equivalences it misses."""
    s = str(s)
    s = re.sub(r"\*+", "", s)                                    # markdown emphasis
    s = re.sub(r"^\s*(the\s+)?(output|result|answer)\s*(is)?\s*:?\s*", "", s, flags=re.I)
    s = re.sub(r"[\s'\"`]", "", s).lower()
    # canonicalize mapping literals: {k:v,...} is order-independent, lists are NOT
    if len(s) > 1 and s[0] == "{" and s[-1] == "}" and ":" in s:
        s = "{" + ",".join(sorted(s[1:-1].split(","))) + "}"
    return s


def entropy(counts: list[int]) -> float:
    n = sum(counts)
    if n <= 1:
        return 0.0
    h = -sum((c / n) * math.log(c / n) for c in counts if c)
    return h / math.log(n)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default=str(_PROJ / "data/nla/n13/answer_entropy.jsonl"))
    ap.add_argument("--out", default=str(_PROJ / "data/nla/n13/answer_entropy_hardened.jsonl"))
    args = ap.parse_args()

    n_changed = n_flip = 0
    rows = []
    for line in open(args.inp):
        r = json.loads(line)
        if "error" in r:
            rows.append(r)
            continue
        c: Counter = Counter()
        for ans, k in r["answer_counts"].items():
            c[harden(ans)] += k
        modal, modal_n = c.most_common(1)[0]
        truth = harden(r["truth"])
        e = round(entropy(list(c.values())), 4)
        was_correct = r["modal_correct"]
        n_changed += int(e != r["entropy"])
        n_flip += int((modal == truth) != was_correct)
        r.update({
            "entropy": e, "n_distinct": len(c),
            "modal_answer": modal, "modal_share": round(modal_n / r["k"], 4),
            "modal_correct": bool(modal == truth),
            "any_correct": bool(truth in c),
            "frac_correct": round(c.get(truth, 0) / r["k"], 4),
            "answer_counts": dict(c),
            "normalization": "hardened",
        })
        rows.append(r)

    with open(args.out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"[reharden] {len(rows)} rows -> {args.out}")
    print(f"[reharden] entropy changed on {n_changed} items; correctness flipped on {n_flip}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
