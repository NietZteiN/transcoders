"""N5 — pair each wrong case with a correct control.

Matching cascade (greedy, seeded, without replacement):
  1. same snippet_id, same tier, correct                 — the ideal: same problem, same obfuscation
  2. same snippet_id, nearest tier on the ladder         — same problem, different obfuscation
  3. same (dataset, tier, language), closest length      — stratum match

HARD CONSTRAINT: reject any pair with |log(len_wrong / len_correct)| > log(1.5).
Wrong traces are 1.6× longer than correct ones at the median (686 vs 418 tokens), so length is
itself a correctness signal. An unmatched-length control would confound every downstream
comparison — better to leave a wrong case unpaired and say so than to pair it badly.
"""
from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

SEED = 20260724
TIER_LADDER = ["L0", "L1", "L1b", "L2", "L3"]
MAX_LOG_RATIO = math.log(1.5)


def _tier_dist(a: str | None, b: str | None) -> int:
    if a == b:
        return 0
    if a in TIER_LADDER and b in TIER_LADDER:
        return abs(TIER_LADDER.index(a) - TIER_LADDER.index(b))
    return 99


def build_pairs(cases: list[dict]) -> tuple[list[dict], dict]:
    """Return (pairs, report). Each pair: {wrong, control, method, len_ratio}."""
    rng = random.Random(SEED)
    wrong = [c for c in cases if c.get("correct") is False and not c.get("truncated")]
    correct = [c for c in cases if c.get("correct") is True]
    wrong.sort(key=lambda c: c["task_key"])          # deterministic order
    used: set[str] = set()
    pairs, unpaired = [], []

    def ok_len(w, c) -> bool:
        lw, lc = max(w.get("reply_tokens") or 1, 1), max(c.get("reply_tokens") or 1, 1)
        return abs(math.log(lw / lc)) <= MAX_LOG_RATIO

    for w in wrong:
        cands = [c for c in correct if c["task_key"] not in used and ok_len(w, c)]
        pick, method = None, None
        # 1. same problem + same tier
        for c in cands:
            if c["snippet_id"] == w["snippet_id"] and c["tier"] == w["tier"]:
                pick, method = c, "same_problem_same_tier"
                break
        # 2. same problem, nearest tier
        if pick is None:
            same = [c for c in cands if c["snippet_id"] == w["snippet_id"]]
            if same:
                same.sort(key=lambda c: (_tier_dist(c["tier"], w["tier"]), c["task_key"]))
                pick, method = same[0], "same_problem_other_tier"
        # 3. stratum match on closest length
        if pick is None:
            strat = [c for c in cands if c["dataset"] == w["dataset"]
                     and c["tier"] == w["tier"] and c["language"] == w["language"]]
            if not strat:
                strat = [c for c in cands if c["language"] == w["language"] and c["tier"] == w["tier"]]
            if strat:
                lw = max(w.get("reply_tokens") or 1, 1)
                strat.sort(key=lambda c: (abs(math.log(max(c.get("reply_tokens") or 1, 1) / lw)),
                                          c["task_key"]))
                pick, method = strat[0], "stratum_length"
        if pick is None:
            unpaired.append(w["task_key"])
            continue
        used.add(pick["task_key"])
        lr = (max(w.get("reply_tokens") or 1, 1)) / max(pick.get("reply_tokens") or 1, 1)
        pairs.append({"wrong": w["task_key"], "control": pick["task_key"], "method": method,
                      "tier": w["tier"], "dataset": w["dataset"], "language": w["language"],
                      "len_ratio": round(lr, 3)})

    from collections import Counter
    ratios = sorted(p["len_ratio"] for p in pairs)
    report = {
        "n_wrong": len(wrong), "n_paired": len(pairs), "n_unpaired": len(unpaired),
        "unpaired": unpaired,
        "by_method": dict(Counter(p["method"] for p in pairs)),
        "by_tier": dict(Counter(p["tier"] or "slice" for p in pairs)),
        "len_ratio": {"median": ratios[len(ratios) // 2] if ratios else None,
                      "min": ratios[0] if ratios else None,
                      "max": ratios[-1] if ratios else None},
        "seed": SEED, "max_log_ratio": round(MAX_LOG_RATIO, 4),
    }
    return pairs, report


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    PROJ = _HERE.parent.parent
    ap.add_argument("--src", default=str(PROJ / "data/nla/overnight/2026-08-04/enriched.json"))
    ap.add_argument("--out", default=str(PROJ / "data/nla/n5/pairs.json"))
    args = ap.parse_args()
    cases = json.load(open(args.src))
    pairs, report = build_pairs(cases)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"pairs": pairs, "report": report}, indent=1))
    print(json.dumps(report, indent=1))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
