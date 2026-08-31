"""Pick the snippet set for the coupling follow-up, from the cheap behavioural screen.

N11's coupling cell pointed the right way (3/3 items where the model *stated* the injected
algorithm carried a recurrent internal read, vs 8.1% where it did not) but rested on three
items, because the model is behaviourally deceived only ~4% of the time. Read budget is ~84 s
per item; a stated answer is one batched generation. So the screen finds the rare deceived
items and this picks what to spend reads on.

**Matched controls are the point.** Deceived items are not a random sample — a snippet the
model gets wrong may be longer, weirder, or assigned an algorithm whose vocabulary fires more
easily. Each `stated_hit` item is therefore paired with a non-hit item that shares the
**injected algorithm** (so the target label's base rate is held fixed) and is nearest in **code
length**. Without that, "deceived items have more capability reads" could just mean "deceived
items are longer".

Selecting on the behavioural outcome is legitimate here because the read rate is reported
*within* stratum (stated vs not-stated). It does not license an unconditional deception rate,
so the screen's own marginal rate is carried through to the output for the write-up.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
PROJ = _HERE.parent.parent
OBF_ROOT = Path("/data/jvl210002/my_downloads/allocation_replication/data/obf")


def code_len(dataset: str, snippet: str) -> int:
    p = next((OBF_ROOT / dataset / snippet / "adversarial_rename").glob("*.java"), None)
    return p.stat().st_size if p else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--screen", default=str(PROJ / "data/nla/n11/screen.jsonl"))
    ap.add_argument("--out", default=str(PROJ / "data/nla/n11/coupling_snippets.txt"))
    ap.add_argument("--report", default=str(PROJ / "data/nla/n11/coupling_selection.json"))
    ap.add_argument("--max-controls-per-hit", type=int, default=1)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.screen) if l.strip()]
    c2 = [r for r in rows if r["condition"] == "C2"]
    parsed = [r for r in c2 if r["parsed_answer"]]
    hits = [r for r in parsed if r["stated_hit"]]
    non = [r for r in parsed if not r["stated_hit"]]

    by_algo = defaultdict(list)
    for r in non:
        by_algo[r["injected_algorithm"]].append(r)
    for v in by_algo.values():
        v.sort(key=lambda r: code_len(r["dataset"], r["snippet"]))

    chosen_ctrl, used = [], set()
    matched_hits = set()
    for h in hits:
        L = code_len(h["dataset"], h["snippet"])
        pool = [r for r in by_algo.get(h["injected_algorithm"], [])
                if (r["dataset"], r["snippet"]) not in used]
        pool.sort(key=lambda r: abs(code_len(r["dataset"], r["snippet"]) - L))
        for r in pool[: args.max_controls_per_hit]:
            used.add((r["dataset"], r["snippet"]))
            chosen_ctrl.append(r)
            matched_hits.add((h["dataset"], h["snippet"]))

    selected = hits + chosen_ctrl
    lines = sorted({f"{r['dataset']}/{r['snippet']}" for r in selected})
    Path(args.out).write_text("\n".join(lines) + "\n")

    report = {
        "screen_rows": len(rows),
        "c2_total": len(c2), "c2_parsed": len(parsed),
        "marginal_stated_hit_rate_parsed": round(len(hits) / len(parsed), 4) if parsed else None,
        "n_stated_hit": len(hits),
        "n_matched_controls": len(chosen_ctrl),
        "n_selected_snippets": len(lines),
        "hits_by_algorithm": {k: sum(1 for h in hits if h["injected_algorithm"] == k)
                              for k in sorted({h["injected_algorithm"] for h in hits})},
        # Per-hit, not per-algorithm: with two hits sharing an algorithm and only one control
        # available, the per-algorithm test reported both as matched.
        "unmatched_hits": [f"{h['dataset']}/{h['snippet']}" for h in hits
                           if (h["dataset"], h["snippet"]) not in matched_hits],
        "note": ("Read rate must be reported WITHIN stratum (stated vs not-stated); this "
                 "selection is stratified on the behavioural outcome and does not support an "
                 "unconditional deception-rate claim."),
    }
    Path(args.report).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\nwrote {args.out} ({len(lines)} snippets)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
