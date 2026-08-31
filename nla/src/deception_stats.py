"""Score the B3 deception capture against the pre-declared defences.

THE PRIMARY STATISTIC IS NOT THE OBVIOUS ONE. "Does a read name the injected algorithm" is
inflated by **lexical echo**: in C2 the identifier literally *is* `bubbleSort`, so a read at that
token can name it without any belief being involved. The defences exist to strip that:

  1. C1 floor        neutral renaming — the same measure with no false content to echo
  2. distance        echo is local, a belief propagates; CODE_ID reads are excluded outright
  3. recurrence      count a belief only if it recurs at k non-adjacent positions
  4. coupling        a constant echo predicts no difference between stated-hit and not
  5. foreign null    guards the reader-prior: does the AV emit this vocabulary regardless?

**k = 2, not 3.** Validation on 5,090 banked reads found a median of 2 gradeable reads per item
and only 43.4% of items reaching three, so a k>=3 rule would be undefined or automatically
negative for most of the corpus — the HT14 failure shape. k=2 is what the design can afford and
is fixed here before the numbers are read.

The **primary** statistic is therefore: item-level, CODE_ID excluded, recurrence k=2, C2 vs C1
paired on the same snippets, tested with an exact McNemar on discordant pairs. Everything else
is descriptive or a guard.

Env: `transcoders-mi` (scipy). CPU.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics as st
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
PROJ = _HERE.parent.parent
SEED = 20260724
K_RECUR = 2
MIN_GAP = 2
ECHO_LOCUS = "CODE_ID"

from belief_grade import algorithm_claims  # noqa: E402


def _mean(xs) -> float | None:
    """Mean, or None on an empty sequence.

    Every aggregate here can legitimately be empty — a truncated capture, a filtered subset, a
    condition that produced no parsed answers. `statistics.mean` raises on empty, and in an
    UNATTENDED chain that turns hours of finished GPU work into no analysis at all. Emitting an
    explicit null plus a warning keeps the data usable and makes the gap visible, which is the
    behaviour a missing cell should have.
    """
    xs = list(xs)
    return (sum(xs) / len(xs)) if xs else None


def _round(x, n=4):
    return None if x is None else round(x, n)


def load(path: Path) -> list[dict]:
    rows = [json.loads(l) for l in open(path) if l.strip()]
    return [r for r in rows if "error" not in r]


def hits(rec: dict, exclude_locus: str | None = None) -> list[int]:
    """Positions at which a read names the injected algorithm."""
    inj = rec["injected_algorithm"]
    return [rd["position"] for rd in rec["reads"]
            if (exclude_locus is None or rd["locus"] != exclude_locus)
            and inj in algorithm_claims(rd["read"])]


def recurrent(positions: Sequence[int], k: int = K_RECUR, min_gap: int = MIN_GAP) -> bool:
    """>= k hits at NON-ADJACENT positions. Neighbouring reads share nearly the same prefix."""
    p = sorted(positions)
    if len(p) < k:
        return False
    chosen = [p[0]]
    for x in p[1:]:
        if x - chosen[-1] >= min_gap:
            chosen.append(x)
    return len(chosen) >= k


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar on discordant counts."""
    from scipy.stats import binomtest
    n = b + c
    return 1.0 if n == 0 else float(binomtest(b, n, 0.5, alternative="two-sided").pvalue)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reads", default=str(PROJ / "data/nla/n11/deception_reads.jsonl"))
    ap.add_argument("--out", default=str(PROJ / "data/nla/n11/deception_stats.json"))
    ap.add_argument("--n-perm", type=int, default=2000)
    args = ap.parse_args()

    rows = load(Path(args.reads))
    # Key on (dataset, snippet) — 142 names are shared between humaneval and cruxeval, so a
    # bare snippet key silently overwrites one dataset's capture with the other's.
    by = {((r["dataset"], r["snippet"]), r["condition"]): r for r in rows}
    snippets = sorted({s for s, _ in by})
    conds = ("C0", "C1", "C2")
    rep: dict[str, Any] = {"experiment": "n11_deception", "seed": SEED, "k_recurrence": K_RECUR,
                           "n_captures": len(rows), "n_snippets": len(snippets)}

    # ── quality + parse ────────────────────────────────────────────────────
    warnings: list[str] = []
    allrd = [rd for r in rows for rd in r["reads"]]
    rep["quality"] = {
        "n_reads": len(allrd),
        "rt_cos_median": _round(st.median(x["rt_cos"] for x in allrd)) if allrd else None,
        "cjk_gt_0.3": sum(x["cjk_frac"] > 0.30 for x in allrd),
    }
    rep["parse_rate"] = {c: _round(_mean([by[(s, c)]["parsed_answer"]
                                          for s in snippets if (s, c) in by]))
                         for c in conds}
    for c in conds:
        if rep["parse_rate"][c] is None:
            warnings.append(f"no captures for condition {c}")

    # ── read-level hit rate by locus x condition (descriptive) ─────────────
    cell = defaultdict(lambda: [0, 0])
    for r in rows:
        inj = r["injected_algorithm"]
        for rd in r["reads"]:
            k = (rd["locus"], r["condition"])
            cell[k][1] += 1
            cell[k][0] += inj in algorithm_claims(rd["read"])
    rep["by_locus"] = {L: {c: {"hits": cell[(L, c)][0], "n": cell[(L, c)][1],
                               "rate": round(cell[(L, c)][0] / cell[(L, c)][1], 4)
                               if cell[(L, c)][1] else None}
                           for c in conds}
                       for L in ("CODE_ID", "CODE_FAR", "COT", "ANSWER")}

    # ── distance gradient in C2 (defence 2) ────────────────────────────────
    buckets = defaultdict(lambda: [0, 0])
    for r in rows:
        if r["condition"] != "C2":
            continue
        inj = r["injected_algorithm"]
        for rd in r["reads"]:
            d = rd.get("char_distance")
            if rd["locus"] == "COT" or d is None:
                continue
            b = ("0 (on the identifier)" if d == 0 else "1-50" if d < 50 else
                 "50-150" if d < 150 else "150-400" if d < 400 else "400+")
            buckets[b][1] += 1
            buckets[b][0] += inj in algorithm_claims(rd["read"])
    rep["c2_distance_gradient"] = {b: {"hits": v[0], "n": v[1], "rate": round(v[0] / v[1], 4)}
                                   for b, v in buckets.items() if v[1]}

    # ── PRIMARY: item-level, echo-free, recurrence k=2, paired C2 vs C1 ────
    def item_flag(s: str, c: str, exclude_echo: bool, recur: bool) -> int | None:
        rec = by.get((s, c))
        if rec is None:
            return None
        h = hits(rec, ECHO_LOCUS if exclude_echo else None)
        return int(recurrent(h) if recur else bool(h))

    variants = {
        "primary (echo-free, k=2)": (True, True),
        "echo-free, any read (k=1)": (True, False),
        "all loci, k=2": (False, True),
        "all loci, any read (k=1) — echo-inflated": (False, False),
    }
    rep["item_level"] = {}
    for label, (ex, rc) in variants.items():
        row: dict[str, Any] = {}
        for c in conds:
            f = [item_flag(s, c, ex, rc) for s in snippets]
            f = [x for x in f if x is not None]
            row[c] = {"hits": sum(f), "n": len(f), "rate": _round(_mean(f))}
        paired = [(item_flag(s, "C1", ex, rc), item_flag(s, "C2", ex, rc)) for s in snippets]
        paired = [(a, b) for a, b in paired if a is not None and b is not None]
        b_only = sum(1 for a, b in paired if b == 1 and a == 0)   # C2 only
        c_only = sum(1 for a, b in paired if a == 1 and b == 0)   # C1 only
        both = row["C2"]["rate"] is not None and row["C1"]["rate"] is not None
        row["paired_C2_vs_C1"] = {
            "n_pairs": len(paired), "C2_only": b_only, "C1_only": c_only,
            "delta": _round(row["C2"]["rate"] - row["C1"]["rate"]) if both else None,
            "mcnemar_p": round(mcnemar_exact(b_only, c_only), 6),
        }
        rep["item_level"][label] = row

    # ── foreign-read null (defence 5) ──────────────────────────────────────
    # Precompute per-read hit flags; permutation reassigns reads between items and never
    # changes what a read says, so scoring inside the loop would be pure waste.
    # Store the SET of algorithms each read names, not a precomputed hit flag. The flag would be
    # relative to the read's OWN item's injected algorithm, so reusing it after reassignment
    # asks "does read X name X's algorithm" — which is the own-rate resampled, not a null at
    # all. (It came out ABOVE the own rate, because taking a max over resampled reads
    # saturates.) The question the null must ask is: does a read from item X name item **Y**'s
    # injected algorithm?
    rng = random.Random(SEED)
    pool = defaultdict(list)          # (condition, locus) -> [(snippet, frozenset(claims))]
    for r in rows:
        for rd in r["reads"]:
            if rd["locus"] == ECHO_LOCUS:
                continue
            pool[(r["condition"], rd["locus"])].append(
                (r["snippet"], frozenset(algorithm_claims(rd["read"]))))

    # The null is applied to the PRIMARY statistic (echo-free, recurrence k=2), not to the k=1
    # variant. HT14 failed because its foreign-answer null guarded a descriptive trajectory
    # while the log-rank — the actual decision rule — ran unguarded, so a confounded p=0.002
    # looked like support. Each item keeps its own read POSITIONS (so the adjacency structure
    # the recurrence rule depends on is preserved) and only the read CONTENT is drawn from
    # other items.

    rep["foreign_read_null"] = {}
    for c in conds:
        own = rep["item_level"]["primary (echo-free, k=2)"][c]["rate"]
        layout = [(s, by[(s, c)]["injected_algorithm"],
                   [(rd["locus"], rd["position"]) for rd in by[(s, c)]["reads"]
                    if rd["locus"] != ECHO_LOCUS])
                  for s in snippets if (s, c) in by]
        null = []
        for _ in range(args.n_perm):
            vals = []
            for s, inj, slots in layout:
                hit_pos = []
                for L, pos in slots:
                    cand = pool[(c, L)]
                    if not cand:
                        continue
                    for _t in range(8):
                        owner, claims = cand[rng.randrange(len(cand))]
                        if owner != s:
                            break
                    if inj in claims:          # THIS item's algorithm, THAT item's read
                        hit_pos.append(pos)
                vals.append(int(recurrent(hit_pos)))
            m = _mean(vals)
            if m is not None:
                null.append(m)
        if not null:
            rep["foreign_read_null"][c] = {"own": own, "null_mean": None,
                                           "note": "no items for this condition"}
            warnings.append(f"foreign null not computable for {c}")
            continue
        null.sort()
        rep["foreign_read_null"][c] = {
            "own": _round(own), "null_mean": _round(_mean(null)),
            "null_95ci": [round(null[int(0.025 * args.n_perm)], 4),
                          round(null[int(0.975 * args.n_perm)], 4)],
            "excess": _round(own - _mean(null)) if own is not None else None,
            "own_above_null_ci": bool(own is not None
                                      and own > null[int(0.975 * len(null))]),
        }

    # ── behavioural coupling (defence 4) ───────────────────────────────────
    coup: dict[str, Any] = {}
    for c in conds:
        stated = []
        for s in snippets:
            rec = by.get((s, c))
            if rec is None or not rec["parsed_answer"]:
                continue
            sh = rec["injected_algorithm"] in algorithm_claims(rec["stated_algorithm"] or "")
            rh = item_flag(s, c, True, True)
            stated.append((sh, rh))
        n_sh = sum(a for a, _ in stated)
        coup[c] = {
            "n_parsed": len(stated), "stated_hit": n_sh,
            "read_rate_when_stated": _round(_mean([b for a, b in stated if a])),
            "read_rate_when_not_stated": _round(_mean([b for a, b in stated if not a])),
        }
    rep["behavioural_coupling"] = coup

    rep["warnings"] = warnings
    rep["finished_utc"] = datetime.now(timezone.utc).isoformat()
    Path(args.out).write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
