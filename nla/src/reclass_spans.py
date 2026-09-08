"""H-W33 — reclassify obfuscated-tier spans by what the tier DID, not by what the name looks like.

Pre-registered: `log/nla-harness/2026-09-08_l3-span-classes-prereg.md`.

`src/convert_stimuli.py` decides `l1_neutral` vs `adversarial` with a regex on the new name's shape.
That works for Python's `var_68f8` and fails for JavaScript's `a` / `uepoi`, so 1,703 L3 spans carry
a decoy label on code that (per 2026-09-07) contains no decoy.

This asks the question the label is meant to answer: for this original identifier, is the name the
tier gave it the one **L1** gave it (a meaningless rename) or the one **L1b** gave it (a plausible
but wrong name)? Both are looked up in the H-W28 pairings, so the classification is a verification
against another tier rather than a guess from a string.

The `adversarial` test is applied BEFORE the `nonsense` test on purpose: if a tier ever does carry a
decoy, this ordering finds it instead of absorbing it into the nonsense class. The rule is built so
its own premise can fail.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

_PROJ = Path(__file__).resolve().parents[2]


def load_pairings() -> dict[tuple[str, str, str], dict[str, str]]:
    """(dataset, snippet_id, tier) -> {original: renamed}, from H-W28's validated output."""
    out: dict[tuple[str, str, str], dict[str, str]] = {}
    p = _PROJ / "data/stimuli/pairing_recovered.jsonl"
    for line in open(p):
        r = json.loads(line)
        out[(r["dataset"], r["snippet_id"], r["tier"])] = r["pairs"]
    return out


def classify(orig: str, name: str, l1: dict[str, str], l1b: dict[str, str]) -> tuple[str, str]:
    """(class, evidence). Order is frozen: unchanged -> adversarial -> nonsense -> unresolved."""
    if name == orig:
        return "unchanged", "name == original"
    if orig in l1b and l1b[orig] == name:
        return "adversarial", f"matches L1b rename {l1b[orig]!r}"
    if orig in l1 and l1[orig] == name:
        return "nonsense", f"matches L1 rename {l1[orig]!r}"
    # JavaScript L1 pairings were REFUSED by H-W28 for lack of ground truth, so the positive
    # "matches L1" test is unavailable there. The negative one is not: if the original HAS an L1b
    # rename and this name is not it, `adversarial` is excluded on evidence even though `nonsense`
    # cannot be positively confirmed. That asymmetry is reported, not smoothed over.
    if orig in l1b:
        return "not_adversarial", f"L1b renamed it {l1b[orig]!r}, this is {name!r}"
    return "unresolved", f"no usable pairing (L1={l1.get(orig)!r}, L1b={l1b.get(orig)!r})"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(_PROJ / "data/stimuli/span_classes_v2.jsonl"))
    ap.add_argument("--report", default=str(_PROJ / "data/stimuli/span_classes_v2_report.json"))
    args = ap.parse_args()

    pair = load_pairings()
    rows: dict[tuple[str, str], dict[str, dict]] = {}
    for ds in ("a", "b"):
        for line in open(_PROJ / f"data/stimuli/dataset_{ds}/dataset_{ds}.jsonl"):
            r = json.loads(line)
            rows.setdefault((ds, r["snippet_id"]), {})[r["tier"]] = r

    tally: Counter = Counter()
    moved: Counter = Counter()
    out = open(args.out, "w")
    n = 0
    for (ds, sid), tiers in sorted(rows.items()):
        l1 = pair.get((ds, sid, "L1"), {})
        l1b = pair.get((ds, sid, "L1b"), {})
        inv = {}                                   # per tier: renamed -> original, from its pairing
        for tier in ("L1", "L1b", "L3"):
            row = tiers.get(tier)
            if row is None:
                continue
            lang = row["language"]
            pt = pair.get((ds, sid, tier), {})
            inv[tier] = {v: k for k, v in pt.items()}
            for sp in (row.get("meta") or {}).get("span_info") or []:
                old = sp.get("cls")
                name = (sp.get("name") or "").strip()
                orig = inv[tier].get(name)
                if orig is None:
                    new, ev = "unresolved", "renamed name has no pairing back to an original"
                else:
                    new, ev = classify(orig, name, l1, l1b)
                tally[(tier, lang, old, new)] += 1
                if old in ("adversarial", "l1_neutral") and new not in ("unresolved",):
                    moved[(tier, lang, old, new)] += 1
                out.write(json.dumps({"dataset": ds, "snippet_id": sid, "tier": tier,
                                      "language": lang, "name": name, "original": orig,
                                      "class_old": old, "class_new": new, "evidence": ev}) + "\n")
                n += 1
    out.close()

    # gates
    def resolvable(tier: str) -> tuple[int, Counter]:
        c = Counter()
        for (t, lang, old, new), k in tally.items():
            if t == tier and new not in ("unresolved", "unchanged"):
                c[new] += k
        return sum(c.values()), c

    n3, c3 = resolvable("L3")
    n1b, c1b = resolvable("L1b")
    frac_non = c3["nonsense"] / n3 if n3 else float("nan")
    frac_adv_l1b = c1b["adversarial"] / n1b if n1b else float("nan")
    # `nonsense` needs the L1 pairing (Python only); `not_adversarial` needs only the L1b pairing
    # and is available in both languages. The frozen rule is scored on `nonsense`; the weaker
    # exclusion is reported alongside so the JavaScript half is not silently dropped.
    frac_non = c3["nonsense"] / n3 if n3 else float("nan")
    frac_excl = (c3["nonsense"] + c3["not_adversarial"]) / n3 if n3 else float("nan")
    # The verdict is the THREE-WAY rule as frozen in the prereg. `frac_excl` is computed and
    # reported but must not enter the verdict: `W33-CONFIRMED-BY-EXCLUSION` was a fourth outcome I
    # invented mid-run after seeing that JavaScript cannot supply the positive test, and adding a
    # verdict word once the data is visible is the forking path this thread has avoided repeatedly.
    # The exclusion evidence is strong and is reported as a post-hoc reading, clearly labelled.
    v33a = ("W33-MIXED" if c3["adversarial"] > 0
            else "W33-CONFIRMED" if frac_non >= 0.95
            else "W33-UNRESOLVED")
    v33b = "PASS" if frac_adv_l1b >= 0.95 else "FAIL"

    rep = {"experiment": "W33_span_reclass", "n_spans": n,
           "L3_resolvable": n3, "L3_classes": dict(c3), "L3_frac_nonsense": frac_non, "L3_frac_not_adversarial_or_nonsense": frac_excl,
           "L1b_resolvable": n1b, "L1b_classes": dict(c1b), "L1b_frac_adversarial": frac_adv_l1b,
           "H_W33a_frozen_rule": v33a, "post_hoc_frac_excluded_from_adversarial": frac_excl, "H_W33b_control": v33b,
           "tally": {"|".join(str(x) for x in k): v for k, v in sorted(tally.items(), key=str)}}
    json.dump(rep, open(args.report, "w"), indent=2)

    print(f"{n} spans written to {args.out}\n")
    print(f"{'tier':5s} {'lang':11s} {'old':12s} -> {'new':12s} {'count':>6s}")
    for k, v in sorted(tally.items(), key=str):
        print(f"{k[0]:5s} {k[1]:11s} {str(k[2]):12s} -> {k[3]:12s} {v:>6d}")
    print(f"\nH-W33a  L3 resolvable renames {n3}: {dict(c3)}")
    print(f"        frac nonsense {frac_non:.4f} | frac (nonsense or not-adversarial) {frac_excl:.4f} -> {v33a}")
    print(f"H-W33b  L1b resolvable renames {n1b}: {dict(c1b)}  frac adversarial {frac_adv_l1b:.4f} -> {v33b}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
