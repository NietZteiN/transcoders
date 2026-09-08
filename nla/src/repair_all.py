"""H-W28 — re-derive the original↔renamed identifier pairing for the WHOLE stimulus set.

Pre-registered: `log/nla-harness/2026-09-07_pairing-recovery-prereg.md`. The generator's
`rename_map` uses equal-length positional matching and writes `?unpairedN` keys when that fails,
which is 42–100 % of keys depending on tier and language. `repair_pairs.recover()` (H-W22) rebuilds
the correspondence by aligning identifier-masked token sequences; it was validated and used for the
W corpus' 60 items only. This applies it to both datasets and all three renamed tiers.

The one non-obvious choice is the ANCHOR, and it follows from `2026-09-07_l3-is-l1-not-l1b.md`:
L3 = L1 ∘ L2, so L3 differs from L0 by a rename AND a flattening but from **L2 by a rename only**.
Since L2 keeps L0's identifiers, an L2→L3 pair IS an L0→L3 pair. Both routes are scored here so the
choice is measured rather than asserted (H-W28b).

Accuracy is gated per (dataset, tier, language) cell against the pairs the pipeline did record, and
a cell that fails the gate is REFUSED — reported, never written. Output is a sibling file; the
stimulus files are never edited in place.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from repair_pairs import recover  # noqa: E402

_PROJ = _HERE.parent.parent
RENAMED_TIERS = ("L1", "L1b", "L3")
ANCHOR = {"L1": "L0", "L1b": "L0", "L3": "L2"}      # frozen; H-W28b also scores L3 on L0
MIN_PRECISION = 0.95
MIN_COMPARABLE = 20


def is_sentinel(k: str) -> bool:
    return str(k).startswith("?unpaired")


def load(ds: str) -> dict[str, dict[str, dict]]:
    """{snippet_id: {tier: row}} for one dataset."""
    out: dict[str, dict[str, dict]] = {}
    for line in open(_PROJ / f"data/stimuli/dataset_{ds}/dataset_{ds}.jsonl"):
        r = json.loads(line)
        out.setdefault(r["snippet_id"], {})[r["tier"]] = r
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(_PROJ / "data/stimuli/pairing_recovered.jsonl"))
    ap.add_argument("--report", default=str(_PROJ / "data/stimuli/pairing_recovery_report.json"))
    ap.add_argument("--dry-run", action="store_true", help="score only; write nothing")
    args = ap.parse_args()

    # cell -> validation counters; recovered pairs held back until the gate is scored
    val: dict[tuple, dict] = defaultdict(lambda: {"agree": 0, "compared": 0, "recorded_real": 0})
    l3_alt: dict[tuple, dict] = defaultdict(lambda: {"agree": 0, "compared": 0})
    cov: dict[tuple, dict] = defaultdict(lambda: {"keys": 0, "sentinel": 0, "filled": 0})
    staged: list[dict] = []

    for ds in ("a", "b"):
        items = load(ds)
        for sid, tiers in sorted(items.items()):
            for tier in RENAMED_TIERS:
                row = tiers.get(tier)
                anch = tiers.get(ANCHOR[tier])
                if row is None or anch is None:
                    continue
                lang = row["language"]
                cell = (ds, tier, lang)
                meta = row.get("meta") or {}
                rmap = meta.get("rename_map") or {}
                real = {k: v for k, v in rmap.items() if not is_sentinel(k)}
                cov[cell]["keys"] += len(rmap)
                cov[cell]["sentinel"] += sum(1 for k in rmap if is_sentinel(k))
                val[cell]["recorded_real"] += len(real)

                rec, diag = recover(anch["code"], row["code"], lang)
                # H-W28a: score on the originals both sources name
                for o, r in rec.items():
                    if o in real:
                        val[cell]["compared"] += 1
                        val[cell]["agree"] += int(real[o] == r)
                # H-W28b: the same measurement with L3 anchored on L0 instead
                if tier == "L3" and tiers.get("L0") is not None:
                    alt, _ = recover(tiers["L0"]["code"], row["code"], lang)
                    for o, r in alt.items():
                        if o in real:
                            l3_alt[cell]["compared"] += 1
                            l3_alt[cell]["agree"] += int(real[o] == r)

                merged = {o: {"renamed": v, "source": "pipeline"} for o, v in real.items()}
                new = {o: {"renamed": r, "source": "recovered"}
                       for o, r in rec.items() if o not in merged}
                cov[cell]["filled"] += len(new)
                staged.append({"dataset": ds, "snippet_id": sid, "tier": tier, "language": lang,
                               "anchor": ANCHOR[tier], "pipeline": merged, "recovered": new,
                               "diag": diag})

    def prec(d: dict) -> float:
        return d["agree"] / d["compared"] if d["compared"] else float("nan")

    cells = {}
    for cell, d in sorted(val.items()):
        p = prec(d)
        gate = ("REFUSED-UNVALIDATED" if d["compared"] < MIN_COMPARABLE
                else "ACCEPTED" if p >= MIN_PRECISION else "REFUSED-PRECISION")
        cells["|".join(cell)] = {
            "precision": p, "compared": d["compared"], "agree": d["agree"],
            "recorded_real": d["recorded_real"],
            "recall": d["compared"] / d["recorded_real"] if d["recorded_real"] else float("nan"),
            "gate": gate, **{k: v for k, v in cov[cell].items()}}

    # Pre-registered borrowing clause: a cell with too few checkable pairs of its own may take the
    # verdict of the SAME (tier, language) in the other dataset, if that one was accepted on its own
    # evidence. Marked distinctly so a consumer can exclude borrowed cells; it never overrides a
    # cell that failed on precision, only one that could not be scored at all.
    for c, v in cells.items():
        if v["gate"] != "REFUSED-UNVALIDATED":
            continue
        ds, tier, lang = c.split("|")
        other = cells.get("|".join(("b" if ds == "a" else "a", tier, lang)))
        if other and other["gate"] == "ACCEPTED":
            v["gate"] = "ACCEPTED-BORROWED"
            v["borrowed_from"] = "|".join(("b" if ds == "a" else "a", tier, lang))
            v["borrowed_precision"] = other["precision"]
            v["borrowed_n"] = other["compared"]

    accepted = {c for c, v in cells.items() if v["gate"].startswith("ACCEPTED")}
    written = 0
    if not args.dry_run:
        with open(args.out, "w") as f:
            for rec_row in staged:
                key = "|".join((rec_row["dataset"], rec_row["tier"], rec_row["language"]))
                pairs = dict(rec_row["pipeline"])
                if key in accepted:
                    pairs.update(rec_row["recovered"])
                    written += len(rec_row["recovered"])
                f.write(json.dumps({
                    "dataset": rec_row["dataset"], "snippet_id": rec_row["snippet_id"],
                    "tier": rec_row["tier"], "language": rec_row["language"],
                    "anchor": rec_row["anchor"], "cell_gate": cells[key]["gate"],
                    "pairs": {o: p["renamed"] for o, p in pairs.items()},
                    "source": {o: p["source"] for o, p in pairs.items()}}) + "\n")

    report = {"experiment": "W28_pairing_recovery", "min_precision": MIN_PRECISION,
              "min_comparable": MIN_COMPARABLE, "anchor": ANCHOR, "cells": cells,
              "l3_anchor_comparison": {
                  "|".join(c): {"L2_anchored": prec(val[c]), "L0_anchored": prec(d),
                                "compared_L2": val[c]["compared"], "compared_L0": d["compared"]}
                  for c, d in sorted(l3_alt.items())},
              "pairs_written": written, "rows": len(staged), "dry_run": bool(args.dry_run)}
    json.dump(report, open(args.report, "w"), indent=2)

    print(f"{'cell':<22} {'gate':<20} {'prec':>6} {'cmp':>5} {'real':>5} "
          f"{'sentinel/keys':>14} {'filled':>7}")
    for c, v in cells.items():
        print(f"{c:<22} {v['gate']:<20} {v['precision']:>6.3f} {v['compared']:>5} "
              f"{v['recorded_real']:>5} {v['sentinel']:>6}/{v['keys']:<7} {v['filled']:>7}")
    print("\nH-W28b — L3 anchor comparison (precision on the same comparable pairs):")
    for c, v in report["l3_anchor_comparison"].items():
        print(f"  {c:<22} L2-anchored {v['L2_anchored']:.3f} (n={v['compared_L2']})   "
              f"L0-anchored {v['L0_anchored']:.3f} (n={v['compared_L0']})")
    print(f"\nrows {len(staged)} · pairs written {written} · report {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
