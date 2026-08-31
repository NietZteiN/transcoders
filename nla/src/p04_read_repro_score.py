"""Compare read-reproducibility replicates across processes (same card).

`p04_read_repro.py` reports the WITHIN-process comparison and saves its activation matrix. This
joins two saved matrices to give the CROSS-process figure, which is the one that matters: the
generation floor was measured across processes on one card, so the read side has to be tested
the same way to be comparable.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJ = Path(__file__).resolve().parent.parent.parent
import sys  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from p04_read_repro import summarize  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(PROJ / "data/nla/p0/p04/readrepro"))
    ap.add_argument("--tags", default="R1,R2")
    args = ap.parse_args()
    root = Path(args.root)
    t1, t2 = [t.strip() for t in args.tags.split(",")]

    rep = {"experiment": "p0.4_read_reproducibility_cross_process", "tags": [t1, t2]}
    f1, f2 = root / f"{t1}.npy", root / f"{t2}.npy"
    if not (f1.exists() and f2.exists()):
        rep["verdict"] = "NEEDS_DATA"
        rep["reason"] = f"missing {[str(f) for f in (f1, f2) if not f.exists()]}"
    else:
        a, b = np.load(f1), np.load(f2)
        rep["cross_process"] = summarize(a, b)
        for t in (t1, t2):
            w = root / f"{t}_within.json"
            if w.exists():
                rep[f"within_process_{t}"] = json.loads(w.read_text())["within_process"]
        # The frozen rule: bit-identical reads mean the floor is a generation phenomenon and
        # read-side measures carry no caveat.
        rep["verdict"] = ("READS REPRODUCE EXACTLY — floor is generation-only"
                          if rep["cross_process"]["frac_bit_identical"] == 1.0
                          else "READS ALSO DRIFT — the floor is pipeline-wide")
    rep["finished_utc"] = datetime.now(timezone.utc).isoformat()
    (root / "read_repro.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
