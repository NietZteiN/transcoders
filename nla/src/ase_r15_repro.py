"""H-R15 — the run-to-run REPLICATION FLOOR of this pipeline, and whether H-R14's one positive effect
survives it. Pre-registered in log/nla-harness/2026-09-17_replication-floor-prereg.md; thresholds are
read from that entry and hard-coded here, so no verdict can be moved after seeing the data.

WHAT IS AND IS NOT BEING MEASURED. Every run in this family already shared one hard-coded
`torch.manual_seed(20260724)`, so the H-R7 -> H-R14 drift was never a seed change: it mixed RNG-stream
position, hardware/kernel nondeterminism, and (for `ridge_map` alone) a genuine change of vectors.
Here the snippet set, the packs, the runtime and the vectors are all held fixed and the ONLY thing that
varies is `--seed` -- except in replicate A2, where not even that varies. A2 is the control that decides
how to read everything else: this thread already measured 0.8333 per-item agreement on a comparable
pipeline with deterministic kernels PINNED (2026-08-29), so a same-seed re-run is not expected to
reproduce, and if it does not, "seed noise" is the wrong name for the floor.

CONTRASTS ARE ALWAYS COMPUTED WITHIN A REPLICATE (B2-B1, B3-B1, never B2-A1), so no contrast straddles
two decoding streams -- that is the pairing our published intervals already rely on, and comparing
across it would inflate the floor for free.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ase_bakeoff_stats import cw, load, load_truth  # noqa: E402

TAG = "[R15]"
DET_ACC_TOL, DET_AGREE_MIN = 0.005, 0.99      # H-R15b
AUTO_REPL, AUTO_FAIL = 0.02, 0.01             # H-R15c


def case_preds(path: Path) -> dict:
    """{(snippet, run_idx, case_id): label} over every banked run."""
    out = {}
    for l in open(path):
        if not l.strip():
            continue
        r = json.loads(l)
        for i, run in enumerate(r["runs"]):
            for c, v in run["pred"].items():
                out[(r["snippet"], i, c)] = v
    return out


def agreement(a: dict, b: dict) -> tuple[float, int]:
    """Fraction of shared case-runs whose emitted label matches. An unparsed case counts as a label
    (None) on both sides, so 'both failed to answer' is agreement -- that is the honest reading of
    'did the pipeline do the same thing twice'."""
    keys = set(a) | set(b)
    if not keys:
        return float("nan"), 0
    return float(np.mean([a.get(k) == b.get(k) for k in keys])), len(keys)


def split(path: Path, truth: dict) -> dict:
    """acc / parse / acc-given-parsed over every case x run."""
    npar = ncase = ncorr = 0
    for l in open(path):
        if not l.strip():
            continue
        r = json.loads(l)
        tr = truth[r["snippet"]]
        for run in r["runs"]:
            for c, t in tr.items():
                ncase += 1
                got = run["pred"].get(c)
                if got is not None:
                    npar += 1
                    ncorr += (got == t)
    return {"acc": ncorr / ncase, "parse": npar / ncase, "acc_given_parsed": ncorr / npar}


def contrast(arm: dict, base: dict) -> float:
    keys = sorted(set(arm) & set(base))
    x = np.array([arm[k][0] - base[k][0] for k in keys])
    n = np.array([arm[k][1] for k in keys], dtype=float)
    return cw(x, n)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--h14-dir", required=True, help="H-R14 bakeoff_full dir (replicate A1)")
    ap.add_argument("--repro-dir", required=True, help="dir with <rep>.<arm>.jsonl replicates")
    ap.add_argument("--h14-stats", required=True, help="H-R14 stats json (supplies the half-widths)")
    ap.add_argument("--packs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    truth = load_truth(args.packs)
    H, R = Path(args.h14_dir), Path(args.repro_dir)

    # half-widths h come from the PUBLISHED H-R14 intervals, not retyped
    h14 = json.load(open(args.h14_stats))["scorings"]["cn"]["vs_unsteered"]
    half = {a: (d["ci95"][1] - d["ci95"][0]) / 2 for a, d in h14.items()}

    files = {"A1.unsteered": H / "unsteered.jsonl",
             "A1.ridge_map": H / "ridge_map.jsonl",
             "A1.codesteer_auto": H / "codesteer_auto.jsonl"}
    for p in sorted(R.glob("*.jsonl")):
        files[p.stem] = p                       # e.g. "B1.unsteered"
    missing = [k for k, v in files.items() if not v.exists()]
    st: dict = {"experiment": "R15_replication_floor",
                "prereg": "log/nla-harness/2026-09-17_replication-floor-prereg.md",
                "present": sorted(k for k, v in files.items() if v.exists()), "missing": missing,
                "half_widths_from_h14": half}

    have = {k: v for k, v in files.items() if v.exists()}
    st["per_run"] = {k: split(v, truth) for k, v in have.items()}
    preds = {k: case_preds(v) for k, v in have.items()}
    scored = {k: load(v, "cn", truth) for k, v in have.items()}

    # ---- H-R15b: is a same-seed re-run a reproduction? ----
    if "A2.unsteered" in have:
        ag, n = agreement(preds["A1.unsteered"], preds["A2.unsteered"])
        d = abs(st["per_run"]["A2.unsteered"]["acc"] - st["per_run"]["A1.unsteered"]["acc"])
        st["H_R15b"] = {"agreement": ag, "n_case_runs": n, "abs_acc_delta": d,
                        "verdict": ("SEED-DETERMINISTIC" if d <= DET_ACC_TOL and ag >= DET_AGREE_MIN
                                    else "SEED-NOT-DETERMINISTIC")}

    # ---- pairwise agreement table (descriptive; compare with 0.8333 from 2026-08-29) ----
    uns = [k for k in have if k.endswith(".unsteered")]
    st["agreement"] = {}
    for i, x in enumerate(sorted(uns)):
        for y in sorted(uns)[i + 1:]:
            a, n = agreement(preds[x], preds[y])
            st["agreement"][f"{x} vs {y}"] = {"agreement": a, "n": n,
                                              "same_seed": x.startswith("A") and y.startswith("A")}

    # ---- H-R15a: within-replicate contrasts, and how far they move ----
    st["contrasts"] = {}
    for arm in ("ridge_map", "codesteer_auto"):
        per = {}
        for rep, basekey in (("A1", "A1.unsteered"), ("B", "B1.unsteered")):
            armkey = f"{rep}.{arm}" if rep == "A1" else next(
                (k for k in have if k.endswith(f".{arm}") and k.startswith("B")), None)
            if armkey in have and basekey in have:
                per[rep] = contrast(scored[armkey], scored[basekey])
        if len(per) == 2:
            s = abs(per["B"] - per["A1"])
            st["contrasts"][arm] = {"delta_A1": per["A1"], "delta_B": per["B"], "drift": s,
                                    "half_width": half.get(arm),
                                    "drift_over_half": s / half[arm] if half.get(arm) else None}
    if st["contrasts"]:
        ratios = {a: v["drift_over_half"] for a, v in st["contrasts"].items()}
        st["H_R15a"] = {"drift_over_half": ratios,
                        "verdict": ("CI-OPTIMISTIC" if any(r >= 0.5 for r in ratios.values())
                                    else "CI-HONEST" if all(r <= 1 / 3 for r in ratios.values())
                                    else "CI-MARGINAL")}

    # ---- H-R15c: does codesteer_auto's effect survive a replicate? ----
    if "codesteer_auto" in st["contrasts"]:
        d = st["contrasts"]["codesteer_auto"]["delta_B"]
        st["H_R15c"] = {"delta_B": d, "delta_A1": st["contrasts"]["codesteer_auto"]["delta_A1"],
                        "verdict": ("AUTO-REPLICATES" if d >= AUTO_REPL
                                    else "AUTO-DOES-NOT-REPLICATE" if d <= AUTO_FAIL
                                    else "AUTO-PARTIAL")}

    # ---- the unsteered baseline's own spread across replicates ----
    accs = [st["per_run"][k]["acc"] for k in uns]
    if len(accs) > 1:
        st["unsteered_spread"] = {"runs": {k: st["per_run"][k]["acc"] for k in sorted(uns)},
                                  "range": max(accs) - min(accs), "sd": float(np.std(accs, ddof=1))}

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(st, indent=1, default=str))

    print(f"\n{TAG} ===== per run =====")
    for k in sorted(st["per_run"]):
        v = st["per_run"][k]
        print(f"  {k:24s} acc {v['acc']:.4f}  parse {v['parse']:.4f}  acc|p {v['acc_given_parsed']:.4f}")
    if "unsteered_spread" in st:
        u = st["unsteered_spread"]
        print(f"{TAG} unsteered across {len(u['runs'])} runs: range {u['range']:.4f}  sd {u['sd']:.4f}")
    print(f"\n{TAG} ===== per-item label agreement (2026-08-29 banked 0.8333 on the other pipeline) =====")
    for k, v in st["agreement"].items():
        print(f"  {k:46s} {v['agreement']:.4f}  ({'SAME seed' if v['same_seed'] else 'diff seed'})")
    print(f"\n{TAG} ===== contrasts, computed WITHIN each replicate =====")
    for a, v in st["contrasts"].items():
        print(f"  {a:16s} A1 {v['delta_A1']:+.4f}  B {v['delta_B']:+.4f}  drift {v['drift']:.4f} "
              f"= {v['drift_over_half']:.2f} x the published half-width ({v['half_width']:.4f})")
    for h in ("H_R15a", "H_R15b", "H_R15c"):
        if h in st:
            print(f"{TAG} {h}: {st[h]['verdict']}")
    print(f"\n{TAG} wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
