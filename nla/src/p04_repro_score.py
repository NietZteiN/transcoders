"""P0.4-repro — the per-item reproducibility floor.

Reports two numbers that every paired per-item claim in this programme should carry:

  same-GPU agreement   A1 vs A2, one job, one allocated card  -> run-to-run nondeterminism
  cross-node agreement A1 vs B1, different physical cards      -> card-to-card nondeterminism

separately for the UNSTEERED baseline (where any disagreement is pure decoding nondeterminism)
and for the STEERED condition (which adds the hook's own arithmetic). Marginal accuracies are
reported alongside, because the P0.4 finding was precisely that marginals can agree exactly while
individual items do not — and it is the marginal that everyone quotes.

No decision rule. This is calibration: the cross-node figure is the floor, and a paired per-item
effect smaller than it is not resolvable by this pipeline.

No GPU. Seconds.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJ = Path(__file__).resolve().parent.parent.parent


def load(rep_dir: Path) -> dict[str, Any]:
    base, steer = {}, {}
    bp, sp = rep_dir / "baseline.jsonl", rep_dir / "steer_results.jsonl"
    if bp.exists():
        for l in open(bp):
            if l.strip():
                r = json.loads(l)
                base[r["snippet_id"]] = r
    if sp.exists():
        for l in open(sp):
            if not l.strip():
                continue
            r = json.loads(l)
            if "error" in r or r.get("condition") != "V4_oracle":
                continue
            steer[r["snippet_id"]] = r
    node = (rep_dir / "node.txt").read_text().strip() if (rep_dir / "node.txt").exists() else None
    gpu = (rep_dir / "gpu.txt").read_text().strip() if (rep_dir / "gpu.txt").exists() else None
    return {"base": base, "steer": steer, "node": node, "gpu": gpu, "dir": rep_dir.name}


def agree(a: dict, b: dict, key: str) -> dict[str, Any]:
    shared = sorted(set(a) & set(b))
    if not shared:
        return {"n_shared": 0, "agreement": None}
    same = [bool(a[s][key]) == bool(b[s][key]) for s in shared]
    return {
        "n_shared": len(shared),
        "agreement": round(st.mean(same), 4),
        "n_disagree": sum(1 for x in same if not x),
        "acc_a": round(st.mean(bool(a[s][key]) for s in shared), 4),
        "acc_b": round(st.mean(bool(b[s][key]) for s in shared), 4),
        "marginal_gap": round(abs(st.mean(bool(a[s][key]) for s in shared)
                                  - st.mean(bool(b[s][key]) for s in shared)), 4),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(PROJ / "data/nla/p0/p04/repro"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    root = Path(args.root)
    out_p = Path(args.out) if args.out else root / "p04_repro.json"

    tags = [t for t in ("A1", "A2", "B1", "D1", "D2", "E1", "E2") if (root / t).exists()]
    reps = {t: load(root / t) for t in tags}
    rep: dict[str, Any] = {
        "experiment": "p0.4_repro_per_item_floor",
        "entry": "log/nla-harness/2026-08-29_reproducibility-floor.md",
        "replicates": {t: {"node": r["node"], "gpu": r["gpu"],
                           "n_baseline": len(r["base"]), "n_steered": len(r["steer"])}
                       for t, r in reps.items()},
    }
    missing = [t for t, r in reps.items() if not r["base"] or not r["steer"]]
    if missing:
        rep["verdict"] = "NEEDS_DATA"
        rep["reason"] = f"replicates incomplete: {missing}"
    else:
        pairs = {"same_gpu_A1_vs_A2": ("A1", "A2"), "cross_node_A1_vs_B1": ("A1", "B1"),
                 "cross_node_A2_vs_B1": ("A2", "B1"),
                 # D1/D2 are the same pairing as A1/A2 with deterministic kernels pinned; the
                 # comparison of the two agreements is the whole experiment. D1_vs_A1 asks the
                 # separate question of whether determinism CHANGES answers rather than only
                 # stabilising them — if it does, deterministic runs cannot be pooled with the bank.
                 "same_gpu_deterministic_D1_vs_D2": ("D1", "D2"),
                 "deterministic_vs_default_D1_vs_A1": ("D1", "A1"),
                 # E1/E2: the A1/A2 pairing again with MAX_NEW_GEN 1100 -> 2048. If the floor is
                 # partly replies truncated before they emit an answer line, this pair should
                 # agree better than A1/A2 did.
                 "same_gpu_budget2048_E1_vs_E2": ("E1", "E2"),
                 "budget2048_vs_1100_E1_vs_A1": ("E1", "A1")}
        pairs = {n: (x, y) for n, (x, y) in pairs.items() if x in reps and y in reps}
        for name, (x, y) in pairs.items():
            rep[name] = {
                "baseline_l1b": agree(reps[x]["base"], reps[y]["base"], "l1b_correct"),
                "baseline_l0": agree(reps[x]["base"], reps[y]["base"], "l0_correct"),
                "steered_V4_alpha1": agree(reps[x]["steer"], reps[y]["steer"], "correct"),
            }
        same = rep.get("same_gpu_A1_vs_A2", {})
        cross = rep.get("cross_node_A1_vs_B1", {})
        det = rep.get("same_gpu_deterministic_D1_vs_D2")
        if det:
            b, t = det["baseline_l1b"]["agreement"], det["steered_V4_alpha1"]["agreement"]
            rep["determinism"] = {
                "deterministic_same_gpu_baseline": b,
                "deterministic_same_gpu_steered": t,
                "default_same_gpu_baseline": same.get("baseline_l1b", {}).get("agreement"),
                "default_same_gpu_steered": same.get("steered_V4_alpha1", {}).get("agreement"),
                # The rule frozen in p04_det_sbatch.sh, applied arithmetically.
                "verdict": ("FLOOR IS A CHOICE — deterministic kernels reproduce exactly"
                            if b == 1.0 and t == 1.0 else
                            "FLOOR IS STRUCTURAL — something survives kernel pinning"),
            }
        bud = rep.get("same_gpu_budget2048_E1_vs_E2")
        if bud:
            b, t = bud["baseline_l1b"]["agreement"], bud["steered_V4_alpha1"]["agreement"]
            db, dt = same.get("baseline_l1b", {}), same.get("steered_V4_alpha1", {})
            rep["truncation"] = {
                "budget2048_same_gpu_baseline": b, "budget2048_same_gpu_steered": t,
                "budget1100_same_gpu_baseline": db.get("agreement"),
                "budget1100_same_gpu_steered": dt.get("agreement"),
                "parse_rate_2048_E1": None, "parse_rate_1100_A1": None,
                "verdict": ("TRUNCATION WAS A COMPONENT — a bigger budget raises agreement"
                            if (b or 0) > (db.get("agreement") or 1)
                            and (t or 0) > (dt.get("agreement") or 1)
                            else "TRUNCATION IS NOT THE MECHANISM — the floor is unmoved"),
            }
            for tag, key in (("E1", "parse_rate_2048_E1"), ("A1", "parse_rate_1100_A1")):
                bs = reps[tag]["base"]
                if bs:
                    rep["truncation"][key] = round(
                        st.mean(bool(v.get("l1b_parsed")) for v in bs.values()), 4)
        rep["floor"] = {
            "same_gpu_baseline": same.get("baseline_l1b", {}).get("agreement"),
            "cross_node_baseline": cross.get("baseline_l1b", {}).get("agreement"),
            "same_gpu_steered": same.get("steered_V4_alpha1", {}).get("agreement"),
            "cross_node_steered": cross.get("steered_V4_alpha1", {}).get("agreement"),
            "note": ("The cross-node figures are the floor any paired per-item claim must carry. "
                     "A same-GPU agreement of 1.0 would mean the pipeline is deterministic given "
                     "a fixed card and that all irreproducibility is card-to-card."),
        }
        rep["verdict"] = "MEASURED"

    rep["finished_utc"] = datetime.now(timezone.utc).isoformat()
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
