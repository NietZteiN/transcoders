"""H-C4/H-C5/H-C6: apply the frozen dose-response rules to the per-dose gate rows.

Pre-registered in log/nla-harness/2026-09-12_dose-response-prereg.md. Thresholds are module
constants here rather than a config because they encode a *derivation* that must not drift: the
banked L7 `S_swap` is +73.15, so a 0.02 change in the fidelity ratio is 0.02 * 73.15 = +1.46 nats
of `S_c3`, and +1.46 is the step size every rule below is written on.

WHY `S_c3` PAIRED PER ITEM AND NOT THE RATIO. `S_swap` is the raw clean state written at the span
positions; it does not involve the trained AV/AR at all, so it is a constant baseline across dose
points and the fidelity ratio moves only through `S_c3`. Comparing two ratio-of-means would throw
away the pairing and compare intervals that overlap by construction (the banked L7 ratio CI is
~0.10 wide against a 0.02 effect); pairing per item over the same 60 items is far more sensitive.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

LAYER = 7
STEP_NATS = 1.46            # = 0.02 * 73.15, the banked L7 S_swap
CONTROL_DRIFT_NATS = 4.0    # H-C5 flag: > this and H-C4's verdict is provisional
N_BOOT = 10_000
SEED = 20260724
# H-C4's pre-flight: the re-extracted activations must reproduce the banked L7 norms exactly.
BANKED_INJECTION_SCALE = 5100
BANKED_MEAN_AV_TRAIN = 5015.6255
MEAN_AV_TOL = 0.5


def boot(x, seed_off: int = 0) -> dict:
    x = np.asarray(x, dtype=float)
    if not len(x):
        return {"mean": float("nan"), "ci95": [float("nan")] * 2, "n": 0}
    rng = np.random.default_rng(SEED + seed_off)
    b = np.sort(x[rng.integers(0, len(x), size=(N_BOOT, len(x)))].mean(1))
    return {"mean": float(x.mean()),
            "ci95": [float(b[int(.025 * N_BOOT)]), float(b[int(.975 * N_BOOT)])], "n": int(len(x))}


def paired(a: dict, b: dict, seed_off: int = 0) -> dict:
    """Paired difference a - b over the snippet_ids present in BOTH, refusing a silent mismatch."""
    keys = sorted(set(a) & set(b))
    if not keys:
        return {"mean": float("nan"), "ci95": [float("nan")] * 2, "n": 0, "n_common": 0}
    out = boot([a[k] - b[k] for k in keys], seed_off)
    out["n_common"] = len(keys)
    out["n_a_only"] = len(set(a) - set(b))
    out["n_b_only"] = len(set(b) - set(a))
    return out


def load_c3(rows_path: Path, layer: int = LAYER) -> dict[str, float]:
    """{snippet_id: dG_S_c3_L<layer>} from a gate_rows.jsonl."""
    key = f"dG_S_c3_L{layer}"
    out: dict[str, float] = {}
    for line in open(rows_path):
        r = json.loads(line)
        if key in r:
            out[r["snippet_id"]] = float(r[key])
    if not out:
        raise SystemExit(f"{rows_path}: no {key} — wrong layer or wrong root?")
    return out


def load_swap(rows_path: Path, layer: int = LAYER) -> dict[str, float]:
    key = f"dG_S_swap_L{layer}"
    return {json.loads(l)["snippet_id"]: float(json.loads(l)[key])
            for l in open(rows_path) if key in json.loads(l)}


def check_extract(norms_path: Path) -> dict:
    """Pre-flight: re-extracted activations must reproduce the banked L7 norms (prereg gate)."""
    d = json.loads(Path(norms_path).read_text())
    s = d["layers"][str(LAYER)]
    inj_ok = int(s["injection_scale"]) == BANKED_INJECTION_SCALE
    mean_ok = abs(float(s["mean_av_train"]) - BANKED_MEAN_AV_TRAIN) <= MEAN_AV_TOL
    return {"injection_scale": s["injection_scale"], "injection_scale_expected": BANKED_INJECTION_SCALE,
            "mean_av_train": s["mean_av_train"], "mean_av_train_expected": BANKED_MEAN_AV_TRAIN,
            "n_rows": d.get("n_rows"), "passes": bool(inj_ok and mean_ok)}


def verdict(step_hi: dict, step_lo: dict) -> str:
    """The frozen H-C4 table. `step_hi` = 100%-50%, `step_lo` = 50%-25%, both paired on S_c3."""
    def clears(d: dict) -> bool:
        return bool(d["mean"] >= STEP_NATS and d["ci95"][0] > 0)
    if clears(step_hi):
        return "DATA-LIMITED"
    if clears(step_lo):
        return "DATA-SATURATED"
    return "DOSE-INSENSITIVE"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows-25", required=True)
    ap.add_argument("--rows-50", required=True)
    ap.add_argument("--rows-100", required=True, help="the RE-TRAINED 100% control")
    ap.add_argument("--rows-banked", required=True, help="the banked 100% pair's gate rows")
    ap.add_argument("--norms", default=None, help="re-extracted acts/norms.json for the pre-flight gate")
    ap.add_argument("--checks", nargs="*", default=[], help="check.json per dose point, for H-C6")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    st: dict = {"experiment": "C4_dose_response", "layer": LAYER, "seed": SEED, "n_boot": N_BOOT,
                "step_nats": STEP_NATS, "summary": []}

    if args.norms:
        st["extract_preflight"] = check_extract(Path(args.norms))
        if not st["extract_preflight"]["passes"]:
            st["verdict"] = "DOSE-EXTRACT-MISMATCH"
            st["reportable"] = False
            st["summary"].append(f"PRE-FLIGHT FAILED: {st['extract_preflight']}")
            Path(args.out).write_text(json.dumps(st, indent=1))
            print("\n".join(st["summary"]))
            return 3

    c3 = {k: load_c3(Path(v)) for k, v in (("25", args.rows_25), ("50", args.rows_50),
                                          ("100", args.rows_100), ("banked", args.rows_banked))}
    st["S_c3"] = {k: boot(list(v.values()), i) for i, (k, v) in enumerate(c3.items())}
    sw = load_swap(Path(args.rows_banked))
    st["S_swap_banked"] = boot(list(sw.values()), 9)

    # fidelity ratio per dose point, reported as the interpretable summary (NOT the test statistic)
    swap_mean = st["S_swap_banked"]["mean"]
    st["fidelity_ratio"] = {k: (st["S_c3"][k]["mean"] / swap_mean if swap_mean else float("nan"))
                            for k in c3}

    step_hi = paired(c3["100"], c3["50"], 11)
    step_lo = paired(c3["50"], c3["25"], 12)
    st["H_C4"] = {"step_100_minus_50": step_hi, "step_50_minus_25": step_lo,
                  "rule": {"step_nats": STEP_NATS}, "verdict": verdict(step_hi, step_lo)}

    ctrl = paired(c3["100"], c3["banked"], 13)
    drift = bool(abs(ctrl["mean"]) > CONTROL_DRIFT_NATS)
    st["H_C5"] = {"retrained_minus_banked": ctrl, "flag_nats": CONTROL_DRIFT_NATS,
                  "flagged": drift, "note": "DOSE-CONTROL-DRIFT" if drift else "within tolerance"}
    st["reportable"] = True
    st["provisional"] = drift

    if args.checks:                                   # H-C6, descriptive
        # Keyed by the dose ROOT directory (f025/f050/f100), not by anything inside check.json:
        # check.json carries `layer` but not `train_frac`, so keying on its contents would collapse
        # all three dose points onto the key "7" and silently keep only the last.
        prox = {}
        for p in args.checks:
            pp = Path(p)
            d = json.loads(pp.read_text())
            prox[pp.parent.parent.name] = {
                "ar_fve": (d.get("ar_eval") or {}).get("holdout_fve"),
                "ar_n_holdout": (d.get("ar_eval") or {}).get("n_holdout"),
                "av_gap": (d.get("av_eval") or {}).get("holdout_gap_permuted_minus_real"),
                "av_n_holdout": (d.get("av_eval") or {}).get("n_holdout"),
                "cos_cycle_mean": d.get("cos_cycle_mean"), "n_no_tags": d.get("n_no_tags")}
        st["H_C6"] = prox
        st["summary"].append("H-C6 proxies: " + " · ".join(
            f"{k} fve {(v['ar_fve'] or float('nan')):.4f} gap {(v['av_gap'] or float('nan')):+.4f}"
            for k, v in sorted(prox.items())))

    st["summary"] += [
        "S_c3 by dose: " + " · ".join(f"{k}% {st['S_c3'][k]['mean']:+.2f}" if k != "banked"
                                      else f"banked {st['S_c3'][k]['mean']:+.2f}" for k in c3),
        "fidelity ratio: " + " · ".join(f"{k} {st['fidelity_ratio'][k]:.3f}" for k in c3),
        f"H-C4 {st['H_C4']['verdict']}: 100-50 = {step_hi['mean']:+.2f} {step_hi['ci95']} · "
        f"50-25 = {step_lo['mean']:+.2f} {step_lo['ci95']} (step bar {STEP_NATS:+.2f})",
        f"H-C5 control: retrained - banked = {ctrl['mean']:+.2f} {ctrl['ci95']} -> {st['H_C5']['note']}",
    ]
    Path(args.out).write_text(json.dumps(st, indent=1))
    print("\n".join(st["summary"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
