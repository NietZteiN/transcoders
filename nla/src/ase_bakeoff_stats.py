"""H-R2 scoring: the steering bake-off's frozen contrasts, from the per-arm jsonl files of ase_steer_run.py.

Pre-registered: log/nla-harness/2026-09-14_steering-bakeoff-prereg.md (+ arms and residual amendments).

Primary statistic is Pass@1 case-weighted by THEIR definition: sum_i p_i * n_i / sum_i n_i over snippets i.
Every contrast is PAIRED per snippet (both arms restricted to the snippets they share) and bootstrapped
over snippets (cluster bootstrap, N_BOOT 10 000, seed 20260724), resampling snippets and recomputing the
case-weighted difference each draw. Nothing here pools cases across snippets as if independent.

  H-R2a  best-of-ours (erasure, prompt; `knockout` not built) - codesteer, 95 % CI AND a Bonferroni alpha/3
         interval (the prereg fixed 3 arms); the verdict reads on the adjusted one.
  H-R2b  codesteer - rand_prior  ->  SLICE-MATTERS / SLICE-IRRELEVANT.
  H-R2c  restoration (arm - unsteered) / (original - unsteered) per steered arm, bootstrapped as a ratio of
         case-weighted means; `original` is the H-R1 original-condition run restricted to the same snippets.
Every other arm (uniform_prior, swap_oracle, foreign, combined) is reported paired against unsteered with
its pre-declared reading from the amendments; none enters best-of-ours.

BETA-FAULT AMENDMENT (2026-09-14, log/nla-harness/2026-09-14_bakeoff-beta-fault.md): the paper's method
has two readings -- README-exact `codesteer` (all heads in the last 8 layers) and `codesteer_auto` (the
calibrated sparse head subset of Eq. 10). H-R2a's comparator is the BETTER of the two by Pass@1
(conservative for us); H-R2b uses `codesteer`. The identity runs `codesteer_beta0` / `rand_prior_beta0`
(beta_post=0, steering a no-op) are reported paired against unsteered as the NOISE-FLOOR controls: they
measure the split-prefill code path + sampling variance and enter no verdict.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

N_BOOT, SEED = 10_000, 20260724
OURS = ("erasure", "prompt")            # H-R2a's best-of-ours; `knockout` was never built for Llama
STEERED = ("codesteer", "codesteer_auto", "rand_prior", "uniform_prior", "prompt", "swap_oracle", "foreign",
           "erasure", "combined", "ridge_map", "role_proto", "prompt_types")
# H-R6 (ASE) contrasts, prereg log/nla-harness/2026-09-14_better-vector-prereg.md; H-R2a's OURS is
# frozen and does NOT take the H-R6 arms -- they read on their own three rules below.
R6_MIN = 0.05
CONTROLS = ("codesteer_beta0", "rand_prior_beta0")     # identity steering: the noise floor, no verdict
MATCH_TOL = 0.05


def load(path: Path) -> dict[str, tuple[float, int]]:
    """{snippet: (pass@1, n_cases)}; refuses duplicate snippets (a resumed run that double-wrote)."""
    out: dict[str, tuple[float, int]] = {}
    for l in open(path):
        if not l.strip():
            continue
        r = json.loads(l)
        if r["snippet"] in out:
            raise SystemExit(f"{path}: duplicate snippet {r['snippet']}")
        out[r["snippet"]] = (float(r["pass@1"]), int(r["n_cases"]))
    return out


def cw(p: np.ndarray, n: np.ndarray) -> float:
    return float((p * n).sum() / n.sum())


def paired(a: dict, b: dict, seed_off: int, alpha_levels=(0.05, 0.05 / 3)) -> dict:
    """Case-weighted (a - b) over shared snippets with bootstrap CIs at each alpha."""
    keys = sorted(set(a) & set(b))
    if not keys:
        return {"n": 0, "mean": float("nan")}
    pa = np.array([a[k][0] for k in keys]); pb = np.array([b[k][0] for k in keys])
    n = np.array([a[k][1] for k in keys], dtype=float)
    if any(a[k][1] != b[k][1] for k in keys):
        raise SystemExit("case counts differ between arms on shared snippets: packs are not the same")
    rng = np.random.default_rng(SEED + seed_off)
    idx = rng.integers(0, len(keys), size=(N_BOOT, len(keys)))
    boots = np.sort(((pa - pb)[idx] * n[idx]).sum(1) / n[idx].sum(1))
    out = {"n": len(keys), "mean": cw(pa - pb, n), "a": cw(pa, n), "b": cw(pb, n)}
    for al in alpha_levels:
        lo, hi = boots[int(al / 2 * N_BOOT)], boots[int((1 - al / 2) * N_BOOT) - 1]
        out[f"ci{100 * (1 - al):.4g}"] = [float(lo), float(hi)]
    return out


def restoration(arm: dict, uns: dict, orig: dict, seed_off: int) -> dict:
    keys = sorted(set(arm) & set(uns) & set(orig))
    if not keys:
        return {"n": 0}
    A = np.array([arm[k][0] for k in keys]); U = np.array([uns[k][0] for k in keys])
    O = np.array([orig[k][0] for k in keys]); n = np.array([arm[k][1] for k in keys], dtype=float)
    rng = np.random.default_rng(SEED + seed_off)
    idx = rng.integers(0, len(keys), size=(N_BOOT, len(keys)))
    num = ((A - U)[idx] * n[idx]).sum(1) / n[idx].sum(1)
    den = ((O - U)[idx] * n[idx]).sum(1) / n[idx].sum(1)
    ratio = np.sort(num / np.where(np.abs(den) < 1e-9, np.nan, den))
    ratio = ratio[~np.isnan(ratio)]
    point = cw(A - U, n) / cw(O - U, n) if abs(cw(O - U, n)) > 1e-9 else float("nan")
    return {"n": len(keys), "ratio": point, "ci95": [float(ratio[int(.025 * len(ratio))]),
                                                    float(ratio[int(.975 * len(ratio)) - 1])],
            "arm": cw(A, n), "unsteered": cw(U, n), "original": cw(O, n)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", required=True, help="bakeoff dir with <arm>.jsonl")
    ap.add_argument("--original", required=True, help="H-R1 original-condition rows (gate/original.jsonl)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    D = Path(args.dir)
    known = ("unsteered",) + STEERED + CONTROLS
    arms = {p.stem: load(p) for p in sorted(D.glob("*.jsonl")) if p.stem in known}
    if (skipped := [p.name for p in D.glob("*.jsonl") if p.stem not in known]):
        print(f"[stats] ignoring non-arm files in {D}: {skipped}")
    orig = load(Path(args.original))
    if "unsteered" not in arms:
        raise SystemExit("no unsteered.jsonl: every contrast is paired against it")
    uns = arms["unsteered"]
    st: dict = {"experiment": "R2_bakeoff", "n_boot": N_BOOT, "seed": SEED, "arms_present": sorted(arms),
                "n_snippets": {k: len(v) for k, v in arms.items()}, "summary": []}
    st["case_weighted_pass1"] = {k: cw(np.array([x[0] for x in v.values()]),
                                       np.array([x[1] for x in v.values()], dtype=float)) for k, v in arms.items()}
    st["original_on_subset"] = cw(np.array([orig[k][0] for k in uns if k in orig]),
                                  np.array([orig[k][1] for k in uns if k in orig], dtype=float))
    # every arm vs unsteered
    st["vs_unsteered"] = {a: paired(arms[a], uns, i) for i, a in enumerate(sorted(arms)) if a != "unsteered"}
    st["restoration"] = {a: restoration(arms[a], uns, orig, 100 + i)
                         for i, a in enumerate(STEERED) if a in arms}
    # H-R2b
    if "codesteer" in arms and "rand_prior" in arms:
        d = paired(arms["codesteer"], arms["rand_prior"], 200)
        st["H_R2b"] = {**d, "verdict": "SLICE-MATTERS" if d["mean"] > 0 and d["ci95"][0] > 0 else "SLICE-IRRELEVANT"}
    # H-R2a -- comparator = the better of the paper's two readings (beta-fault amendment)
    ours = [a for a in OURS if a in arms]
    theirs = [a for a in ("codesteer", "codesteer_auto") if a in arms]
    if theirs and ours:
        best = max(ours, key=lambda a: st["case_weighted_pass1"][a])
        comp = max(theirs, key=lambda a: st["case_weighted_pass1"][a])
        d = paired(arms[best], arms[comp], 300)
        lo, hi = d["ci98.33"]
        if d["mean"] > 0 and lo > 0:
            v = "SURPASS-CODESTEER"
        elif d["mean"] < 0 and hi < 0:
            v = "BELOW-CODESTEER"
        elif abs(d["mean"]) < MATCH_TOL:
            v = "MATCH-CODESTEER"
        else:
            v = "INCONCLUSIVE"
        st["H_R2a"] = {"best_of_ours": best, "candidates": ours, "comparator": comp, "comparators": theirs,
                       **d, "verdict": v, "note": "verdict reads on the Bonferroni alpha/3 interval (ci98.33)"}
    if "codesteer" in arms and "codesteer_auto" in arms:
        st["auto_minus_none"] = paired(arms["codesteer_auto"], arms["codesteer"], 400)
    # H-R6 -- oracle-free meaning-installing vectors (each rule reads only if both arms exist)
    r6 = {}
    if "ridge_map" in arms and "erasure" in arms:
        d = paired(arms["ridge_map"], arms["erasure"], 500)
        r6["a"] = {**d, "verdict": "MAP-BEATS-MEAN" if d["mean"] >= R6_MIN and d["ci95"][0] > 0 else "MAP-NOT-BETTER"}
    elif "erasure" in arms and "ridge_map" not in arms:
        r6["a"] = {"verdict": "MAP-LEARNS-NOTHING-OR-UNRUN"}
    if "role_proto" in arms and "foreign" in arms:
        d = paired(arms["role_proto"], arms["foreign"], 510)
        r6["b"] = {**d, "verdict": "CATEGORY-MEANING-HELPS" if d["mean"] >= R6_MIN and d["ci95"][0] > 0
                   else "CATEGORY-MEANING-INERT"}
    lat = [a for a in ("ridge_map", "role_proto") if a in arms]
    if lat and "prompt_types" in arms:
        bl = max(lat, key=lambda a: st["case_weighted_pass1"][a])
        d = paired(arms[bl], arms["prompt_types"], 520)
        r6["c"] = {**d, "best_latent": bl,
                   "verdict": "LATENT-BEATS-PROMPT" if d["mean"] > 0 and d["ci95"][0] > 0 else "PROMPT-SUFFICES"}
    if r6:
        st["H_R6"] = r6
    # summary
    st["summary"].append(f"original(subset) {st['original_on_subset']:.4f} · " + " · ".join(
        f"{a} {st['case_weighted_pass1'][a]:.4f}" for a in ["unsteered"] + [x for x in STEERED if x in arms]))
    for a, d in st["vs_unsteered"].items():
        st["summary"].append(f"{a} - unsteered = {d['mean']:+.4f} {np.round(d['ci95'], 4).tolist()} n={d['n']}")
    for a, d in st["restoration"].items():
        if d.get("n"):
            st["summary"].append(f"restoration {a}: {100 * d['ratio']:.1f} % [{100 * d['ci95'][0]:.1f}, {100 * d['ci95'][1]:.1f}]")
    if "H_R2b" in st:
        st["summary"].append(f"H-R2b {st['H_R2b']['verdict']}: codesteer - rand_prior = {st['H_R2b']['mean']:+.4f} {np.round(st['H_R2b']['ci95'], 4).tolist()}")
    if "H_R2a" in st:
        st["summary"].append(f"H-R2a {st['H_R2a']['verdict']}: {st['H_R2a']['best_of_ours']} - {st['H_R2a']['comparator']} = "
                             f"{st['H_R2a']['mean']:+.4f} 95% {np.round(st['H_R2a']['ci95'], 4).tolist()} "
                             f"adj {np.round(st['H_R2a']['ci98.33'], 4).tolist()}")
    if "auto_minus_none" in st:
        d = st["auto_minus_none"]
        st["summary"].append(f"codesteer_auto - codesteer = {d['mean']:+.4f} {np.round(d['ci95'], 4).tolist()}")
    for k, lbl in (("a", "ridge_map - erasure"), ("b", "role_proto - foreign"), ("c", "best latent - prompt_types")):
        if k in st.get("H_R6", {}):
            d = st["H_R6"][k]
            st["summary"].append(f"H-R6{k} {d['verdict']}" + (f": {lbl} = {d['mean']:+.4f} {np.round(d['ci95'], 4).tolist()}"
                                                              if "mean" in d else ""))
    for a in CONTROLS:
        if a in st["vs_unsteered"]:
            d = st["vs_unsteered"][a]
            st["summary"].append(f"noise floor {a} - unsteered = {d['mean']:+.4f} {np.round(d['ci95'], 4).tolist()} (no verdict)")
    Path(args.out).write_text(json.dumps(st, indent=1))
    print("\n".join(st["summary"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
