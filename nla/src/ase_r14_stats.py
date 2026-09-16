"""H-R14 scoring: the FULL aligned HumanEval-X corpus (148 snippets / 1,742 cases) against the
pre-registered H-R14a-e rules and against the paper's own Table 2 / Table 4 rows.

Pre-registered in log/nla-harness/2026-09-16_full-corpus-prereg.md. Thresholds are read from that entry
and hard-coded here; nothing is tunable from the CLI, so a verdict cannot be moved after seeing data.

WHY A SEPARATE FILE FROM `ase_bakeoff_stats.py`. That scorer's outputs are already published (the
50-snippet bake-off report and the H-R7 entry), so it is left byte-identical -- re-running it must keep
reproducing those numbers. Every primitive is imported from it rather than reimplemented: `load` (which
also enforces "no duplicate snippet", the resumed-run trap), `load_truth`, `paired` (the cluster
bootstrap, and the guard that refuses a contrast whose arms disagree on case counts), `restoration`,
`cw`, `N_BOOT`, `SEED`. Only the H-R14 decision rules are new.

TWO DIFFERENCES FROM H-R7's RULES, BOTH PRE-REGISTERED:
  * H-R14c adds a MAGNITUDE requirement to `NLA-BEATS-CODESTEER` (>= +0.05 AND the alpha/3 interval
    excluding 0). H-R2a's `SURPASS-CODESTEER` read on the interval alone, which at n=148 could return
    "surpass" on a +0.01 effect that is statistically clean and practically nothing.
  * The restoration ratio is GATED on DAMAGE-PRESENT. Its denominator is the damage; at DAMAGE-WEAK the
    bootstrap ratio is a heavy-tailed artefact of a near-zero denominator, not an estimate. The paper's
    104.99 % / 107.1 % rest on a 36.3-point denominator, so the comparison is only meaningful if ours is
    real too.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ase_bakeoff_stats import (N_BOOT, SEED, cw, load, load_truth, paired,  # noqa: E402
                               restoration)

TAG = "[R14]"
DAMAGE_MIN = 0.03        # H-R14a
EFFECT_MIN = 0.05        # H-R14b/c/d, and MATCH_TOL for the inert/match bands
ARMS = ("unsteered", "codesteer", "codesteer_auto", "ridge_map", "erasure", "prompt", "swap_oracle")

# The paper's own rows, for the H-R14e cross-setup comparison. Qwen2.5-7B; percentages as printed.
# Table 2 = aggregate over all 4 obfuscations; Table 4 = the identifier-renaming stratum (ours).
PAPER = {
    "humaneval": {"table2": {"orig": 76.49, "obf": 64.01, "steer": 77.38, "restor": 107.1},
                  "table4_renaming": {"orig": 76.49, "obf": 40.20, "steer": 78.30, "restor": 104.99}},
    "cruxeval": {"table2": {"orig": 83.11, "obf": 71.40, "steer": 80.43, "restor": 77.1},
                 "table4_renaming": {"orig": 83.11, "obf": 74.51, "steer": 85.48, "restor": 127.50}},
}


def ci_at(a: dict, b: dict, seed_off: int, alpha: float) -> list[float]:
    """Paired case-weighted (a-b) interval at an arbitrary alpha (for the Bonferroni readings)."""
    keys = sorted(set(a) & set(b))
    x = np.array([a[k][0] - b[k][0] for k in keys])
    n = np.array([a[k][1] for k in keys], dtype=float)
    rng = np.random.default_rng(SEED + seed_off)
    idx = rng.integers(0, len(keys), size=(N_BOOT, len(keys)))
    boots = np.sort((x[idx] * n[idx]).sum(1) / n[idx].sum(1))
    return [float(boots[int(alpha / 2 * N_BOOT)]), float(boots[int((1 - alpha / 2) * N_BOOT) - 1])]


def band(mean: float, ci: list[float], pos: str, neg: str, mid: str, thr: float = EFFECT_MIN) -> str:
    if mean >= thr and ci[0] > 0:
        return pos
    if mean <= -thr and ci[1] < 0:
        return neg
    if abs(mean) < thr:
        return mid
    return "UNRESOLVED"


def parse_rate(path: Path) -> float:
    rows = [json.loads(l) for l in open(path) if l.strip()]
    n = sum(r["n_cases"] for r in rows)
    return float(sum(r["parsed_frac"] * r["n_cases"] for r in rows) / n) if n else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", required=True, help="bakeoff_full dir with <arm>.jsonl")
    ap.add_argument("--original", required=True, help="original_unsteered.jsonl (the L0 condition)")
    ap.add_argument("--packs", nargs="+", required=True, help="case packs carrying expected_bool")
    ap.add_argument("--dataset", default="humaneval", choices=("humaneval", "cruxeval"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    truth = load_truth(args.packs)
    D = Path(args.dir)
    st: dict = {"experiment": "R14_full_corpus", "dataset": args.dataset, "n_boot": N_BOOT, "seed": SEED,
                "thresholds": {"damage_min": DAMAGE_MIN, "effect_min": EFFECT_MIN},
                "prereg": "log/nla-harness/2026-09-16_full-corpus-prereg.md", "scorings": {}}

    for scoring in ("cn", "pass1"):
        arms = {p.stem: load(p, scoring, truth) for p in sorted(D.glob("*.jsonl"))
                if p.stem in ARMS}
        missing = [a for a in ARMS if a not in arms]
        orig = load(Path(args.original), scoring, truth)
        uns = arms["unsteered"]
        # Every arm must cover the SAME snippets; paired() would otherwise silently intersect and the
        # arms would be compared on different corpora.
        sets = {a: set(v) for a, v in arms.items()} | {"original_unsteered": set(orig)}
        common = set.intersection(*sets.values())
        S: dict = {"arms_present": sorted(arms), "arms_missing": missing,
                   "n_snippets": {a: len(v) for a, v in sets.items()},
                   "n_common_snippets": len(common),
                   "coverage_equal": all(len(v) == len(common) for v in sets.values())}
        acc = {a: cw(np.array([v[k][0] for k in common]),
                     np.array([v[k][1] for k in common], dtype=float)) for a, v in arms.items()}
        acc["original_unsteered"] = cw(np.array([orig[k][0] for k in common]),
                                       np.array([orig[k][1] for k in common], dtype=float))
        S["accuracy"] = acc
        S["n_cases"] = int(sum(uns[k][1] for k in common))
        S["parse_rate"] = {a: parse_rate(D / f"{a}.jsonl") for a in arms}
        S["parse_rate"]["original_unsteered"] = parse_rate(Path(args.original))
        S["vs_unsteered"] = {a: paired(arms[a], uns, i) for i, a in enumerate(sorted(arms)) if a != "unsteered"}

        # ---- H-R14a: damage ----
        dmg = paired(orig, uns, 700)
        dv = ("DAMAGE-PRESENT" if dmg["mean"] >= DAMAGE_MIN and dmg["ci95"][0] > 0
              else "DAMAGE-WEAK" if dmg["mean"] >= DAMAGE_MIN else "DAMAGE-ABSENT")
        S["H_R14a"] = {**dmg, "verdict": dv}

        # ---- H-R14b: their arms, Bonferroni alpha/2 over the two CodeSteer readings ----
        b: dict = {}
        for i, a in enumerate(("codesteer", "codesteer_auto")):
            if a in arms:
                d = paired(arms[a], uns, 710 + i)
                d["ci97.5"] = ci_at(arms[a], uns, 720 + i, 0.05 / 2)
                d["verdict"] = band(d["mean"], d["ci97.5"], "CODESTEER-RESTORES", "CODESTEER-HARMS",
                                    "CODESTEER-INERT")
                b[a] = d
        if b:
            vs = [d["verdict"] for d in b.values()]
            S["H_R14b"] = {**b, "verdict": ("CODESTEER-RESTORES" if "CODESTEER-RESTORES" in vs
                                            else "CODESTEER-HARMS" if "CODESTEER-HARMS" in vs
                                            else "CODESTEER-INERT" if all(v == "CODESTEER-INERT" for v in vs)
                                            else "CODESTEER-UNRESOLVED")}

        # ---- H-R14c: ours vs theirs, comparator = the BETTER CodeSteer arm, alpha/3 ----
        theirs = [a for a in ("codesteer", "codesteer_auto") if a in arms]
        if "ridge_map" in arms and theirs:
            comp = max(theirs, key=lambda a: acc[a])
            d = paired(arms["ridge_map"], arms[comp], 300)
            d["ci98.33"] = ci_at(arms["ridge_map"], arms[comp], 310, 0.05 / 3)
            S["H_R14c"] = {"comparator": comp, "comparators": theirs, **d,
                           "verdict": band(d["mean"], d["ci98.33"], "NLA-BEATS-CODESTEER",
                                           "CODESTEER-BEATS-NLA", "NLA-MATCHES-CODESTEER")}

        # ---- H-R14d: the prompting baseline, alpha/3 ----
        if "ridge_map" in arms and "prompt" in arms:
            d = paired(arms["ridge_map"], arms["prompt"], 320)
            d["ci98.33"] = ci_at(arms["ridge_map"], arms["prompt"], 330, 0.05 / 3)
            S["H_R14d"] = {**d, "verdict": ("LATENT-BEATS-PROMPT"
                                            if d["mean"] >= EFFECT_MIN and d["ci98.33"][0] > 0
                                            else "PROMPT-SUFFICES")}

        # ---- restoration ratios, GATED on DAMAGE-PRESENT (prereg) ----
        if dv == "DAMAGE-PRESENT":
            S["restoration"] = {a: restoration(arms[a], uns, orig, 100 + i)
                                for i, a in enumerate(sorted(arms)) if a != "unsteered"}
        else:
            S["restoration"] = {"withheld": f"H-R14a returned {dv}; a ratio over a denominator whose CI "
                                            f"contains 0 is not an estimate (prereg gate)"}

        # ---- H-R14e: cross-setup comparison to the paper (descriptive) ----
        p = PAPER[args.dataset]
        our_drop = 100.0 * dmg["mean"]
        their_drop = p["table4_renaming"]["orig"] - p["table4_renaming"]["obf"]
        ratio = our_drop / their_drop if their_drop else float("nan")
        S["H_R14e"] = {
            "ours_pct": {"orig": 100 * acc["original_unsteered"], "obf": 100 * acc["unsteered"],
                         **{f"steer_{a}": 100 * acc[a] for a in theirs},
                         **({"steer_ridge_map": 100 * acc["ridge_map"]} if "ridge_map" in arms else {})},
            "ours_drop_pts": our_drop, "their_drop_pts": their_drop, "drop_ratio_ours_over_theirs": ratio,
            "paper": p,
            "verdict": ("DAMAGE-FAR-WEAKER" if ratio < 1 / 3 else
                        "DAMAGE-COMPARABLE" if ratio <= 3 else "DAMAGE-FAR-STRONGER"),
            "caveat": "cross-SETUP, not a replication: their 4 models are all outside our permitted set, "
                      "their renaming generator and equivalence checker are not in the artifact, and only "
                      "1 of their 4 obfuscation families is testable here. A gap is un-attributable "
                      "between model, renamer and protocol.",
        }
        st["scorings"][scoring] = S

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(st, indent=1, default=str))

    # ---- console summary: primary scoring first ----
    for scoring in ("cn", "pass1"):
        S = st["scorings"][scoring]
        lab = "c/n (primary)" if scoring == "cn" else "Pass@1 (their statistic)"
        print(f"\n{TAG} ===== {lab} · {S['n_common_snippets']} snippets / {S['n_cases']} cases =====")
        if not S["coverage_equal"]:
            print(f"{TAG} WARNING: arms cover different snippet sets: {S['n_snippets']}")
        if S["arms_missing"]:
            print(f"{TAG} arms missing: {S['arms_missing']}")
        for a, v in sorted(S["accuracy"].items(), key=lambda kv: -kv[1]):
            d = S["vs_unsteered"].get(a)
            delta = f"  Δ vs unsteered {d['mean']:+.4f} [{d['ci95'][0]:+.4f},{d['ci95'][1]:+.4f}]" if d else ""
            print(f"  {a:20s} {v:.4f}  parse {S['parse_rate'][a]:.3f}{delta}")
        for h in ("H_R14a", "H_R14b", "H_R14c", "H_R14d", "H_R14e"):
            if h in S:
                x = S[h]
                extra = ""
                if h == "H_R14a":
                    extra = f" {x['mean']:+.4f} [{x['ci95'][0]:+.4f},{x['ci95'][1]:+.4f}]"
                elif h in ("H_R14c", "H_R14d"):
                    extra = f" {x['mean']:+.4f} ci98.33 [{x['ci98.33'][0]:+.4f},{x['ci98.33'][1]:+.4f}]"
                elif h == "H_R14e":
                    extra = (f" ours {x['ours_drop_pts']:+.2f} pts vs theirs {x['their_drop_pts']:+.2f} pts "
                             f"(ratio {x['drop_ratio_ours_over_theirs']:.3f})")
                print(f"  {h}: {x['verdict']}{extra}")
        r = S["restoration"]
        if "withheld" in r:
            print(f"  restoration: WITHHELD — {r['withheld']}")
        else:
            for a, v in sorted(r.items(), key=lambda kv: -(kv[1].get("ratio") or -9e9)):
                if v.get("n"):
                    print(f"  restoration {a:18s} {100*v['ratio']:+7.1f} % "
                          f"[{100*v['ci95'][0]:+.1f}, {100*v['ci95'][1]:+.1f}]")
    print(f"\n{TAG} wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
