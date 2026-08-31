"""P1b — depth for a READ: a per-layer dense probe on the residual stream.

Pre-registration: log/nla-harness/2026-08-30_p1b-read-depth-prereg.md. Every constant here is
quoted from it.

P0.4 showed layer 13 is not a better place to WRITE (the oracle nets 0.0000 there). P0.1 said the
task direction is most coherent at 13. This asks whether it is a better place to READ, with the
dense linear decoder the project charter requires as the baseline for any interpretability claim.

TWO THINGS THAT WOULD SILENTLY FAKE A RESULT, AND WHAT IS DONE ABOUT THEM.
  1. Each snippet contributes an L0 and an L1b vector. Ungrouped CV puts both in train and test,
     so the probe memorises the snippet rather than the tier and AUC runs toward 1.0. Folds are
     therefore GroupKFold on snippet_id. A label-permutation control is run through the identical
     pipeline and must land near 0.5.
  2. P0.1 found relative magnitude peaks at exactly layer 20. A probe that only tracks activation
     size would manufacture a depth effect out of scale, so a norm-only probe is fitted at every
     layer and the semantic claim has to beat it.

Bootstrap resampling is over SNIPPETS, not vectors: a snippet's two tiers are not independent.

Reads are bit-exact (2026-08-30), so this measurement carries no reproducibility caveat.
Env `nla-mi`, one GPU, ~10 min.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_NLA_ROOT = _HERE.parent
_PROJ = _NLA_ROOT.parent
sys.path.insert(0, str(_NLA_ROOT / "vendor" / "nla-repo"))
sys.path.insert(0, str(_HERE))

SEED = 20260724
C_FROZEN = 1.0          # frozen, not tuned — see the prereg
N_FOLDS = 5
N_BOOT = 2000
DELTA = 0.05            # the pre-registered contrast threshold
PERM_MAX = 0.60         # permutation control must sit below this
CEILING = 0.95          # AUC at every layer above this => UNINFORMATIVE
INSTRUMENT_LAYER, TEST_LAYER = 20, 13


def grouped_cv_auc(X, y, groups, seed=SEED):
    """Out-of-fold AUC with folds grouped so a snippet never spans train and test."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    oof = np.zeros(len(y), dtype=float)
    for tr, te in GroupKFold(n_splits=N_FOLDS).split(X, y, groups):
        sc = StandardScaler().fit(X[tr])
        m = LogisticRegression(C=C_FROZEN, max_iter=5000).fit(sc.transform(X[tr]), y[tr])
        oof[te] = m.predict_proba(sc.transform(X[te]))[:, 1]
    return float(roc_auc_score(y, oof)), oof


def boot_ci_diff(y, oof_a, oof_b, groups, seed=SEED):
    """CI on AUC(a) - AUC(b), resampling SNIPPETS so paired tiers move together."""
    from sklearn.metrics import roc_auc_score
    rng = random.Random(seed)
    uniq = sorted(set(groups))
    idx_by_g = {g: [i for i, gg in enumerate(groups) if gg == g] for g in uniq}
    out = []
    for _ in range(N_BOOT):
        pick = [idx_by_g[uniq[rng.randrange(len(uniq))]] for _ in range(len(uniq))]
        idx = [i for chunk in pick for i in chunk]
        yy = y[idx]
        if len(set(yy.tolist())) < 2:
            continue
        out.append(roc_auc_score(yy, oof_a[idx]) - roc_auc_score(yy, oof_b[idx]))
    out.sort()
    return [round(out[int(0.025 * len(out))], 4), round(out[int(0.975 * len(out))], 4)] \
        if out else [None, None]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=str(_PROJ / "data/nla/p0/p1b/read_probe.json"))
    ap.add_argument("--acts-out", default=str(_PROJ / "data/nla/p0/p1b/acts.npz"))
    args = ap.parse_args()

    import torch
    from layer_rotation import all_layer_acts
    from steer_run import TARGET_MODEL, build_user, load_pairs
    from transformers import AutoModelForCausalLM, AutoTokenizer

    pairs = load_pairs(args.limit, random.Random(SEED))
    tokz = AutoTokenizer.from_pretrained(TARGET_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        TARGET_MODEL, torch_dtype=torch.bfloat16, device_map=args.device).eval()

    rows, y_tier, y_corr, groups, corr_mask = [], [], [], [], []
    # Correctness labels come from the P0.4 L20 baseline — unsteered greedy generations on these
    # exact items. Reusing them rather than regenerating keeps this a pure read-side measurement.
    base_p = _PROJ / "data/nla/p0/p04/L20/baseline.jsonl"
    base = {json.loads(l)["snippet_id"]: json.loads(l)
            for l in open(base_p) if l.strip()} if base_p.exists() else {}

    for i, p in enumerate(pairs, 1):
        for tier, code, call in (("L0", p["code_l0"], p["call_l0"]),
                                 ("L1b", p["code_l1b"], p["call_l1b"])):
            rows.append(all_layer_acts(model, tokz, build_user(code, call)))
            y_tier.append(0 if tier == "L0" else 1)
            groups.append(p["snippet_id"])
            b = base.get(p["snippet_id"])
            keep = tier == "L1b" and b is not None
            corr_mask.append(keep)
            y_corr.append(int(b["l1b_correct"]) if keep else -1)
        if i % 20 == 0:
            print(f"[p1b] {i}/{len(pairs)} pairs extracted", flush=True)

    A = np.stack(rows)                      # [n_vectors, n_layers, d]
    y_tier = np.array(y_tier)
    y_corr = np.array(y_corr)
    corr_mask = np.array(corr_mask)
    groups = np.array(groups)
    n_layers = A.shape[1]
    Path(args.acts_out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.acts_out, acts=A, y_tier=y_tier, y_corr=y_corr,
                        corr_mask=corr_mask, groups=groups)
    print(f"[p1b] acts {A.shape} -> {args.acts_out}", flush=True)

    rep = {"experiment": "p1b_read_depth_probe",
           "prereg": "log/nla-harness/2026-08-30_p1b-read-depth-prereg.md",
           "model": TARGET_MODEL, "seed": SEED, "C": C_FROZEN, "folds": N_FOLDS,
           "n_vectors": int(A.shape[0]), "n_layers": int(n_layers), "d_model": int(A.shape[2]),
           "cv": "GroupKFold on snippet_id", "layers": []}

    oof_store: dict[tuple[str, int], np.ndarray] = {}
    for L in range(n_layers):
        X = A[:, L, :]
        auc_full, oof = grouped_cv_auc(X, y_tier, groups)
        auc_norm, _ = grouped_cv_auc(np.linalg.norm(X, axis=1, keepdims=True), y_tier, groups)
        rng = np.random.default_rng(SEED + L)
        auc_perm, _ = grouped_cv_auc(X, rng.permutation(y_tier), groups)
        oof_store[("tier", L)] = oof

        Xc, yc, gc = X[corr_mask], y_corr[corr_mask], groups[corr_mask]
        auc_c, oof_c = grouped_cv_auc(Xc, yc, gc)
        oof_store[("corr", L)] = oof_c

        rep["layers"].append({
            "layer": L,
            "tier_auc": round(auc_full, 4),
            "tier_auc_norm_only": round(auc_norm, 4),
            "tier_auc_permuted": round(auc_perm, 4),
            "correctness_auc": round(auc_c, 4),
        })
        print(f"[p1b] L{L:02d} tier {auc_full:.4f} (norm {auc_norm:.4f}, perm {auc_perm:.4f}) "
              f"· correctness {auc_c:.4f}", flush=True)

    lay = {r["layer"]: r for r in rep["layers"]}
    # ── the frozen rules ──────────────────────────────────────────────────────
    perm_max = max(r["tier_auc_permuted"] for r in rep["layers"])
    all_ceiling = all(r["tier_auc"] >= CEILING for r in rep["layers"])
    d_tier = lay[TEST_LAYER]["tier_auc"] - lay[INSTRUMENT_LAYER]["tier_auc"]
    ci_tier = boot_ci_diff(y_tier, oof_store[("tier", TEST_LAYER)],
                           oof_store[("tier", INSTRUMENT_LAYER)], groups)
    beats_norm = lay[TEST_LAYER]["tier_auc"] - lay[TEST_LAYER]["tier_auc_norm_only"]

    if perm_max > PERM_MAX:
        verdict = f"NOT REPORTABLE — permutation control reached {perm_max} (> {PERM_MAX})"
    elif all_ceiling:
        verdict = "UNINFORMATIVE — every layer is above the ceiling; depths cannot be ranked"
    elif d_tier >= DELTA and ci_tier[0] is not None and ci_tier[0] > 0 and beats_norm >= DELTA:
        verdict = "L13 IS A BETTER READ SITE"
    else:
        verdict = "NO READ-DEPTH ADVANTAGE"

    yc = y_corr[corr_mask]
    d_corr = lay[TEST_LAYER]["correctness_auc"] - lay[INSTRUMENT_LAYER]["correctness_auc"]
    rep["primary_H_R1"] = {
        "delta_auc_L13_minus_L20": round(d_tier, 4), "ci95": ci_tier, "threshold": DELTA,
        "L13_beats_norm_only_by": round(beats_norm, 4),
        "permutation_max": perm_max, "all_layers_at_ceiling": all_ceiling,
        "verdict": verdict,
    }
    rep["secondary_H_R2_correctness"] = {
        "delta_auc_L13_minus_L20": round(d_corr, 4),
        "ci95": boot_ci_diff(yc, oof_store[("corr", TEST_LAYER)],
                             oof_store[("corr", INSTRUMENT_LAYER)], groups[corr_mask]),
        "n": int(corr_mask.sum()), "n_positive": int((yc == 1).sum()),
        "caveat": "underpowered at n=60; a null here is not evidence of absence",
    }
    rep["H_R3_curve"] = {
        "argmax_tier_layer": max(rep["layers"], key=lambda r: r["tier_auc"])["layer"],
        "argmax_tier_auc": max(r["tier_auc"] for r in rep["layers"]),
        "argmax_correctness_layer": max(rep["layers"],
                                        key=lambda r: r["correctness_auc"])["layer"],
        "argmax_correctness_auc": max(r["correctness_auc"] for r in rep["layers"]),
    }
    rep["finished_utc"] = datetime.now(timezone.utc).isoformat()
    Path(args.out).write_text(json.dumps(rep, indent=2))
    print(f"\n[p1b] PRIMARY: {verdict}")
    print(f"[p1b] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
