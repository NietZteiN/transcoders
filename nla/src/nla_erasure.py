"""H-E1/H-E2/H-E3 -- the cross-item decoy-ERASURE vector, built with no autoencoder.

Pre-registered in log/nla-harness/2026-09-12_erasure-vector-prereg.md.

For span s of item i at layer K, with the banked clean state h0_s and decoy state h1b_s:

    Delta_s        = h0_s - h1b_s
    E_K^(-i)       = mean of Delta_s over every span NOT in item i          (leave-one-item-out)
    erase_alpha    : v_s = h1b_s + alpha * |Delta_s| * unit(E_K^(-i))
    erase_rand     : the same with a seeded random unit direction           (norm-matched control)
    erase_own      : v_s = h1b_s + |Delta_s| * unit(Delta_s) = h0_s         (== swap; identity gate)
    erase_flip     : the LOO mean of sigma_s * Delta_s with a seeded random sign per span (2026-09-13,
                     H-E7 prereg): stays inside span(Delta) at unit norm and destroys only the shared
                     direction. At 12B L32 a Gaussian direction scores -359 (job 391152), so erase_rand is
                     uninformative there and erase_flip is the directional null H-E7c reads.

2026-09-13 (H-E7/H-C11): the primary layer is a CLI argument (`--layer`, default 7); the L2-13 band runs only
when every band layer is banked. `--compare-root` points at a SECOND gate root on the same host (our 12B pair,
against the released pair in the main root): its `edit`/`foreign` bytes are written in this job as
`edit_ours`/`foreign_ours`, licensed by gate G1 (identical span positions, h0/h1b agreeing) and gated by G2
(reproduce that root's banked rows to <= 0.05 nats).

Every arm is written through the same PositionReplacer as the banked gate, so `erase_own` must reproduce
`dG_S_swap_L7` to <= 0.05 nats on every item or the run is a harness fault (exit 3). The readout is G_sum,
paired per item, unchanged from the rest of the family.

WHY |Delta_s| AND NOT THE RAW MEAN. At L7 the mean displacement has norm 315 against a mean |Delta_s| of
1300 (cos 0.26 on average): adding the raw mean would move the state ~6 % of its norm and could only fail
for want of dose. Matching each span's OWN displacement magnitude asks the sharper question -- moved as
far as the clean state is, but along the shared direction, how much of swap is recovered?
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
_PROJ = _HERE.parents[1]

from nla_ml_gate import load_cfg, load_host, load_vectors, root, select_pairs  # noqa: E402
from steer import PositionReplacer  # noqa: E402

TAG = "[ERASE]"
PRIMARY_LAYER = 7
ALPHAS = (0.5, 1.0, 2.0)
PRIMARY_ALPHA = 1.0
BAND = list(range(2, 14))
BETA_BAND = 0.35
SETS = {"L7": ([PRIMARY_LAYER], 1.0), "band": (BAND, BETA_BAND)}
REPEAT_ARMS = ("edit", "foreign", "swap", "self")
COMPARE_ARMS = ("edit_ours", "foreign_ours")
G1_COS_MIN = 0.999          # H-E7 prereg G1: per-span cos(h0_main, h0_compare) and likewise h1b
# The three arms that write BANKED BYTES: they must be present AND reproduce, because `all({})` is True and a
# layer whose banked rows lacked these keys would otherwise pass the identity gate vacuously (H-E8 prereg,
# 2026-09-13 -- strengthened before any H-E8 data existed; this can only turn a pass into a failure).
REQUIRED_GATED = ("edit", "foreign", "swap")
N_BOOT = 10_000
# frozen rule constants (prereg)
DIRECTIONAL_NATS = 3.0
SHARE_BAR = 0.55            # banked removal share 0.493 (T_L1_sub / T_L0_all = 20.47 / 41.52) + slack
IDENT_TOL = 0.05
SELF_TOL = 1.0
# Vector-space gate for `erase_own` (added 2026-09-13 after job 392723; see
# log/nla-harness/2026-09-13_erasure-gate-fault.md). `erase_own` is the first arm in this programme built by
# ARITHMETIC rather than by writing banked bytes, and `h1b + |D|*unit(D)` equals `h0` to 3e-05 on a vector of
# norm 5427 without being bit-identical: 111 of 1 205 760 components (0.0092 %) land on a different bf16 value
# after the write's cast, and that moves the summed teacher-forced logp by up to 1.2 nats. So the SCORE-level
# 0.05-nat check cannot express the claim; the claim ("the erasure arithmetic reconstructs h0") is about
# vectors, and is gated here. IDENT_TOL/SELF_TOL are unchanged and still gate the three banked-byte arms,
# where they reproduce at exactly 0.0. Bars set from the measured values with ~30x headroom, not tuned to pass.
VEC_TOL = 1e-3              # measured max abs(erase_own - h0) in fp32: 3.05e-05
BF16_FRAC_TOL = 5e-4        # measured differing-component fraction after the bf16 cast: 9.2e-05


def boot(x, seed: int) -> dict:
    x = np.asarray(x, dtype=float)
    if not len(x):
        return {"mean": float("nan"), "ci95": [float("nan")] * 2, "n": 0}
    r = np.random.default_rng(seed)
    b = np.sort(x[r.integers(0, len(x), size=(N_BOOT, len(x)))].mean(1))
    return {"mean": float(x.mean()), "ci95": [float(b[int(.025 * N_BOOT)]), float(b[int(.975 * N_BOOT)])],
            "n": int(len(x))}


def loo_means(arr: dict, by: dict[str, list[dict]], signs: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """{snippet_id: unit leave-one-item-out mean of (h0 - h1b) over the OTHER items' spans}.

    `signs` (per span, +-1) builds the `erase_flip` control: same spans, same scales, shared direction destroyed.
    """
    d = (arr["h0"] - arr["h1b"]).astype(np.float64)
    if signs is not None:
        d = d * np.asarray(signs, dtype=np.float64)[:, None]
    tot = d.sum(0)
    n = d.shape[0]
    out = {}
    for sid, rs in by.items():
        idx = [int(r["vec"]) for r in rs]
        e = (tot - d[idx].sum(0)) / max(1, n - len(idx))
        out[sid] = (e / (np.linalg.norm(e) + 1e-12)).astype(np.float32)
    return out


def erase_target(h1b: np.ndarray, delta_norm: float, direction: np.ndarray, alpha: float) -> np.ndarray:
    return (h1b + alpha * delta_norm * direction).astype(np.float32)


def loo_norms(arr: dict, by: dict[str, list[dict]]) -> dict[str, float]:
    """{snippet_id: mean ||h0 - h1b|| over the OTHER items' spans}.

    H-E14 (2026-09-13): every other erasure arm steps by `dn = ||h0_s - h1b_s||`, the TRUE per-span displacement
    norm, which is computed from `h0` -- an oracle the trained AV/AR never sees. `erase_loonorm` replaces it with
    this leave-one-item-out estimate, so direction AND magnitude come only from other items and the arm is what a
    deployed method could actually reproduce. Directional verdicts are unaffected (every compared arm shared the
    same `dn`), but the comparison against the trained `edit` is only fair with this arm.
    """
    dn = np.linalg.norm((arr["h0"] - arr["h1b"]).astype(np.float64), axis=1)
    out: dict[str, float] = {}
    for sid, rs in by.items():
        own = {int(r["vec"]) for r in rs}
        others = [k for k in range(len(dn)) if k not in own]
        out[sid] = float(dn[others].mean()) if others else float(dn.mean())
    return out


def flip_signs(n_spans: int, seed: int, K: int) -> np.ndarray:
    """Seeded +-1 per span for `erase_flip` (H-E7 prereg): seed + crc32('L<K>#flip')."""
    rng = np.random.default_rng(seed + zlib.crc32(f"L{K}#flip".encode()))
    return rng.choice(np.array([-1.0, 1.0]), size=n_spans)


def anchoring_identity(main: tuple, other: tuple, allow_subset: bool = False) -> dict:
    """GATE G1 (H-E7 prereg): two gate roots on the same host resolved the SAME spans at the SAME positions,
    and captured the same h0/h1b there. Only then may the other root's edit bytes be written in this job.

    `allow_subset` is for the in-job smoke only (3 items against a 60-item compare root): the main root's
    items must then be a subset, with every one of ITS spans paired. The full run requires equality."""
    by_m, arr_m = main; by_o, arr_o = other
    same = set(by_m) == set(by_o) or (allow_subset and set(by_m) <= set(by_o))
    rep_: dict = {"same_items": bool(same), "allow_subset": bool(allow_subset),
                  "n_spans_main": int(sum(len(v) for v in by_m.values())),
                  "n_spans_other": int(sum(len(v) for v in by_o.values())),
                  "n_rows_main": int(len(arr_m["h0"])), "n_rows_other": int(len(arr_o["h0"])),
                  "position_mismatches": 0, "pairs": []}
    pairs: list[tuple[int, int]] = []
    for sid in sorted(set(by_m) & set(by_o)):
        m = {r["span_i"]: r for r in by_m[sid]}; o = {r["span_i"]: r for r in by_o[sid]}
        if set(m) != set(o):
            rep_["position_mismatches"] += len(set(m) ^ set(o)); continue
        for si in m:
            if list(m[si]["positions"]) != list(o[si]["positions"]):
                rep_["position_mismatches"] += 1
            pairs.append((int(m[si]["vec"]), int(o[si]["vec"])))
    rep_["n_paired_spans"] = len(pairs)
    if pairs:
        im = np.array([a for a, _ in pairs]); io = np.array([b for _, b in pairs])
        def cos_rows(A, B):
            A = A.astype(np.float64); B = B.astype(np.float64)
            return (A * B).sum(1) / (np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1) + 1e-12)
        for k in ("h0", "h1b"):
            c = cos_rows(arr_m[k][im], arr_o[k][io])
            rep_[f"{k}_cos_min"] = float(c.min()); rep_[f"{k}_max_abs"] = float(np.abs(arr_m[k][im] - arr_o[k][io]).max())
    rep_["passes"] = bool(rep_["same_items"] and rep_["position_mismatches"] == 0
                          and rep_["n_paired_spans"] == rep_["n_spans_main"]
                          and (allow_subset or rep_["n_spans_main"] == rep_["n_spans_other"])
                          and rep_.get("h0_cos_min", 0.0) >= G1_COS_MIN and rep_.get("h1b_cos_min", 0.0) >= G1_COS_MIN)
    rep_["cos_min_required"] = G1_COS_MIN
    rep_.pop("pairs")
    return rep_


def _edit_verdict(d: dict, suffix: str = "") -> str:
    lo, hi = d["ci95"]
    d["ci_width"] = float(hi - lo)
    return ("ERASURE-BEATS-EDIT" if lo > 0 else "EDIT-CARRIES-CONTENT" if hi < 0 else "ERASURE-MATCHES-EDIT") + suffix


def rules(rows: list[dict], seed: int, S: str = "L7") -> dict:
    """The frozen H-E1/H-E2/H-E3 table on the primary-set beta=1 cells, paired per item.

    2026-09-13 additions (H-E7 prereg), computed whenever their arms are present: H-E7c (`erase_flip` as the
    directional null -- the gate at 12B where random is uninformative), H-E7b (`edit_ours`), H-C11 (released
    `edit - foreign`). At 4B the E3 gate (vs random) stays the one that voids E1/E2, as frozen on 2026-09-12;
    at any other layer the prereg names E7c as the voiding gate.
    """
    k = lambda a: f"dG_{a}_{S}"  # noqa: E731
    have = [r for r in rows if all(k(a) in r for a in ("erase_1.0", "erase_rand", "edit", "swap"))]
    e1 = np.array([r[k("erase_1.0")] for r in have]); rnd = np.array([r[k("erase_rand")] for r in have])
    ed = np.array([r[k("edit")] for r in have]); sw = np.array([r[k("swap")] for r in have])
    out = {"n": len(have), "set": S,
           "H_E3": boot(e1 - rnd, seed + 31), "H_E1": boot(e1 - ed, seed + 32),
           "H_E2": boot(e1 - SHARE_BAR * sw, seed + 33),
           "share": float(e1.mean() / sw.mean()) if len(have) and sw.mean() else float("nan")}
    d3 = out["H_E3"]
    out["verdict_E3"] = ("ERASURE-DIRECTIONAL" if d3["n"] and d3["mean"] >= DIRECTIONAL_NATS and d3["ci95"][0] > 0
                         else "ERASURE-NOISE")
    out["verdict_E1"] = _edit_verdict(out["H_E1"])
    lo2, hi2 = out["H_E2"]["ci95"]
    out["verdict_E2"] = ("ERASURE-WITHIN-SHARE" if hi2 < 0 else "ERASURE-EXCEEDS-SHARE" if lo2 > 0
                         else "SHARE-UNRESOLVED")
    gate_verdict = out["verdict_E3"]
    fl = [r for r in have if k("erase_flip") in r]
    if fl:
        d7 = boot([r[k("erase_1.0")] - r[k("erase_flip")] for r in fl], seed + 34)
        out["H_E7c"] = d7
        out["verdict_E7c"] = ("ERASURE-DIRECTIONAL" if d7["n"] and d7["mean"] >= DIRECTIONAL_NATS and d7["ci95"][0] > 0
                              else "ERASURE-NOISE")
        if S != "L7":
            gate_verdict = out["verdict_E7c"]
    ln = [r for r in have if k("erase_loonorm") in r]
    if ln:
        # H-E14: what the oracle magnitude was worth, and whether the DEPLOYABLE arm is still directional.
        out["H_E14"] = boot([r[k("erase_1.0")] - r[k("erase_loonorm")] for r in ln], seed + 41)
        lo14, hi14 = out["H_E14"]["ci95"]
        out["verdict_E14"] = ("ORACLE-SCALED" if out["H_E14"]["mean"] >= DIRECTIONAL_NATS and lo14 > 0
                              else "ORACLE-FREE")
        fl14 = [r for r in ln if k("erase_flip") in r]
        if fl14:
            d14 = boot([r[k("erase_loonorm")] - r[k("erase_flip")] for r in fl14], seed + 42)
            out["H_E14b"] = d14
            out["verdict_E14b"] = ("DEPLOYABLE-DIRECTIONAL" if d14["mean"] >= DIRECTIONAL_NATS and d14["ci95"][0] > 0
                                   else "DEPLOYABLE-NOISE")
    ou = [r for r in have if k("edit_ours") in r]
    if ou:
        out["H_E7b"] = boot([r[k("erase_1.0")] - r[k("edit_ours")] for r in ou], seed + 35)
        out["verdict_E7b"] = _edit_verdict(out["H_E7b"], "-OURS")
        fo = [r for r in ou if k("foreign_ours") in r and k("foreign") in r]
        out["edit_minus_edit_ours"] = boot([r[k("edit")] - r[k("edit_ours")] for r in ou], seed + 36)
        out["spec_ours"] = boot([r[k("edit_ours")] - r[k("foreign_ours")] for r in fo], seed + 37)
    sp = [r for r in have if k("foreign") in r]
    if sp:
        c11 = boot([r[k("edit")] - r[k("foreign")] for r in sp], seed + 38)
        out["H_C11"] = c11
        out["verdict_C11"] = ("SPEC-LIVE" if c11["n"] and c11["mean"] >= DIRECTIONAL_NATS and c11["ci95"][0] > 0
                              else "SPEC-FLAT")
    if gate_verdict == "ERASURE-NOISE":
        for key in ("verdict_E1", "verdict_E2", "verdict_E7b"):
            if key in out:
                out[key] += " (VOID: directional gate noise)"
    out["voiding_gate"] = "E3 (random)" if S == "L7" else "E7c (flip)"
    return out


def vector_identity(arr: dict) -> dict:
    """GATE: does the erasure arithmetic reconstruct h0? A statement about vectors, not scores.

    `erase_own` is `erase_target(h1b, |D|, D/|D|, 1.0)`, which is `h0` up to fp32 rounding. This checks that
    directly, plus how much of the difference survives the bf16 cast the write performs -- the quantity that
    actually reaches the model. Replaces the score-level erase_own check, which cannot hold in bf16.

    SCOPE, stated because it is narrower than what it replaces: this recomputes D from h0/h1b, so it is
    self-consistent by construction. It catches a bug in `erase_target` (its job) but NOT a mismatch between the
    banked vectors and what the hook actually writes -- that remains covered by the score-level gate on the
    three banked-byte arms (edit/foreign/swap), which reproduce at exactly 0.0.
    """
    h0 = arr["h0"].astype(np.float32)
    h1b = arr["h1b"].astype(np.float32)
    d = h0 - h1b
    dn = np.linalg.norm(d, axis=1)
    v = np.stack([erase_target(h1b[i], float(dn[i]), d[i] / (float(dn[i]) + 1e-12), 1.0)
                  for i in range(len(h0))])
    max_abs = float(np.abs(v - h0).max())
    bf = torch.from_numpy(v).to(torch.bfloat16) != torch.from_numpy(np.ascontiguousarray(h0)).to(torch.bfloat16)
    frac = float(bf.float().mean())
    return {"max_abs_fp32": max_abs, "vec_tol": VEC_TOL,
            "bf16_differing_frac": frac, "bf16_frac_tol": BF16_FRAC_TOL,
            "n_spans": int(len(h0)), "bf16_spans_affected": int(bf.any(1).sum()),
            "passes": bool(max_abs <= VEC_TOL and frac <= BF16_FRAC_TOL)}


def identity(rows: list[dict], banked: dict[str, dict], S: str = "L7",
             banked_ours: dict[str, dict] | None = None) -> dict:
    """GATE: edit/foreign/swap reproduce the banked rows (bit-identical writes); self == 0.

    `erase_own` is measured here too but is NOT a pass/fail: it is the bf16 write-chaos diagnostic, reported as
    `erase_own_chaos_nats` so the floor is visible next to the effects it does and does not threaten. See
    log/nla-harness/2026-09-13_erasure-gate-fault.md.

    `banked_ours` (H-E7 gate G2): the compare root's gate rows; `edit_ours`/`foreign_ours` must reproduce them.
    """
    worst: dict[str, float] = {}
    checks = [("erase_own", f"dG_S_swap_{S}"), ("edit", f"dG_S_edit_{S}"),
              ("foreign", f"dG_S_foreign_{S}"), ("swap", f"dG_S_swap_{S}")]
    for r in rows:
        b = banked.get(r["snippet_id"])
        if not b:
            continue
        for arm, bk in checks:
            if f"dG_{arm}_{S}" in r and bk in b:
                worst[arm] = max(worst.get(arm, 0.0), abs(r[f"dG_{arm}_{S}"] - b[bk]))
        bo = (banked_ours or {}).get(r["snippet_id"])
        if bo:
            for arm, bk in (("edit_ours", f"dG_S_edit_{S}"), ("foreign_ours", f"dG_S_foreign_{S}")):
                if f"dG_{arm}_{S}" in r and bk in bo:
                    worst[arm] = max(worst.get(arm, 0.0), abs(r[f"dG_{arm}_{S}"] - bo[bk]))
        if f"dG_self_{S}" in r:
            worst["self"] = max(worst.get("self", 0.0), abs(r[f"dG_self_{S}"]))
    gated = {a: v for a, v in worst.items() if a not in ("self", "erase_own")}
    missing = [a for a in REQUIRED_GATED if a not in gated]
    passes = (not missing and all(v <= IDENT_TOL for v in gated.values())
              and worst.get("self", 0.0) <= SELF_TOL)
    return {"worst_abs_diff": worst, "gated_arms": sorted(gated), "missing_gated": missing,
            "tol": IDENT_TOL, "self_tol": SELF_TOL,
            "erase_own_chaos_nats": worst.get("erase_own"), "passes": bool(passes)}


def score(rows: list[dict], banked: dict[str, dict], seed: int, vec: dict | None = None,
          sets: dict | None = None, S: str = "L7", banked_ours: dict[str, dict] | None = None,
          anchoring: dict | None = None) -> dict:
    sets = sets or SETS
    st: dict = {"experiment": "E_erasure_vector", "n_items": len(rows), "seed": seed, "alphas": list(ALPHAS),
                "primary_set": S, "sets": {k: {"layers": v[0], "beta": v[1]} for k, v in sets.items()},
                "identity": identity(rows, banked, S, banked_ours), "arms": {}, "summary": []}
    if anchoring is not None:
        st["anchoring_identity"] = anchoring
        if not anchoring["passes"]:
            st["verdict"] = "E-HARNESS-FAULT"; st["reportable"] = False
            st["summary"].append(f"ANCHORING IDENTITY (G1) FAILED: {anchoring}")
            return st
    if vec is not None:
        st["vector_identity"] = vector_identity(vec)
        if not st["vector_identity"]["passes"]:
            st["verdict"] = "E-HARNESS-FAULT"
            st["reportable"] = False
            st["summary"].append(f"VECTOR IDENTITY FAILED: {st['vector_identity']}")
            return st
    if not st["identity"]["passes"]:
        st["verdict"] = "E-HARNESS-FAULT"; st["reportable"] = False
        st["summary"].append(f"IDENTITY FAILED (gated arms {st['identity']['gated_arms']}): "
                             f"{st['identity']['worst_abs_diff']}")
        return st
    for key in sorted({k for r in rows for k in r if k.startswith("dG_")}):
        st["arms"][key] = boot([r[key] for r in rows if key in r], seed + zlib.crc32(key.encode()) % 1000)
    st["rules"] = rules(rows, seed, S)
    # descriptive: erasure vs a single other item's true term, per set
    for s in sets:
        a, f = f"dG_erase_1.0_{s}", f"dG_foreign_{s}"
        d = [r[a] - r[f] for r in rows if a in r and f in r]
        st["rules"][f"erase_minus_foreign_{s}"] = boot(d, seed + 40)
    # descriptive: released c3 transport at this layer, if the banked rows carry it
    c3 = [(banked[r["snippet_id"]][f"dG_S_c3_{S}"], banked[r["snippet_id"]][f"dG_S_swap_{S}"]) for r in rows
          if r["snippet_id"] in banked and f"dG_S_c3_{S}" in banked[r["snippet_id"]]]
    if c3:
        st["rules"]["banked_c3_over_swap"] = float(np.mean([a for a, _ in c3]) / (np.mean([b for _, b in c3]) or np.nan))
    # descriptive: does an item gain more when its own displacement aligns with the shared direction?
    g = [(r[f"dG_erase_1.0_{S}"], r[f"cos_delta_E_{S}"]) for r in rows if f"cos_delta_E_{S}" in r and f"dG_erase_1.0_{S}" in r]
    if len(g) > 3:
        from scipy.stats import spearmanr
        rho, p = spearmanr([x for x, _ in g], [y for _, y in g])
        st["rules"]["spearman_gain_vs_cos"] = {"rho": float(rho), "p": float(p), "n": len(g)}
    st["reportable"] = True
    R = st["rules"]; A = st["arms"]
    order = ("swap", "erase_own", "erase_0.5", "erase_1.0", "erase_2.0", "erase_loonorm", "erase_flip",
             "erase_rand", "edit", "foreign", "edit_ours", "foreign_ours")
    for s in sets:
        st["summary"].append(f"{s}: " + " · ".join(f"{a} {A[f'dG_{a}_{s}']['mean']:+.2f}" for a in order
                                                    if f"dG_{a}_{s}" in A))
    st["summary"] += [
        f"H-E3 {R['verdict_E3']}: erase_1.0 - erase_rand = {R['H_E3']['mean']:+.2f} {R['H_E3']['ci95']} (bar +{DIRECTIONAL_NATS})",
        f"H-E1 {R['verdict_E1']}: erase_1.0 - edit = {R['H_E1']['mean']:+.2f} {R['H_E1']['ci95']} width {R['H_E1']['ci_width']:.2f}",
        f"H-E2 {R['verdict_E2']}: erase_1.0 - {SHARE_BAR}*swap = {R['H_E2']['mean']:+.2f} {R['H_E2']['ci95']} · share {R['share']:.3f}",
    ]
    if "H_E7c" in R:
        st["summary"].append(f"H-E7c {R['verdict_E7c']}: erase_1.0 - erase_flip = {R['H_E7c']['mean']:+.2f} "
                             f"{R['H_E7c']['ci95']} (bar +{DIRECTIONAL_NATS}) · voiding gate: {R['voiding_gate']}")
    if "H_E14" in R:
        st["summary"].append(f"H-E14 {R['verdict_E14']}: erase_1.0 - erase_loonorm = {R['H_E14']['mean']:+.2f} "
                             f"{R['H_E14']['ci95']} (bar +{DIRECTIONAL_NATS} = the oracle magnitude's worth)"
                             + (f" · H-E14b {R['verdict_E14b']}: loonorm - flip = {R['H_E14b']['mean']:+.2f} "
                                f"{R['H_E14b']['ci95']}" if "H_E14b" in R else ""))
    if "H_E7b" in R:
        st["summary"].append(f"H-E7b {R['verdict_E7b']}: erase_1.0 - edit_ours = {R['H_E7b']['mean']:+.2f} "
                             f"{R['H_E7b']['ci95']} width {R['H_E7b']['ci_width']:.2f} · edit - edit_ours "
                             f"{R['edit_minus_edit_ours']['mean']:+.2f} {R['edit_minus_edit_ours']['ci95']} · "
                             f"SPEC ours {R['spec_ours']['mean']:+.2f} {R['spec_ours']['ci95']}")
    if "H_C11" in R:
        st["summary"].append(f"H-C11 {R['verdict_C11']}: edit - foreign = {R['H_C11']['mean']:+.2f} {R['H_C11']['ci95']}"
                             + (f" · banked c3/swap {R['banked_c3_over_swap']:.3f}" if "banked_c3_over_swap" in R else ""))
    return st


def main() -> int:  # noqa: C901
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(_PROJ / "nla/configs/nla_ml_gate.yaml"))
    ap.add_argument("--root", default=None)
    ap.add_argument("--traces", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--max-hours", type=float, default=1.5)
    ap.add_argument("--layer", type=int, default=PRIMARY_LAYER, help="primary layer (set name L<K>)")
    ap.add_argument("--no-band", action="store_true",
                    help="primary set only: at 4B every band layer is banked, so a depth sweep would re-run the "
                         "already-published L2-13 beta=0.35 set on every invocation (H-E8 prereg)")
    ap.add_argument("--compare-root", default=None,
                    help="H-E7: a second gate root on the same host whose edit/foreign bytes are written as "
                         "edit_ours/foreign_ours (gates G1/G2)")
    args = ap.parse_args()

    cfg = load_cfg(Path(args.config))
    if "qwen" in cfg["train"]["host"]["model_id"].lower():
        print(f"{TAG} REFUSED: Chinese models are not run in this project."); return 2
    args._cfg = cfg
    args.repair = bool(cfg["repair"]); args.max_spans_per_item = int(cfg["max_spans_per_item"])
    rt = root(cfg, args.root)
    args.traces = args.traces or str(_PROJ / "data/nla/p0/trace_llr/gemma4b/traces.jsonl")
    gdir = rt / cfg["gate_subdir"]; vdir = gdir / "vectors"
    out = Path(args.out or (gdir / "erasure")); out.mkdir(parents=True, exist_ok=True)
    seed = int(cfg["seed"]); torch.manual_seed(seed)
    d_model = int(cfg["train"]["host"]["d_model"])

    traces = {t["snippet_id"]: t for t in map(json.loads, open(args.traces))}
    banked = {r["snippet_id"]: r for r in map(json.loads, open(gdir / "gate_rows.jsonl"))}
    P = int(args.layer); S = f"L{P}"
    band_ok = (not args.no_band) and all((vdir / f"L{K}.npz").exists() for K in BAND)
    sets = {S: ([P], 1.0)}
    if band_ok:
        sets["band"] = (BAND, BETA_BAND)
    else:
        why = "suppressed by --no-band" if args.no_band else f"not banked in {vdir}"
        print(f"{TAG} band L{BAND[0]}-{BAND[-1]} {why}: primary set only", flush=True)
    layers = sorted({K for Ks, _ in sets.values() for K in Ks})
    vec = {K: load_vectors(vdir, K) for K in layers}
    ref = vec[layers[0]][0]
    for K in layers[1:]:
        assert set(vec[K][0]) == set(ref), f"L{K}: item set differs"
    E = {K: loo_means(vec[K][1], vec[K][0]) for K in layers}
    Eflip = {K: loo_means(vec[K][1], vec[K][0], flip_signs(len(vec[K][1]["h0"]), seed, K)) for K in layers}
    Dloo = {K: loo_norms(vec[K][1], vec[K][0]) for K in layers}
    ours = None; banked_ours = None; anchoring = None
    if args.compare_root:
        cg = Path(args.compare_root) / cfg["gate_subdir"]
        ours = load_vectors(cg / "vectors", P)
        banked_ours = {r["snippet_id"]: r for r in map(json.loads, open(cg / "gate_rows.jsonl"))}
        anchoring = anchoring_identity(vec[P], ours, allow_subset=bool(args.smoke or args.limit))
        print(f"{TAG} G1 anchoring identity vs {cg}: {anchoring}", flush=True)
        if not anchoring["passes"]:
            out.mkdir(parents=True, exist_ok=True)
            st = {"experiment": "E_erasure_vector", "verdict": "E-HARNESS-FAULT", "reportable": False,
                  "anchoring_identity": anchoring, "summary": [f"ANCHORING IDENTITY (G1) FAILED: {anchoring}"]}
            (out / "erasure_stats.json").write_text(json.dumps(st, indent=1))
            print(f"{TAG} {st['summary'][0]}", flush=True); return 3

    model, _tok = load_host(cfg, args.device)
    reps = {K: PositionReplacer(model, K) for K in layers}

    @torch.no_grad()
    def logp(pids, rids, expect: int) -> float:
        seq = torch.tensor([list(pids) + list(rids)], device=model.device)
        for r in reps.values():
            r.reset()
        lg = model(seq, use_cache=False, logits_to_keep=len(rids) + 1).logits[0, :-1]
        tgt = seq[0, len(pids):]
        tot = 0.0
        for s in range(0, lg.shape[0], 256):
            tot += float(torch.log_softmax(lg[s:s + 256].float(), -1).gather(1, tgt[s:s + 256, None])[:, 0].sum())
        n_w = sum(r.n_positions_written for r in reps.values())
        if n_w != expect:
            raise RuntimeError(f"replacers wrote {n_w} positions, expected {expect}")
        return tot

    def targets(K: int, sid: str, arm: str) -> dict:
        by, arr = vec[K]
        tg: dict[int, torch.Tensor] = {}
        for r in by[sid]:
            i = int(r["vec"])
            if arm == "self":
                a, b = r["self_range"]
                for q, j in zip(r["positions"], range(a, b)):
                    tg[q] = torch.from_numpy(arr["self_all"][j])
                continue
            if arm in ("edit", "foreign"):
                v = arr[arm][i]
            elif arm in COMPARE_ARMS:
                # the compare root's bytes at ITS index for this span (G1 guarantees the span sets coincide)
                by_o, arr_o = ours
                j = next(int(q["vec"]) for q in by_o[sid] if q["span_i"] == r["span_i"])
                v = arr_o[arm.split("_")[0]][j]
            elif arm == "swap":
                v = arr["h0"][i]
            else:
                h1b, h0 = arr["h1b"][i], arr["h0"][i]
                delta = h0 - h1b; dn = float(np.linalg.norm(delta))
                if arm == "erase_own":
                    v = erase_target(h1b, dn, delta / (dn + 1e-12), 1.0)
                elif arm == "erase_rand":
                    rng = np.random.default_rng(seed + zlib.crc32(f"{sid}#{r['span_i']}#L{K}#erand".encode()))
                    u = rng.standard_normal(d_model).astype(np.float32); u /= np.linalg.norm(u) + 1e-12
                    v = erase_target(h1b, dn, u, PRIMARY_ALPHA)
                elif arm == "erase_flip":
                    v = erase_target(h1b, dn, Eflip[K][sid], PRIMARY_ALPHA)
                elif arm == "erase_loonorm":
                    # the ONLY arm that touches no h0 belonging to this item: LOO direction AND LOO magnitude
                    v = erase_target(h1b, Dloo[K][sid], E[K][sid], PRIMARY_ALPHA)
                else:                                   # erase_<alpha>
                    v = erase_target(h1b, dn, E[K][sid], float(arm.split("_")[1]))
            vt = torch.from_numpy(np.ascontiguousarray(v))
            for q in r["positions"]:
                tg[q] = vt
        return tg

    def run(Ks, sid, arm, beta, pids, rids) -> float:
        n = 0
        for K, r in reps.items():
            t = targets(K, sid, arm) if K in Ks else None
            r.set_beta(beta); r.set_targets(t); n += len(t or {})
        return logp(pids, rids, n)

    arms = ([f"erase_{a}" for a in ALPHAS] + ["erase_flip", "erase_loonorm", "erase_rand", "erase_own"]
            + list(REPEAT_ARMS))
    if ours is not None:
        arms += list(COMPARE_ARMS)
    sids = [p["snippet_id"] for p in select_pairs(args, traces) if p["snippet_id"] in ref]
    if args.limit:
        sids = sids[:args.limit]
    print(f"{TAG} {len(sids)} items · arms {arms} · sets {sets}", flush=True)

    t0 = time.time(); rows: list[dict] = []; n_fwd = 0
    for sid in sids:
        tr = traces[sid]; pids, rids = tr["l1b_prompt_ids"], tr["l0_reply_ids"]
        row: dict = {"snippet_id": sid, "n_tok": len(pids) + len(rids)}
        for r in reps.values():
            r.set_targets(None)
        base = logp(pids, rids, 0); n_fwd += 1
        row["logp_U"] = base
        for sname, (Ks, beta) in sets.items():
            for arm in arms:
                if arm in ("erase_own", "self") + COMPARE_ARMS and sname != S:
                    continue                          # identity arms only where the gate is defined
                row[f"dG_{arm}_{sname}"] = run(set(Ks), sid, arm, beta, pids, rids) - base; n_fwd += 1
        # how aligned is this item's own displacement with the shared direction it was written along?
        by, arr = vec[P]
        idx = [int(r["vec"]) for r in by[sid]]
        dl = (arr["h0"][idx] - arr["h1b"][idx]).mean(0)
        row[f"cos_delta_E_{S}"] = float(dl @ E[P][sid] / (np.linalg.norm(dl) + 1e-12))
        rows.append(row)
        print(f"{TAG} {len(rows)}/{len(sids)} {sid} · {n_fwd} fwd · {(time.time()-t0)/60:.1f} min · "
              f"{S} swap {row[f'dG_swap_{S}']:+.1f} own {row[f'dG_erase_own_{S}']:+.1f} e1 {row[f'dG_erase_1.0_{S}']:+.1f} "
              f"flip {row[f'dG_erase_flip_{S}']:+.1f} edit {row[f'dG_edit_{S}']:+.1f}"
              + (f" ours {row[f'dG_edit_ours_{S}']:+.1f}" if ours is not None else ""), flush=True)
        if time.time() - t0 > args.max_hours * 3600:
            print(f"{TAG} wall-clock stop after {len(rows)} items", flush=True); break
    for r in reps.values():
        r.close()
    (out / "erasure_rows.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    # vec[P] = (by_sid, arrays); the vector gate needs the primary-layer arrays only
    st = score(rows, banked, seed, vec=vec[P][1], sets=sets, S=S, banked_ours=banked_ours, anchoring=anchoring)
    (out / "erasure_stats.json").write_text(json.dumps(st, indent=1))
    for line in st["summary"]:
        print(f"{TAG} {line}", flush=True)
    print(f"{TAG} wrote {out}", flush=True)
    return 0 if st.get("reportable") else 3


if __name__ == "__main__":
    raise SystemExit(main())
