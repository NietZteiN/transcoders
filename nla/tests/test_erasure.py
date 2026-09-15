"""CPU tests for nla_erasure: the LOO mean, the identity of erase_own with h0, and the frozen rule table."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nla_erasure import (BF16_FRAC_TOL, SHARE_BAR, VEC_TOL, anchoring_identity,  # noqa: E402
                         erase_target, flip_signs, identity, loo_means, rules, score, vector_identity)

SEED = 20260724


def _vectors(n_items=4, spans=3, d=16, seed=0):
    rng = np.random.default_rng(seed)
    by, h0, h1b = {}, [], []
    for i in range(n_items):
        by[f"s{i}"] = []
        for j in range(spans):
            by[f"s{i}"].append({"vec": len(h0), "span_i": j, "positions": [j]})
            h1b.append(rng.normal(size=d)); h0.append(h1b[-1] + rng.normal(size=d) + np.arange(d) * 0.1)
    return by, {"h0": np.array(h0, dtype=np.float32), "h1b": np.array(h1b, dtype=np.float32)}


def test_loo_mean_excludes_the_items_own_spans_and_is_unit():
    by, arr = _vectors()
    E = loo_means(arr, by)
    d = arr["h0"].astype(float) - arr["h1b"].astype(float)
    for sid, rs in by.items():
        own = {int(r["vec"]) for r in rs}
        others = [k for k in range(d.shape[0]) if k not in own]
        want = d[others].mean(0); want /= np.linalg.norm(want)
        assert np.allclose(E[sid], want, atol=1e-5)
        assert abs(np.linalg.norm(E[sid]) - 1) < 1e-5


def test_erase_own_reproduces_h0_exactly_in_float32():
    by, arr = _vectors()
    for i in range(arr["h0"].shape[0]):
        h1b, h0 = arr["h1b"][i], arr["h0"][i]
        delta = h0 - h1b; dn = float(np.linalg.norm(delta))
        v = erase_target(h1b, dn, delta / dn, 1.0)
        assert np.allclose(v, h0, atol=1e-4)


def test_erase_alpha_moves_exactly_alpha_times_the_displacement_norm():
    by, arr = _vectors()
    E = loo_means(arr, by)
    h1b, h0 = arr["h1b"][0], arr["h0"][0]
    dn = float(np.linalg.norm(h0 - h1b))
    for a in (0.5, 1.0, 2.0):
        v = erase_target(h1b, dn, E["s0"], a)
        assert abs(np.linalg.norm(v - h1b) - a * dn) < 1e-3 * dn


def _rows(n, e1, rnd, ed, sw, noise=1.0, seed=0):
    rng = np.random.default_rng(seed)
    return [{"snippet_id": f"s{i}", "dG_erase_1.0_L7": e1 + rng.normal(0, noise),
             "dG_erase_rand_L7": rnd + rng.normal(0, noise), "dG_edit_L7": ed + rng.normal(0, noise),
             "dG_swap_L7": sw + rng.normal(0, noise)} for i in range(n)]


def test_rules_predicted_pattern_directional_matches_within_share():
    r = rules(_rows(60, e1=32.0, rnd=-15.0, ed=32.0, sw=73.0), SEED)
    assert r["verdict_E3"] == "ERASURE-DIRECTIONAL"
    assert r["verdict_E1"] == "ERASURE-MATCHES-EDIT"
    assert r["verdict_E2"] == "ERASURE-WITHIN-SHARE"          # 32 < 0.55*73 = 40.15


def test_rules_edit_carries_content_and_exceeds_share():
    r = rules(_rows(60, e1=20.0, rnd=-15.0, ed=32.0, sw=73.0), SEED)
    assert r["verdict_E1"] == "EDIT-CARRIES-CONTENT"
    r2 = rules(_rows(60, e1=60.0, rnd=-15.0, ed=32.0, sw=73.0), SEED)
    assert r2["verdict_E1"] == "ERASURE-BEATS-EDIT" and r2["verdict_E2"] == "ERASURE-EXCEEDS-SHARE"


def test_noise_voids_e1_and_e2():
    r = rules(_rows(60, e1=-14.0, rnd=-15.0, ed=32.0, sw=73.0), SEED)
    assert r["verdict_E3"] == "ERASURE-NOISE"
    assert "VOID" in r["verdict_E1"] and "VOID" in r["verdict_E2"]


def test_identity_gate_catches_a_drifted_swap():
    rows = _rows(5, e1=32, rnd=-15, ed=32, sw=73, noise=0.0)
    for r in rows:
        r["dG_erase_own_L7"] = r["dG_swap_L7"]; r["dG_foreign_L7"] = 23.0; r["dG_self_L7"] = 0.0
    banked = {r["snippet_id"]: {"dG_S_swap_L7": 73.0, "dG_S_edit_L7": 32.0, "dG_S_foreign_L7": 23.0} for r in rows}
    assert identity(rows, banked)["passes"]
    # 2026-09-13: erase_own is a reported diagnostic, not a gated arm, so drifting IT must not fail the gate...
    rows[2]["dG_erase_own_L7"] += 0.2
    assert identity(rows, banked)["passes"]
    # ...while drifting a banked-byte arm still must.
    rows[2]["dG_swap_L7"] += 0.2
    assert not identity(rows, banked)["passes"]
    st = score(rows, banked, SEED)
    assert st["verdict"] == "E-HARNESS-FAULT" and st["reportable"] is False


# --- added 2026-09-13 after job 392723 (log/nla-harness/2026-09-13_erasure-gate-fault.md) --------------------
# The score-level erase_own check could not express its own claim: `h1b + |D|*unit(D)` equals `h0` to 3e-05 but
# is not bit-identical, and in bf16 that moves the summed logp by ~1 nat. The claim is about vectors, so it is
# gated in vector space now. These tests pin both halves of that fix.

def test_vector_gate_passes_when_arithmetic_is_right_but_bytes_differ():
    rng = np.random.default_rng(7)
    h1b = (rng.standard_normal((32, 64)).astype(np.float32) * 100.0)
    h0 = h1b + rng.standard_normal((32, 64)).astype(np.float32) * 25.0
    g = vector_identity({"h0": h0, "h1b": h1b})
    # not bit-identical for most spans, yet the arithmetic is correct -> the gate passes
    assert g["passes"], g
    assert g["max_abs_fp32"] <= VEC_TOL
    assert g["bf16_differing_frac"] <= BF16_FRAC_TOL


def test_vector_gate_fails_when_the_erasure_ARITHMETIC_is_wrong():
    """The gate recomputes D from h0/h1b, so it is self-consistent by construction: it catches a bug in
    `erase_target` (its actual job) but NOT a mismatch between the banked vectors and what the hook writes.
    Scope recorded deliberately -- the score-level gate on the banked-byte arms still covers the wiring."""
    import nla_erasure
    rng = np.random.default_rng(8)
    h1b = rng.standard_normal((16, 64)).astype(np.float32) * 100.0
    h0 = h1b + rng.standard_normal((16, 64)).astype(np.float32) * 25.0
    good = nla_erasure.erase_target
    try:  # a real bug class: the displacement norm dropped from the step
        nla_erasure.erase_target = lambda h, dn, dirn, a: (h + a * dirn).astype(np.float32)
        assert not vector_identity({"h0": h0, "h1b": h1b})["passes"]
    finally:
        nla_erasure.erase_target = good
    assert vector_identity({"h0": h0, "h1b": h1b})["passes"]


def test_erase_own_is_a_reported_diagnostic_not_a_pass_fail():
    """A 1-nat erase_own gap must NOT fail the gate; a 1-nat swap gap still must."""
    banked = {"s1": {"dG_S_swap_L7": 70.0, "dG_S_edit_L7": 30.0, "dG_S_foreign_L7": 25.0}}
    ok = [{"snippet_id": "s1", "dG_erase_own_L7": 71.2, "dG_swap_L7": 70.0,
           "dG_edit_L7": 30.0, "dG_foreign_L7": 25.0, "dG_self_L7": 0.0}]
    g = identity(ok, banked)
    assert g["passes"], g
    assert abs(g["erase_own_chaos_nats"] - 1.2) < 1e-6
    assert "erase_own" not in g["gated_arms"]
    drifted = [dict(ok[0], dG_swap_L7=71.2)]
    assert not identity(drifted, banked)["passes"]


# --- added 2026-09-13 for H-E7/H-C11 (log/nla-harness/2026-09-13_e7-released-erasure-prereg.md) ----------------
# The runner is now layer-parametrised (`--layer 32` on the released 12B pair), has a sign-flip directional null
# (`erase_flip`), and can write a SECOND root's edit/foreign bytes under gates G1 (anchoring) and G2 (rows).

def test_flip_signs_are_seeded_pm1_and_flip_mean_is_unit_inside_span_delta():
    by, arr = _vectors(n_items=6, spans=4)
    s1 = flip_signs(24, SEED, 32); s2 = flip_signs(24, SEED, 32); s3 = flip_signs(24, SEED, 7)
    assert set(np.unique(s1)) <= {-1.0, 1.0} and np.array_equal(s1, s2) and not np.array_equal(s1, s3)
    Ef = loo_means(arr, by, s1); E = loo_means(arr, by)
    d = arr["h0"].astype(float) - arr["h1b"].astype(float)
    for sid, rs in by.items():
        assert abs(np.linalg.norm(Ef[sid]) - 1) < 1e-5
        own = {int(r["vec"]) for r in rs}
        others = [k for k in range(d.shape[0]) if k not in own]
        # lies in span(Delta_others): the residual after projecting onto that span is ~0
        Q, _ = np.linalg.qr(d[others].T)
        assert np.linalg.norm(Ef[sid] - Q @ (Q.T @ Ef[sid])) < 1e-6
        # and it is NOT the shared direction (its cosine to E is far below 1)
        assert float(Ef[sid] @ E[sid]) < 0.9


def _rows32(n, e1, flip, ed, fo, ed_ours, fo_ours, sw=41.0, rnd=-359.0, noise=1.0, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        r = {"snippet_id": f"s{i}"}
        for a, m in (("erase_1.0", e1), ("erase_flip", flip), ("edit", ed), ("foreign", fo),
                     ("edit_ours", ed_ours), ("foreign_ours", fo_ours), ("swap", sw), ("erase_rand", rnd)):
            r[f"dG_{a}_L32"] = m + rng.normal(0, noise)
        out.append(r)
    return out


def test_rules_at_L32_read_the_flip_gate_and_add_e7b_c11():
    # the predicted pattern: directional vs flip, released edit matched, ours beaten, SPEC live
    r = rules(_rows32(60, e1=20.0, flip=8.0, ed=20.0, fo=10.7, ed_ours=16.2, fo_ours=11.6), SEED, S="L32")
    assert r["set"] == "L32" and r["voiding_gate"] == "E7c (flip)"
    assert r["verdict_E7c"] == "ERASURE-DIRECTIONAL"
    assert r["verdict_E1"] == "ERASURE-MATCHES-EDIT"
    assert r["verdict_E7b"] == "ERASURE-BEATS-EDIT-OURS"
    assert r["verdict_C11"] == "SPEC-LIVE"
    assert r["verdict_E2"] == "ERASURE-WITHIN-SHARE"          # 20 < 0.55*41 = 22.55
    # E3 vs the catastrophic random is still reported but is NOT the voiding gate at L32
    assert r["verdict_E3"] == "ERASURE-DIRECTIONAL"


def test_flip_noise_voids_e1_e2_e7b_at_L32_even_though_random_is_catastrophic():
    r = rules(_rows32(60, e1=20.0, flip=19.5, ed=19.5, fo=10.7, ed_ours=16.2, fo_ours=11.6), SEED, S="L32")
    assert r["verdict_E3"] == "ERASURE-DIRECTIONAL"           # -359 random would have passed it
    assert r["verdict_E7c"] == "ERASURE-NOISE"
    assert all("VOID" in r[k] for k in ("verdict_E1", "verdict_E2", "verdict_E7b"))
    assert "VOID" not in r["verdict_C11"]                      # C11 does not depend on the erasure direction


def test_rules_at_L7_keep_the_2026_09_12_gate():
    rows = _rows(60, e1=32.0, rnd=-15.0, ed=32.0, sw=73.0)
    for q in rows:
        q["dG_erase_flip_L7"] = q["dG_erase_1.0_L7"]          # flip indistinguishable...
    r = rules(rows, SEED)
    assert r["verdict_E7c"] == "ERASURE-NOISE" and r["voiding_gate"] == "E3 (random)"
    assert "VOID" not in r["verdict_E1"]                       # ...yet at 4B the frozen E3 gate decides


def test_anchoring_identity_passes_on_identical_spans_and_fails_on_position_or_vector_drift():
    main = _vectors(n_items=3, spans=2, seed=1)
    by, arr = main
    other = ({k: [dict(q) for q in v] for k, v in by.items()},
             {"h0": arr["h0"] + 1e-4, "h1b": arr["h1b"].copy()})
    g = anchoring_identity(main, other)
    assert g["passes"] and g["n_paired_spans"] == 6 and g["h0_cos_min"] >= 0.999, g
    # a span resolved at a different position in the other root
    other[0]["s1"][0]["positions"] = [5]
    assert not anchoring_identity(main, other)["passes"]
    other[0]["s1"][0]["positions"] = by["s1"][0]["positions"]
    # the same positions but a different captured state (h1b rotated)
    bad = {"h0": other[1]["h0"], "h1b": np.roll(other[1]["h1b"], 1, axis=1)}
    assert not anchoring_identity(main, (other[0], bad))["passes"]
    # a missing item
    o2 = ({k: v for k, v in other[0].items() if k != "s2"}, other[1])
    assert not anchoring_identity(main, o2)["passes"]
    # smoke: the MAIN root holds a subset of the items -> passes only with allow_subset
    m2 = ({k: v for k, v in by.items() if k != "s2"}, arr)
    assert not anchoring_identity(m2, other)["passes"]
    g2 = anchoring_identity(m2, other, allow_subset=True)
    assert g2["passes"] and g2["n_paired_spans"] == 4, g2


def test_identity_gate_checks_the_compare_roots_arms_against_its_own_rows():
    rows = [{"snippet_id": "s1", "dG_edit_L32": 19.5, "dG_foreign_L32": 10.7, "dG_swap_L32": 41.0,
             "dG_erase_own_L32": 41.8, "dG_self_L32": 0.0, "dG_edit_ours_L32": 16.2, "dG_foreign_ours_L32": 11.6}]
    released = {"s1": {"dG_S_edit_L32": 19.5, "dG_S_foreign_L32": 10.7, "dG_S_swap_L32": 41.0}}
    ours = {"s1": {"dG_S_edit_L32": 16.2, "dG_S_foreign_L32": 11.6, "dG_S_swap_L32": 41.0}}
    g = identity(rows, released, S="L32", banked_ours=ours)
    assert g["passes"] and {"edit_ours", "foreign_ours"} <= set(g["gated_arms"]), g
    drifted = [dict(rows[0], dG_edit_ours_L32=16.4)]
    assert not identity(drifted, released, S="L32", banked_ours=ours)["passes"]
    # a failed G1 short-circuits score() into a harness fault before any rule is read
    st = score(rows, released, SEED, S="L32", sets={"L32": ([32], 1.0)}, banked_ours=ours,
               anchoring={"passes": False, "position_mismatches": 3})
    assert st["verdict"] == "E-HARNESS-FAULT" and st["reportable"] is False and "rules" not in st


# --- added 2026-09-13 with the H-E8 prereg (2026-09-13_e8-depth-mirror-prereg.md) ------------------------------
# `all({})` is True, so a layer whose banked rows lacked the banked-byte keys would have passed the identity gate
# VACUOUSLY. A depth sweep across 34 layers is exactly where that fires. Strengthened before any H-E8 data.

def test_identity_gate_refuses_to_pass_vacuously_when_banked_keys_are_absent():
    from nla_erasure import REQUIRED_GATED
    rows = [{"snippet_id": "s1", "dG_erase_1.0_L23": 9.0, "dG_erase_flip_L23": 8.0, "dG_self_L23": 0.0}]
    banked = {"s1": {"dG_S_swap_L7": 73.0}}                 # the WRONG layer's keys: nothing to compare at L23
    g = identity(rows, banked, S="L23")
    assert not g["passes"] and sorted(g["missing_gated"]) == sorted(REQUIRED_GATED), g
    # present and reproducing -> passes, and nothing is reported missing
    rows[0].update({"dG_edit_L23": 17.4, "dG_foreign_L23": 15.7, "dG_swap_L23": 51.8})
    banked["s1"].update({"dG_S_edit_L23": 17.4, "dG_S_foreign_L23": 15.7, "dG_S_swap_L23": 51.8})
    g2 = identity(rows, banked, S="L23")
    assert g2["passes"] and g2["missing_gated"] == [], g2
    # one of the three missing is still a refusal
    del banked["s1"]["dG_S_foreign_L23"]
    assert not identity(rows, banked, S="L23")["passes"]


# --- added 2026-09-13 with the H-E14 prereg (2026-09-13_erasure-oracle-norm-prereg.md) -------------------------
# Every other erasure arm steps by the TRUE per-span ||h0 - h1b||, which is oracle information the trained pair
# never sees. `erase_loonorm` removes it: direction AND magnitude from other items only.

def test_loo_norms_exclude_the_items_own_spans():
    from nla_erasure import loo_norms
    by, arr = _vectors(n_items=5, spans=3)
    D = loo_norms(arr, by)
    dn = np.linalg.norm(arr["h0"].astype(float) - arr["h1b"].astype(float), axis=1)
    for sid, rs in by.items():
        own = {int(r["vec"]) for r in rs}
        want = dn[[k for k in range(len(dn)) if k not in own]].mean()
        assert abs(D[sid] - want) < 1e-6
        # and it is NOT simply the global mean (the item's own spans really are excluded)
        assert abs(D[sid] - dn.mean()) > 1e-9


def test_loonorm_arm_steps_by_the_estimated_not_the_true_norm():
    from nla_erasure import loo_norms
    by, arr = _vectors(n_items=5, spans=3)
    E = loo_means(arr, by); D = loo_norms(arr, by)
    i = by["s0"][0]["vec"]
    h1b, h0 = arr["h1b"][i], arr["h0"][i]
    true_dn = float(np.linalg.norm(h0 - h1b))
    v_oracle = erase_target(h1b, true_dn, E["s0"], 1.0)
    v_loo = erase_target(h1b, D["s0"], E["s0"], 1.0)
    assert abs(np.linalg.norm(v_loo - h1b) - D["s0"]) < 1e-3 * D["s0"]
    assert abs(np.linalg.norm(v_oracle - h1b) - true_dn) < 1e-3 * true_dn
    # same direction, different length: the arms differ only in the magnitude they borrow
    u1 = (v_oracle - h1b) / np.linalg.norm(v_oracle - h1b)
    u2 = (v_loo - h1b) / np.linalg.norm(v_loo - h1b)
    assert float(u1 @ u2) > 0.999


def test_rules_report_what_the_oracle_magnitude_was_worth():
    # a site where removing the oracle costs little -> ORACLE-FREE, and the deployable arm stays directional
    rows = _rows(60, e1=35.75, rnd=21.43, ed=32.06, sw=73.15)
    for q in rows:
        q["dG_erase_flip_L7"] = 22.08 + 0.0
        q["dG_erase_loonorm_L7"] = q["dG_erase_1.0_L7"] - 1.0
    r = rules(rows, SEED)
    assert r["verdict_E14"] == "ORACLE-FREE", r["H_E14"]
    assert r["verdict_E14b"] == "DEPLOYABLE-DIRECTIONAL", r["H_E14b"]
    # a site where it costs a lot -> ORACLE-SCALED, and the deployable arm is noise
    for q in rows:
        q["dG_erase_loonorm_L7"] = q["dG_erase_1.0_L7"] - 14.0
    r2 = rules(rows, SEED)
    assert r2["verdict_E14"] == "ORACLE-SCALED"
    assert r2["verdict_E14b"] == "DEPLOYABLE-NOISE"
