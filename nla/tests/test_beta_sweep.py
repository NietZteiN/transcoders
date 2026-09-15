"""CPU tests for the H-S5..H-S9 sweep: layer-set construction and the frozen verdict table.

No model, no GPU, no banked artifacts — `nla_beta_sweep.score()` is pure over its rows, so the
verdict logic can be driven with synthetic rows whose answer is known by construction. This is the
class of bug that has cost this thread whole jobs before (two zero-position bugs produced three
artifact verdicts in one day on 2026-09-07), so the rules are tested before the scheduler sees them.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_SRC))

from nla_beta_sweep import (CORE_ARMS, coverage_set, half, resolve_sets,  # noqa: E402
                           score, spaced_set)

LIVE = list(range(1, 34))
BL = [0.1, 1.0]
KL = [1, 4]
NM = ["faithful", "all_live", "live_no_tail"]
VERIFY = [3, 7]

CFG = {
    "experiment": "S_beta_layerset_test",
    "seed": 20260724,
    "n_boot": 400,                       # small: these tests assert verdicts, not CI width
    "identity": {"beta1_tol_nats": 0.05, "beta0_tol_nats": 0.05,
                 "beta0_requires_same_positions": True, "self_tol_nats": 1.0,
                 "all_live_c3_banked_nats": 54.06, "verify_layers_per_half": 2},
    "rules": {"hs5_additive_ratio": 1.64, "hs5_saturating_ratio": 1.25,
              "hs6_ceiling_k1_nats": 73.15, "hs6_repaired_frac": 0.95,
              "hs7_min_nats": 3.0, "hs9_gate_min_nats": 3.0},
}


def _rows(spec_k1: float, spec_k4: float, *, swap_b01: float = 40.0, swap_b1: float = 40.0,
          spec_sel: tuple[float, float] = (10.0, 6.0), beta0_dG: float = 0.0,
          n: int = 40) -> list[dict]:
    """Synthetic rows with SPEC planted exactly: `edit - foreign` is the quantity score() reads.

    A tiny deterministic per-item wobble keeps the bootstrap from producing a zero-width CI (which
    would make every rule trivially pass and hide a sign error).
    """
    rows = []
    for i in range(n):
        sid = f"item/{i}"
        w = 0.20 * ((i % 5) - 2)                      # mean-zero jitter
        r: dict = {"snippet_id": sid, "half": i % 2, "n_tok": 500, "logp_U": -100.0}
        for k, sp in ((1, spec_k1), (4, spec_k4)):
            r[f"layers_topk{k}"] = sorted(LIVE[:k])
            for b in BL:
                t = f"topk{k}_b{b}"
                r[f"dG_edit_{t}"] = sp + w
                r[f"dG_foreign_{t}"] = 0.0
                for a in ("rt", "c3", "swap"):
                    r[f"dG_{a}_{t}"] = 20.0
                if k == max(KL):
                    r[f"dG_random_{t}"] = -300.0
        for name in NM:
            r[f"layers_{name}"] = sorted(LIVE[:12]) if name == "faithful" else sorted(LIVE)
            for b in BL:
                t = f"{name}_b{b}"
                r[f"dG_edit_{t}"] = spec_k4 + w
                r[f"dG_foreign_{t}"] = 0.0
                r[f"dG_rt_{t}"] = 15.0
                r[f"dG_c3_{t}"] = 30.0
                r[f"dG_swap_{t}"] = (swap_b01 if b == 0.1 else swap_b1) + w
                if name in ("all_live", "live_no_tail"):
                    r[f"dG_random_{t}"] = -300.0
                    r[f"dG_self_{t}"] = 0.0
        for nm2, sp in (("specrank1", spec_sel[0]), ("c3rank1", spec_sel[1])):
            r[f"layers_{nm2}"] = [3] if nm2 == "specrank1" else [7]
            r[f"dG_edit_{nm2}_b1.0"] = sp + w
            r[f"dG_foreign_{nm2}_b1.0"] = 0.0
        for K in VERIFY:
            for a in ("edit", "foreign"):
                r[f"dG_{a}_verifyL{K}_b1.0"] = 10.0 + K
        for a in CORE_ARMS:
            r[f"dG_{a}_beta0_faithful"] = beta0_dG
        rows.append(r)
    return rows


def _gby(rows) -> dict:
    """The banked gate rows the beta=1 identity gate compares against — matching by construction."""
    return {r["snippet_id"]: {f"dG_S_{a}_L{K}": 10.0 + K for K in VERIFY for a in ("edit", "foreign")}
            for r in rows}


def _named():
    return {"faithful": {"layers": LIVE[:12], "selected": False},
            "all_live": {"layers": list(LIVE), "selected": False},
            "live_no_tail": {"layers": list(LIVE), "selected": False}}


def _score(rows, **kw):
    rank = {"spec": {0: LIVE, 1: LIVE}, "c3": {0: LIVE, 1: LIVE}}
    ed = {K: {("x", K)} for K in LIVE}
    return score(rows, kw.pop("cfg", CFG), LIVE, _named(), rank, ed, _gby(rows), VERIFY,
                 BL, KL, NM, 471)


# ── layer-set construction ────────────────────────────────────────────────────────────────────
def test_spaced_set_is_evenly_spaced_and_deduplicated():
    s = spaced_set(LIVE, 12)
    assert len(s) == 12 and s == sorted(set(s))
    assert s[0] == LIVE[0] and s[-1] == LIVE[-1], "a spaced set must span the live range"
    assert spaced_set(LIVE, 1) == [LIVE[len(LIVE) // 2]]


def test_coverage_set_grows_the_editable_union_monotonically():
    """The greedy must be deterministic and must never pick a layer that adds nothing while one
    that adds something is still available — that is the whole point of the COVERAGE arm."""
    ed = {1: {"a", "b"}, 2: {"a", "b"}, 3: {"c"}, 4: {"d", "e"}, 5: set()}
    live = [1, 2, 3, 4, 5]
    assert coverage_set(ed, live, 1) == [1]                  # biggest single set, ties by low index
    got = coverage_set(ed, live, 3)
    u = set().union(*(ed[K] for K in got))
    assert u == {"a", "b", "c", "d", "e"}, f"greedy left coverage on the table: {got}"
    assert coverage_set(ed, live, 3) == got, "greedy is not deterministic"
    assert len(coverage_set(ed, live, 99)) == len(live), "k above |live| must clamp, not raise"


def test_resolve_sets_maps_every_config_spelling():
    cfg = {"sets": {"a": "all", "b": "spaced12", "c": "coverage12", "d": "spec_rank12",
                    "e": [2, 3, 99]}}
    ed = {K: {("x", K)} for K in LIVE}
    out = resolve_sets(cfg, LIVE, ed)
    assert out["a"]["layers"] == LIVE
    assert len(out["b"]["layers"]) == 12
    assert len(out["c"]["layers"]) == 12
    assert out["d"]["layers"] is None and out["d"]["selected"] and out["d"]["k"] == 12
    assert out["e"]["layers"] == [2, 3], "a set must drop layers that are not live"


def test_half_is_the_same_split_as_every_other_run_in_this_family():
    import zlib
    for sid in ("humaneval-x-python/Python/100", "JavaScript/63", "cruxeval-x-python/0"):
        assert half(sid) == (zlib.crc32(sid.encode()) & 1)


# ── the frozen verdict table ──────────────────────────────────────────────────────────────────
def test_hs5_additive_when_the_specific_effect_doubles():
    st = _score(_rows(spec_k1=5.0, spec_k4=10.0))          # ratio 2.0 >= 1.64
    assert st["H_S5"]["verdict"] == "SPEC-ADDITIVE"
    assert st["H_S5"]["best_k"] == 4
    assert st["H_S5"]["ratio_best_over_k1"] == pytest.approx(2.0, abs=0.02)
    assert st["H_S5"]["paired_best_minus_k1"]["ci95"][0] > 0


def test_hs5_saturating_in_the_middle_band():
    st = _score(_rows(spec_k1=5.0, spec_k4=7.0))            # ratio 1.4 in [1.25, 1.64)
    assert st["H_S5"]["verdict"] == "SPEC-SATURATING"


def test_hs5_non_additive_when_flat():
    st = _score(_rows(spec_k1=5.0, spec_k4=5.0))
    assert st["H_S5"]["verdict"] == "SPEC-NON-ADDITIVE"


def test_hs5_non_additive_when_more_layers_are_worse():
    """The H-S1 result must still be expressible: a DROP is never dressed up as an effect."""
    st = _score(_rows(spec_k1=10.0, spec_k4=4.0))
    assert st["H_S5"]["verdict"] == "SPEC-NON-ADDITIVE"
    assert st["H_S5"]["best_k"] == 1


def test_hs6_repaired_only_when_beta_lifts_swap_to_the_k1_ceiling():
    # beta=0.1 restores the ceiling (>= 0.95 * 73.15 = 69.49), beta=1 does not
    st = _score(_rows(5.0, 5.0, swap_b01=70.0, swap_b1=58.0))
    assert st["H_S6"]["verdict"] == "CEILING-REPAIRED"
    assert st["H_S6"]["best_beta"] == 0.1
    # ...and a beta that helps but falls short of the bar must NOT be called repaired
    st2 = _score(_rows(5.0, 5.0, swap_b01=62.0, swap_b1=58.0))
    assert st2["H_S6"]["verdict"] == "NOT-REPAIRED"


def test_hs7_selection_matters_only_past_the_frozen_margin():
    st = _score(_rows(5.0, 5.0, spec_sel=(12.0, 6.0)))      # +6.0 >= 3.0
    assert st["H_S7"]["verdict"] == "SELECTION-MATTERS"
    st2 = _score(_rows(5.0, 5.0, spec_sel=(7.0, 6.0)))      # +1.0 < 3.0
    assert st2["H_S7"]["verdict"] == "SELECTION-IMMATERIAL"


def test_identity_fails_loudly_when_beta0_is_not_a_value_noop():
    """Gate 2: a beta=0 that moves the logits means the interpolation is wired wrong. Non-reportable."""
    st = _score(_rows(5.0, 10.0, beta0_dG=0.9))
    assert st["identity_passes"] is False
    assert st["reportable"] is False
    assert any(not v["passes"] for v in st["identity"]["beta0"].values())


def test_identity_fails_when_beta1_does_not_reproduce_the_banked_rows():
    """Gate 1: beta=1 is supposed to BE the banked replacement; drift means the default path moved."""
    rows = _rows(5.0, 10.0)
    for r in rows:
        r["dG_edit_verifyL3_b1.0"] += 2.0
    st = _score(rows)
    assert st["identity_passes"] is False
    assert st["identity"]["beta1_vs_banked"]["L3_edit"]["passes"] is False


def test_identity_passes_on_a_clean_run_and_the_gate_reports_a_gap():
    st = _score(_rows(5.0, 10.0))
    assert st["identity_passes"] is True and st["reportable"] is True
    assert st["H_S9_gate"]["fires"] in (True, False)
    assert st["H_S9_gate"]["best_config"] is not None
    assert st["n_items"] == 40 and st["summary"], "the summary lines are what the sbatch log shows"


# ── H-C4: --train-frac must subsample TRAIN only, reproducibly ────────────────────────────────
def test_subsample_train_is_seeded_fractional_and_leaves_the_holdout_alone():
    """The whole point of `--train-frac` over `--limit`: the eval set must not move.

    `--limit` truncates the holdout to 32/64 rows, which would make `holdout_fve` incomparable with
    the banked 858/848-row numbers and turn the dose-response into a noise measurement.
    """
    import sys as _s
    from pathlib import Path as _P
    _s.path.insert(0, str(_P(__file__).resolve().parents[1] / "src"))
    from nla_train import subsample_train

    cfg = {"seed": 20260724}
    tr = [{"i": i} for i in range(1000)]
    assert subsample_train(tr, cfg, 7, None, "av") is tr, "None must be a no-op"
    assert subsample_train(tr, cfg, 7, 1.0, "av") is tr, "1.0 must be a no-op"
    q = subsample_train(tr, cfg, 7, 0.25, "av")
    assert len(q) == 250
    # reproducible for a given (layer, side, frac), and NOT a prefix (rows are grouped by document,
    # so a prefix would subsample documents and confound volume with topic coverage)
    assert [r["i"] for r in q] == [r["i"] for r in subsample_train(tr, cfg, 7, 0.25, "av")]
    assert [r["i"] for r in q] != list(range(250)), "must not be a prefix"
    assert [r["i"] for r in q] == sorted(r["i"] for r in q), "kept rows stay in corpus order"
    # different layer or side draws a different subset
    assert [r["i"] for r in q] != [r["i"] for r in subsample_train(tr, cfg, 7, 0.25, "ar")]
    assert [r["i"] for r in q] != [r["i"] for r in subsample_train(tr, cfg, 22, 0.25, "av")]
    # nesting is not required, but the 50 % draw must be a valid larger sample
    assert len(subsample_train(tr, cfg, 7, 0.5, "av")) == 500
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="train-frac"):
            subsample_train(tr, cfg, 7, bad, "av")
