"""Unit tests for the N13 torn-ness instruments. Pure CPU, no model, no GPU.

Every function tested here decides something the GPU stages cannot check for themselves: how a
split answer distribution becomes an entropy number, whether K readings of one vector count as
agreeing, whether a stratified frame is actually stratified, and whether the length matcher
enforces the ratio it claims. A bug in any of them is invisible in the run logs and only shows up
as a wrong result, which is the failure mode this programme has been bitten by most.

Run:
  /data/jvl210002/conda_envs/nla-mi/bin/python -m pytest nla/tests/test_n13.py -q
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "src"))
sys.path.insert(0, str(_HERE.parent.parent / "src" / "analysis"))

from answer_entropy import entropy, summarize  # noqa: E402
from read_instability import (  # noqa: E402
    QUOTA, build_frame, content_words, jaccard, pairwise_stats,
)


# ── Stage 1: entropy ──────────────────────────────────────────────────────────
def test_entropy_zero_when_unanimous():
    assert entropy([8]) == 0.0


def test_entropy_one_when_maximally_split():
    # 8 distinct answers out of 8 samples is total disagreement -> normalized to exactly 1
    assert entropy([1] * 8) == pytest.approx(1.0)


def test_entropy_monotone_in_splitness():
    assert entropy([7, 1]) < entropy([4, 4]) < entropy([2, 2, 2, 2])


def test_entropy_normalizer_is_log_k_not_log_n_distinct():
    """A 4-way split must outrank a 2-way split.

    Dividing by log(n_distinct) would map both to 1.0 and erase the very thing being measured.
    """
    assert entropy([2, 2, 2, 2]) > entropy([4, 4])
    assert entropy([4, 4]) == pytest.approx(math.log(2) / math.log(8))


def _task(truth="927"):
    return {"kind": "output_prediction", "truth": truth, "expected_output": truth}


def test_summarize_unanimous_correct():
    s = summarize(["927"] * 8, _task())
    assert s["n_distinct"] == 1 and s["entropy"] == 0.0
    assert s["modal_correct"] and s["modal_share"] == 1.0 and s["frac_correct"] == 1.0


def test_summarize_uses_harness_normalization():
    """'9 27' / \"'927'\" / '927' are one answer, because grading treats them as one."""
    s = summarize(["927", " 927 ", "'927'", "`927`"], _task())
    assert s["n_distinct"] == 1
    assert s["modal_correct"]


def test_summarize_torn_wrong_is_distinguishable_from_committed_wrong():
    torn = summarize(["1", "2", "3", "4", "5", "6", "7", "8"], _task())
    committed = summarize(["480"] * 8, _task())
    assert not torn["modal_correct"] and not committed["modal_correct"]
    # identical on correctness, opposite on the axis Stage 1 exists to add
    assert torn["entropy"] > committed["entropy"]
    assert committed["entropy"] == 0.0


def test_summarize_unparsed_is_its_own_outcome():
    s = summarize(["927", None, None, "927"], _task())
    assert s["n_unparsed"] == 2
    assert s["n_distinct"] == 2          # "927" and <UNPARSED>
    assert s["frac_correct"] == 0.5


def test_summarize_any_correct_vs_modal_correct_can_disagree():
    """The 'lucky/unstable' cell: right once, but not the plurality answer."""
    s = summarize(["1", "1", "1", "927"], _task())
    assert s["any_correct"] and not s["modal_correct"]


# ── Stage 3: read agreement ───────────────────────────────────────────────────
def test_pairwise_identical_vectors_agree():
    v = [np.array([1.0, 2.0, 3.0])] * 4
    s = pairwise_stats(v, ["sorts the list"] * 4)
    assert s["ar_cos_mean"] == pytest.approx(1.0)
    assert s["n_distinct_texts"] == 1


def test_pairwise_orthogonal_vectors_disagree():
    v = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
    s = pairwise_stats(v, ["sorts a list", "checks primality"])
    assert s["ar_cos_mean"] == pytest.approx(0.0)


def test_ar_space_and_lexical_can_diverge():
    """The reason agreement is measured in AR space rather than on strings.

    Two paraphrases share few words but mean the same thing; AR cosine should see that while
    Jaccard does not. This is the real smoke-test observation (jaccard 0.35, ar_cos 0.98).
    """
    v = [np.array([1.0, 0.0, 0.0]), np.array([0.999, 0.045, 0.0])]
    s = pairwise_stats(v, ["computes the fibonacci recurrence",
                           "recursive integer sequence accumulation"])
    assert s["ar_cos_mean"] > 0.99
    assert s["jaccard_mean"] < 0.2


def test_jaccard_and_stopwords():
    assert jaccard(set(), set()) == 1.0
    # stopwords must not manufacture agreement between unrelated readings
    assert "the" not in content_words("The function is a list")
    assert content_words("sorts the array") == {"sorts", "array"}


# ── Stage 3: the sampling frame ───────────────────────────────────────────────
def _fake_enriched(n_cases=40):
    classes = list(QUOTA)
    out = []
    for i in range(n_cases):
        reads = []
        for j, c in enumerate(classes):
            for k in range(6):
                reads.append({"position": j * 100 + k, "cls": c, "rt_cos": 0.87,
                              "read": "x", "anchor": {"tok": "t", "in_reply": False}})
        out.append({"task_key": f"case{i:03d}", "tier": ["L0", "L1", "L1b", "L2", "L3"][i % 5],
                    "kind": "output_prediction", "correct": i % 2 == 0, "reads": reads})
    return out


def test_build_frame_respects_quotas():
    f = build_frame(_fake_enriched(), None, QUOTA, 20260724)
    got = {}
    for x in f:
        got[x["cls"]] = got.get(x["cls"], 0) + 1
    for cls, want in QUOTA.items():
        assert got[cls] == want, f"{cls}: {got[cls]} != {want}"


def test_build_frame_is_shuffled_so_partial_runs_stay_balanced():
    """A wall-clock cutoff must not yield 58 fn_orig and nothing else.

    The frame is built class-by-class; unshuffled, any prefix is a single class and the floor
    gate — which needs every class present — would be uncomputable on a partial run.
    """
    f = build_frame(_fake_enriched(), None, QUOTA, 20260724)
    prefix = {x["cls"] for x in f[:60]}
    assert len(prefix) >= 3


def test_build_frame_is_deterministic_under_seed():
    a = build_frame(_fake_enriched(), None, QUOTA, 20260724)
    b = build_frame(_fake_enriched(), None, QUOTA, 20260724)
    assert [(x["case"], x["position"]) for x in a] == [(x["case"], x["position"]) for x in b]


def test_build_frame_spreads_over_tiers():
    f = build_frame(_fake_enriched(), None, QUOTA, 20260724)
    adv = {x["tier"] for x in f if x["cls"] == "adversarial"}
    assert len(adv) >= 3, "a class sampled from one tier corner defeats the stratification"


def test_build_frame_handles_none_tier():
    """Slice items carry tier=None; sorting bucket keys crashed on that in the first draft."""
    e = _fake_enriched(10)
    for r in e[:4]:
        r["tier"] = None
    f = build_frame(e, None, QUOTA, 20260724)
    assert len(f) > 0


# ── Stage 4: matching and the horse race ──────────────────────────────────────
def test_match_on_length_enforces_ratio():
    from n13_torn import MAX_LOG_RATIO, match_on_length
    a = [{"task_key": "a1", "banked_reply_tokens": 1000}]
    b = [{"task_key": "b1", "banked_reply_tokens": 100}]     # 10x — far outside log(1.5)
    assert match_on_length(a, b) == []
    b2 = [{"task_key": "b2", "banked_reply_tokens": 900}]
    pairs = match_on_length(a, b2)
    assert len(pairs) == 1
    assert abs(math.log(1000 / 900)) <= MAX_LOG_RATIO


def test_match_on_length_is_one_to_one():
    from n13_torn import match_on_length
    a = [{"task_key": f"a{i}", "banked_reply_tokens": 500} for i in range(3)]
    b = [{"task_key": "b1", "banked_reply_tokens": 500}]
    pairs = match_on_length(a, b)
    assert len(pairs) == 1, "a control must not be reused across pairs"


def test_horse_race_detects_a_real_increment():
    """A predictor that genuinely drives the outcome must be flagged."""
    from n13_torn import horse_race
    rng = np.random.default_rng(0)
    rows = []
    for i in range(200):
        length = float(rng.normal(6, 0.5))
        signal = float(rng.normal(0, 1))
        rows.append({"entropy": 0.3 * length + 0.6 * signal + float(rng.normal(0, 0.1)),
                     "log_reply": length, "mean_rt_cos": signal})
    out = horse_race(rows, ["log_reply", "mean_rt_cos"], n_perm=200)
    inc = [l for l in out["ladder"] if l["added"] == "mean_rt_cos"][0]
    assert inc["delta_r2"] > 0.3
    assert inc["p_perm"] < 0.01 and inc["beats_baseline"]
    assert inc["coef_ci95"][0] > 0          # slope CI clears zero for a real effect


def test_horse_race_rejects_pure_noise():
    """A predictor unrelated to the outcome must NOT be flagged.

    This is the regression test for the bug the dress rehearsal caught. delta R^2 is
    non-negative by construction — adding any column to an OLS fit cannot lower in-sample R^2 —
    so a bootstrap CI on the increment is bounded below by ~0 and "excludes zero" even for pure
    noise. The permutation p-value is the test that actually discriminates, and the coefficient
    CI is free to contain zero.
    """
    from n13_torn import horse_race
    rng = np.random.default_rng(1)
    rows = [{"entropy": float(rng.normal()), "log_reply": float(rng.normal()),
             "mean_rt_cos": float(rng.normal())} for _ in range(200)]
    out = horse_race(rows, ["log_reply", "mean_rt_cos"], n_perm=200)
    inc = [l for l in out["ladder"] if l["added"] == "mean_rt_cos"][0]
    assert inc["delta_r2"] >= 0            # the property that made the old test vacuous
    assert inc["p_perm"] > 0.05 and not inc["beats_baseline"]
    assert inc["coef_ci95"][0] < 0 < inc["coef_ci95"][1]


def test_horse_race_noise_still_buys_some_r2():
    """Documents WHY the permutation null is needed: noise always buys a little R^2."""
    from n13_torn import horse_race
    rng = np.random.default_rng(7)
    rows = [{"entropy": float(rng.normal()), "log_reply": float(rng.normal()),
             "mean_rt_cos": float(rng.normal())} for _ in range(60)]
    out = horse_race(rows, ["log_reply", "mean_rt_cos"], n_perm=200)
    inc = [l for l in out["ladder"] if l["added"] == "mean_rt_cos"][0]
    assert inc["delta_r2"] > 0, "a strictly positive increment from a predictor that is noise"
    assert not inc["beats_baseline"]


def test_cluster_bootstrap_resamples_cases_not_reads():
    """Many reads from ONE case must not look like independent evidence.

    With a single case per stratum there is no between-case variance, so a case-level bootstrap
    must return a degenerate (zero-width) interval. A read-level bootstrap would instead return
    a narrow-but-nonzero CI and manufacture confidence.
    """
    from n13_torn import cluster_bootstrap_diff
    g = {"adversarial": [("c1", 0.5)] * 40, "l1_neutral": [("c2", 0.1)] * 40}
    out = cluster_bootstrap_diff(g, n_boot=200)
    assert out["n_cases_adversarial"] == 1
    assert out["diff"] == pytest.approx(0.4)
    assert out["diff_ci95"][0] == out["diff_ci95"][1] == pytest.approx(0.4)


def test_cluster_bootstrap_detects_a_real_difference():
    from n13_torn import cluster_bootstrap_diff
    rng = np.random.default_rng(3)
    a = [(f"ca{i}", float(rng.normal(0.5, 0.05))) for i in range(30)]
    b = [(f"cb{i}", float(rng.normal(0.1, 0.05))) for i in range(30)]
    out = cluster_bootstrap_diff({"adversarial": a, "l1_neutral": b}, n_boot=500)
    assert out["excludes_zero"]
    assert out["diff"] > 0.3


def test_cluster_bootstrap_null_difference_includes_zero():
    from n13_torn import cluster_bootstrap_diff
    rng = np.random.default_rng(4)
    a = [(f"ca{i}", float(rng.normal(0.3, 0.1))) for i in range(40)]
    b = [(f"cb{i}", float(rng.normal(0.3, 0.1))) for i in range(40)]
    out = cluster_bootstrap_diff({"adversarial": a, "l1_neutral": b}, n_boot=500)
    assert not out["excludes_zero"]
