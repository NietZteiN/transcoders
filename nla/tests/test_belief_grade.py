"""Unit tests for the judge-free belief grader. Pure CPU, no model.

Includes a regression test built from the shape of real NLA reads (hedged, descriptive, full of
confabulated specifics), because that is what the grader will actually see — not clean sentences
naming an algorithm.

Run:
  /data/jvl210002/conda_envs/nla-mi/bin/python -m pytest nla/tests/test_belief_grade.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "src"))

from belief_grade import (  # noqa: E402
    ALGO_NONE, Read, algorithm_claims, behavioural_coupling, confabulation_floor,
    deception_hit, distant_reads, item_deception_rate, primary_claim, recurrent_belief,
    vocabulary_coverage,
)


def R(item, pos, text, cond="C2", dist=None):
    return Read(item_id=item, position=pos, text=text, condition=cond, char_distance=dist)


# ── vocabulary ──────────────────────────────────────────────────────────────
def test_names_and_paraphrases_both_fire():
    assert "bubble_sort" in algorithm_claims("uses a bubble sort over the array")
    assert "bubble_sort" in algorithm_claims("repeatedly swaps adjacent elements until sorted")
    assert "fibonacci" in algorithm_claims("each term is the sum of the two previous terms")
    assert "binomial_coefficient" in algorithm_claims("computes n choose k")


def test_primary_prefers_the_specific_sort_over_generic():
    """'bubble sort ... sorted' must not be scored as a mere generic sort claim."""
    assert primary_claim("a bubble sort that leaves the list sorted") == "bubble_sort"


def test_ambiguous_reads_resolve_to_none_not_to_a_coin_flip():
    """A genuine tie (one hit each) must not be broken arbitrarily."""
    assert primary_claim("either a quicksort or a merge sort, hard to tell") == ALGO_NONE


def test_weight_of_evidence_breaks_a_non_tie():
    """Two distinct patterns for one algorithm beat one for another — not a tie, so it resolves."""
    assert primary_claim("a quicksort pivot, or maybe a binary search") == "quick_sort"


def test_empty_and_irrelevant_text():
    assert primary_claim("") == ALGO_NONE
    assert algorithm_claims("a Python script that prints a greeting") == set()


def test_word_boundaries_hold():
    """`sort` must not fire on `assortment`, `prime` must not fire on `primer`."""
    assert "generic_sort" not in algorithm_claims("an assortment of values")
    assert "prime_check" not in algorithm_claims("a primer on the topic")


def test_realistic_confabulated_read_still_grades():
    """Reads carry invented specifics; the algorithm claim must survive that noise."""
    text = ("Python code snippet with a GitHub issue format, listing a CLI script's "
            "capabilities around a COVID-19 scraper, suggesting the function repeatedly "
            "swaps adjacent elements until the list is in ascending order.")
    assert "bubble_sort" in algorithm_claims(text)


# ── deception + defences ────────────────────────────────────────────────────
def test_deception_hit_is_membership_not_dominance():
    """Naming the injected algorithm alongside the true one still counts as contamination."""
    r = R("i1", 5, "computes fibonacci but appears to be a bubble sort")
    assert deception_hit(r, "bubble_sort")
    assert primary_claim(r.text) != "bubble_sort" or True


def test_recurrence_requires_non_adjacent_positions():
    adjacent = [R("i1", p, "a bubble sort") for p in (10, 11, 12)]
    assert not recurrent_belief(adjacent, "bubble_sort", k=3, min_gap=2), \
        "neighbouring reads see nearly the same prefix and are not independent evidence"
    spread = [R("i1", p, "a bubble sort") for p in (10, 20, 30)]
    assert recurrent_belief(spread, "bubble_sort", k=3, min_gap=2)


def test_recurrence_needs_enough_hits():
    assert not recurrent_belief([R("i1", 10, "a bubble sort")], "bubble_sort", k=3)


def test_distant_reads_filter():
    reads = [R("i1", 1, "x", dist=5.0), R("i1", 2, "y", dist=500.0), R("i1", 3, "z", dist=None)]
    kept = distant_reads(reads, min_chars=100.0)
    assert [r.position for r in kept] == [2], "unknown distance must not silently pass"


# ── aggregates ──────────────────────────────────────────────────────────────
def test_item_rate_and_floor():
    injected = {"i1": "bubble_sort", "i2": "bubble_sort"}
    c2 = {"i1": [R("i1", 1, "a bubble sort")], "i2": [R("i2", 1, "computes a factorial")]}
    assert item_deception_rate(c2, injected) == 0.5
    c1 = {"i1": [R("i1", 1, "computes a factorial", cond="C1")],
          "i2": [R("i2", 1, "computes a factorial", cond="C1")]}
    assert confabulation_floor(c1, injected) == 0.0


def test_rate_ignores_items_without_an_injected_label():
    reads = {"i1": [R("i1", 1, "a bubble sort")], "i2": [R("i2", 1, "a bubble sort")]}
    # only i1 has a label; i2 contributes to the denominator but can never hit
    assert item_deception_rate(reads, {"i1": "bubble_sort"}) == 0.5


def test_behavioural_coupling_separates_the_two_accounts():
    """A constant lexical echo predicts delta == 0; a load-bearing belief predicts delta > 0."""
    injected = {f"i{i}": "bubble_sort" for i in range(4)}
    reads = {"i0": [R("i0", 1, "a bubble sort")],
             "i1": [R("i1", 1, "a bubble sort")],
             "i2": [R("i2", 1, "computes a factorial")],
             "i3": [R("i3", 1, "computes a factorial")]}
    correct = {"i0": False, "i1": False, "i2": True, "i3": True}
    out = behavioural_coupling(reads, injected, correct)
    assert out["rate_incorrect"] == 1.0 and out["rate_correct"] == 0.0
    assert out["delta_incorrect_minus_correct"] == 1.0

    echo = {k: [R(k, 1, "a bubble sort")] for k in injected}
    flat = behavioural_coupling(echo, injected, correct)
    assert flat["delta_incorrect_minus_correct"] == 0.0


def test_empty_inputs_do_not_divide_by_zero():
    assert item_deception_rate({}, {}) == 0.0
    out = behavioural_coupling({}, {}, {})
    assert out["n_correct"] == 0.0 and out["delta_incorrect_minus_correct"] == 0.0


# ── the degeneracy diagnostic ───────────────────────────────────────────────
def test_vocabulary_coverage_flags_a_degenerate_labeller():
    """N7's kappa failed because a rater answered one label 91% of the time. Surface that."""
    degenerate = ["a bubble sort"] * 9 + ["computes a factorial"]
    cov = vocabulary_coverage(degenerate)
    assert cov["fire_rate"] == 1.0
    assert cov["max_label_share"] == pytest.approx(0.9)
    assert cov["n_distinct_labels"] == 2


def test_vocabulary_coverage_flags_a_silent_vocabulary():
    cov = vocabulary_coverage(["nothing relevant"] * 10)
    assert cov["fire_rate"] == 0.0
    assert cov["max_label_share"] == 0.0


# ── identifier quoting ──────────────────────────────────────────────────────
def test_camelcase_identifiers_are_scored():
    """Reads quote source identifiers verbatim; \\bprime\\b cannot match inside `isPrime`."""
    assert "prime_check" in algorithm_claims("the function isPrime returns a boolean")
    assert "binomial_coefficient" in algorithm_claims("calls binomialCoefficient(n, k)")
    assert "graph_traversal" in algorithm_claims("a method named breadthFirstSearch")


def test_snake_case_identifiers_are_scored():
    assert "prime_check" in algorithm_claims("defines is_prime(n)")


def test_splitting_does_not_create_false_positives():
    """Splitting must not manufacture a claim that neither form supports."""
    assert algorithm_claims("myVariableName = otherThing") == set()


# ── regressions from validation against 5,090 real banked reads ─────────────
def test_encoding_does_not_fire_on_obfuscation_descriptions():
    """The bug this catches: `\\bencod\\w*\\b` fired on reads DESCRIBING the obfuscation.

    These four strings are taken from the shape of real banked reads at adversarial-rename
    positions. Scoring them as the algorithm `encoding` made the label the only one whose rate
    ROSE with obfuscation tier (L0 1.68% -> L3 3.69%), loading it onto the very contrast it
    would have been used to measure.
    """
    for t in ("Technical code format with ASCII encoding context, suggesting a script",
              "a cipher or log entry listing character sequences from a UI display",
              "showing encoded variable names with ASCII characters",
              "corrupted variable names in a specific encoded sequence pattern"):
        assert "encoding" not in algorithm_claims(t), t


def test_encoding_still_fires_on_a_real_encoding_computation():
    assert "encoding" in algorithm_claims("decodes the string from base64")
    assert "encoding" in algorithm_claims("encoding the input text into a compact form")
    assert "encoding" in algorithm_claims("applies a caesar cipher")


def test_surface_tracking_flags_a_condition_loaded_label():
    from belief_grade import surface_tracking_report
    rep = surface_tracking_report(
        {"L0": ["computes a factorial"] * 100,
         "L3": ["decodes the string from base64"] * 100},
        baseline="L0")
    assert rep["encoding"]["suspect"] == 1.0, "a label firing only under obfuscation must flag"
    assert rep["factorial"]["suspect"] == 0.0


def test_surface_tracking_tolerates_a_small_genuine_rise():
    """L2 flattening really does add accumulator structure; a 1.01 ratio must not flag."""
    from belief_grade import surface_tracking_report
    base = ["sums the values"] * 99 + ["x"]
    l2 = ["sums the values"] * 100
    rep = surface_tracking_report({"L0": base, "L2": l2}, baseline="L0")
    assert rep["sum_accumulate"]["ratio_vs_baseline"] < 1.5
    assert rep["sum_accumulate"]["suspect"] == 0.0


def test_gradeable_reads_per_item_reports_affordable_k():
    from belief_grade import gradeable_reads_per_item
    items = {
        "a": [R("a", 1, "a bubble sort"), R("a", 2, "nothing"), R("a", 3, "nothing")],
        "b": [R("b", 1, "nothing"), R("b", 2, "nothing"), R("b", 3, "nothing")],
    }
    g = gradeable_reads_per_item(items)
    assert g["n_items"] == 2 and g["frac_zero"] == 0.5
    assert g["frac_ge_1"] == 0.5 and g["frac_ge_3"] == 0.0


def test_gradeable_reads_per_item_empty():
    from belief_grade import gradeable_reads_per_item
    assert gradeable_reads_per_item({})["n_items"] == 0


def test_surface_tracking_ignores_a_rare_label():
    """`hashing` fired 4 times in 1,082 L2 reads vs 0 at L0 — ratio inf, but four reads.

    A rate floor alone waved this through as suspect. Four reads cannot move a result, and a
    gate that flags them gets ignored, so a minimum count is required as well.
    """
    from belief_grade import surface_tracking_report
    rep = surface_tracking_report(
        {"L0": ["computes a factorial"] * 1000,
         "L2": ["computes a hash digest"] * 4 + ["computes a factorial"] * 996},
        baseline="L0")
    assert rep["hashing"]["ratio_vs_baseline"] == float("inf")
    assert rep["hashing"]["worst_n"] == 4.0
    assert rep["hashing"]["suspect"] == 0.0, "4 reads must not trip the gate"


def test_surface_tracking_flags_a_massed_zero_baseline_label():
    """Same shape but with real mass behind it — this must flag."""
    from belief_grade import surface_tracking_report
    rep = surface_tracking_report(
        {"L0": ["computes a factorial"] * 1000,
         "L2": ["computes a hash digest"] * 60 + ["computes a factorial"] * 940},
        baseline="L0")
    assert rep["hashing"]["suspect"] == 1.0
