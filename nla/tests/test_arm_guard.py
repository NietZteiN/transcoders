"""Tests for the zero-written-position guard. CPU-only, no model.

These encode the two 2026-09-06 bugs directly, so a regression reproduces them as failures.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from arm_guard import ArmNotWritten, arm_series, paired, record_positions  # noqa: E402


def rows(n=49, zero_from=None, mismatch=False):
    out = []
    for i in range(n):
        wrote = zero_from is None or i < zero_from
        out.append({"dG_A": 30.0 if wrote else 0.0, "dG_B": 8.0 if wrote else 0.0,
                    "n_pos_A": 12 if wrote else 0,
                    "n_pos_B": (13 if mismatch else 12) if wrote else 0})
    return out


def test_record_positions_stamps_every_arm():
    r = record_positions({}, {"A": {1: None, 2: None}, "B": {}})
    assert r == {"n_pos_A": 2, "n_pos_B": 0}


def test_bug9_zero_position_items_are_dropped_not_averaged():
    """29 of 49 items wrote nothing; the mean must be 30.0, not 30*20/49."""
    v, rep = arm_series(rows(zero_from=20), "A")
    assert rep["n_used"] == 20 and rep["n_dropped_zero_positions"] == 29
    assert sum(v) / len(v) == pytest.approx(30.0)


def test_bug8_an_arm_that_never_wrote_raises_rather_than_scoring_zero():
    """T_L2_all wrote nothing on every item and scored +0.00 with a CI of [0,0]."""
    with pytest.raises(ArmNotWritten):
        arm_series(rows(zero_from=0), "A")


def test_paired_contrast_drops_items_where_either_arm_is_absent():
    d, rep = paired(rows(zero_from=20), "A", "B")
    assert rep["n_used"] == 20 and rep["n_dropped_zero_positions"] == 29
    assert sum(d) / len(d) == pytest.approx(22.0)


def test_paired_refuses_position_count_mismatch():
    """a-b is a content contrast only if both arms perturbed equally."""
    with pytest.raises(ArmNotWritten):
        paired(rows(mismatch=True), "A", "B")


def test_banked_rows_without_position_fields_still_score_but_say_so():
    old = [{"dG_A": 30.0, "dG_B": 8.0} for _ in range(10)]
    v, rep = arm_series(old, "A")
    assert rep["position_field_present"] is False and rep["n_used"] == 10


# ── detector (2): exact-zero, for banked rows that predate the position fields ────────────────

def legacy(n=49, zero_from=None, val=30.0):
    """Rows as written BEFORE record_positions existed — no n_pos_* fields at all."""
    return [{"dG_A": val if (zero_from is None or i < zero_from) else 0.0,
             "dG_B": 8.0 if (zero_from is None or i < zero_from) else 0.0} for i in range(n)]


def test_bug8_on_legacy_rows_an_all_zero_arm_still_raises():
    """Re-scoring 379908 through detector (1) alone reproduced the bad verdict, because those rows
    carry no position field. Detector (2) is what makes the banked data safe."""
    with pytest.raises(ArmNotWritten):
        arm_series(legacy(zero_from=0), "A")


def test_bug9_on_legacy_rows_exact_zeros_are_dropped():
    v, rep = arm_series(legacy(zero_from=20), "A")
    assert rep["n_used"] == 20 and rep["n_dropped_exact_zero"] == 29
    assert sum(v) / len(v) == pytest.approx(30.0)


def test_self_arm_may_legitimately_score_exactly_zero():
    """The identity arm writes each position's own activation back and SHOULD score 0 -- 21 of 49
    items did so exactly. The exception is opt-in per arm so it cannot weaken the detector globally.
    """
    v, rep = arm_series(legacy(zero_from=0), "A", allow_exact_zero=True)
    assert rep["n_used"] == 49 and all(x == 0.0 for x in v)


def test_paired_on_legacy_rows_drops_items_where_either_side_is_an_exact_zero():
    d, rep = paired(legacy(zero_from=20), "A", "B")
    assert rep["n_used"] == 20 and sum(d) / len(d) == pytest.approx(22.0)
