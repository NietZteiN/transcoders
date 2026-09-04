"""Verdict-table tests for Experiment W stage 1. CPU-only, no model, no GPU.

The point of these is that the decision table was frozen in the pre-registration, and the two
branches that matter most are the ones a careless implementation gets wrong:

  * a bare AV->AR round trip that moves G_sum makes every W1 number an artifact, so C1 must
    short-circuit to UNINFORMATIVE **before** H-W1 is read -- not after;
  * a missing behavioural veto must refuse a verdict rather than quietly grant one. The absence
    of exactly this clause is what produced the `S-LIVE` artifact on 2026-09-03.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from nla_writeback import score  # noqa: E402

BASE = {"acc": 0.63, "parse": 0.90}


def rows(dW: float, dC1: float, cos: float, n: int = 60, acc: float = 0.63,
         dC2: float = 0.1) -> list[dict]:
    return [{"snippet_id": f"s{i}", "n_positions": 8, "n_editable": 4, "n_tok": 400,
             "dG_W1_edit": dW, "dG_C1_roundtrip": dC1, "dG_C2_foreign": dC2,
             "dG_C3_ceiling": 1.0,
             "acc_W1_edit": acc, "parse_W1_edit": 0.9,
             "acc_C1_roundtrip": acc, "parse_C1_roundtrip": 0.9,
             "acc_C2_foreign": acc, "parse_C2_foreign": 0.9,
             "acc_C3_ceiling": acc, "parse_C3_ceiling": 0.9,
             "cos_dh_dedit": cos, "cos_dh_dedit_foreign": -0.1,
             "cos_dh_c1_dedit": -0.05} for i in range(n)]


@pytest.fixture()
def out():
    return Path(tempfile.mkdtemp()) / "stats.json"


@pytest.mark.parametrize("dW,dC1,cos,want", [
    (20.0, 0.5, +0.4, "W-STEERS"),
    (0.5, 0.5, +0.4, "W-READOUT-NOT-MECHANISM"),
    (0.5, 0.5, -0.4, "W-UNINFORMATIVE-NO-DELIVERY"),
    (20.0, 30.0, +0.4, "W-UNINFORMATIVE-ROUNDTRIP"),
])
def test_verdict_table_routes(dW, dC1, cos, want, out):
    assert score(rows(dW, dC1, cos), BASE, out)["verdict"] == want


def test_missing_veto_refuses_a_verdict(out):
    """No baseline => the veto cannot be evaluated => no verdict, however good dG_sum looks."""
    assert score(rows(20.0, 0.5, +0.4), None, out)["verdict"] == "W-VETO-MISSING"


def test_behavioural_veto_disqualifies_a_damaging_arm(out):
    """A big dG_sum bought by wrecking the model is not a rescue -- the S-LIVE failure mode."""
    st = score(rows(20.0, 0.5, +0.4, acc=0.10), BASE, out)
    assert st["per_arm"]["W1_edit"]["veto"] is True
    assert st["verdict"] != "W-STEERS"


def test_support_threshold_is_the_frozen_one(out):
    """Just under +12.11 nats must not clear support; comfortably over must."""
    assert score(rows(12.0, 0.5, +0.4), BASE, out)["per_arm"]["W1_edit"]["clears_support"] is False
    assert score(rows(20.0, 0.5, +0.4), BASE, out)["per_arm"]["W1_edit"]["clears_support"] is True


def test_foreign_edit_matching_w1_blocks_a_steering_claim(out):
    """If the wrong-content edit does as well, the effect is generic perturbation, not belief."""
    st = score(rows(20.0, 0.5, +0.4, dC2=25.0), BASE, out)
    assert st["verdict"] != "W-STEERS"
