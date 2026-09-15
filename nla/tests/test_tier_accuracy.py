"""CPU tests for the frozen H-A1/H-A2/H-A3 rules in nla_tier_accuracy.score (no model, no GPU)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nla_tier_accuracy import TIERS, score, verdicts  # noqa: E402

SEED = 20260724


def _rows(rates: dict, n: int = 60, noise: float = 0.0, seed: int = 0) -> list[dict]:
    """Synthetic per-item rows with per-tier pass rates around `rates[t]` (clipped to [0,1])."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        r = {"snippet_id": f"s{i}", "truth": "1"}
        for t in TIERS:
            p = float(np.clip(rates[t] + (rng.normal(0, noise) if noise else 0.0), 0, 1))
            r[f"{t}_rate"] = p
            r[f"{t}_greedy_correct"] = bool(p >= 0.5)
            r[f"{t}_greedy_parsed"] = True
        rows.append(r)
    return rows


def test_flat_ladder_is_floor_flat_not_additive():
    st = score(_rows({t: 0.5 for t in TIERS}), SEED)
    assert st["verdicts"] == {"H_A1": "ERASURE-FLOOR", "H_A2": "FLATTENING-FLAT",
                              "H_A3": "ROUTES-NOT-ADDITIVE"}


def test_expected_ordering_fires_all_three():
    rates = {"L0": 0.70, "L1": 0.65, "L1b": 0.50, "L2": 0.45, "L3": 0.25}
    st = score(_rows(rates, noise=0.05, seed=1), SEED)
    assert st["verdicts"] == {"H_A1": "ERASURE-HEADROOM", "H_A2": "FLATTENING-PENALTY",
                              "H_A3": "ROUTES-COMPOUND"}
    assert st["paired_rate"]["L1_minus_L1b"]["mean"] == pytest.approx(0.15, abs=0.03)


def test_compound_needs_l3_below_l2_not_just_below_l0():
    # L3 == L2 < L0: flattening penalty fires, compounding does not.
    rates = {"L0": 0.70, "L1": 0.70, "L1b": 0.70, "L2": 0.40, "L3": 0.40}
    st = score(_rows(rates, noise=0.05, seed=2), SEED)
    assert st["verdicts"]["H_A2"] == "FLATTENING-PENALTY"
    assert st["verdicts"]["H_A3"] == "ROUTES-NOT-ADDITIVE"
    assert st["verdicts"]["H_A1"] == "ERASURE-FLOOR"


def test_rules_need_ci_not_just_sign():
    # tiny mean difference swamped by noise: sign positive, CI spans 0 -> FLOOR
    rates = {"L0": 0.5, "L1": 0.51, "L1b": 0.50, "L2": 0.5, "L3": 0.5}
    st = score(_rows(rates, noise=0.3, seed=3), SEED)
    assert st["paired_rate"]["L1_minus_L1b"]["ci95"][0] <= 0
    assert st["verdicts"]["H_A1"] == "ERASURE-FLOOR"


def test_greedy_block_and_banked_check_present():
    st = score(_rows({t: 0.6 for t in TIERS}), SEED)
    assert set(st["greedy"]) == set(TIERS)
    assert st["greedy"]["L0"]["acc"] == 1.0       # 0.6 >= 0.5 everywhere
    assert st["banked_check"]["L0"]["within_churn"] is False   # 1.0 vs 0.567 is outside 9/60
    assert "L1b" not in st["vs_L1b_greedy"] and set(st["vs_L1b_greedy"]) == set(TIERS) - {"L1b"}


def test_verdicts_missing_when_a_tier_has_no_rates():
    rows = _rows({t: 0.5 for t in TIERS})
    for r in rows:
        del r["L2_rate"]
    st = score(rows, SEED)
    assert "verdicts" not in st
