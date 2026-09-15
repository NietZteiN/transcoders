"""CPU tests for the frozen H-C4 dose-response table (no model, no GPU, no banked artifacts)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dose_score import (BANKED_INJECTION_SCALE, BANKED_MEAN_AV_TRAIN, STEP_NATS,  # noqa: E402
                        check_extract, load_c3, paired, verdict)


def _series(mean: float, n: int = 60) -> dict:
    """Per-item values with mean-zero jitter, so the bootstrap CI has non-zero width."""
    return {f"item/{i}": mean + 0.3 * ((i % 7) - 3) for i in range(n)}


def test_step_bar_is_the_documented_derivation():
    """0.02 of the banked L7 ceiling (+73.15). If either number moves the rule must be re-derived."""
    assert STEP_NATS == pytest.approx(0.02 * 73.15, abs=0.01)


def test_verdict_data_limited_when_the_top_step_clears():
    hi = paired(_series(10.0), _series(8.0))          # +2.0 >= 1.46, CI excludes 0
    lo = paired(_series(8.0), _series(4.0))
    assert verdict(hi, lo) == "DATA-LIMITED"


def test_verdict_data_saturated_when_only_the_lower_step_clears():
    hi = paired(_series(10.0), _series(9.8))          # +0.2 < 1.46
    lo = paired(_series(9.8), _series(6.0))           # +3.8 clears
    assert verdict(hi, lo) == "DATA-SATURATED"


def test_verdict_dose_insensitive_when_neither_step_clears():
    hi = paired(_series(10.0), _series(9.9))
    lo = paired(_series(9.9), _series(9.8))
    assert verdict(hi, lo) == "DOSE-INSENSITIVE"


def test_a_step_that_is_large_but_noisy_does_not_clear():
    """The rule needs BOTH the magnitude and a CI excluding 0 — magnitude alone is not enough."""
    import random
    rnd = random.Random(0)
    a = {f"item/{i}": 10.0 + rnd.gauss(0, 60) for i in range(60)}
    b = {f"item/{i}": 8.0 + rnd.gauss(0, 60) for i in range(60)}
    d = paired(a, b)
    assert d["ci95"][0] <= 0, "fixture should be noisy enough to not clear"
    assert verdict(d, paired(_series(1.0), _series(1.0))) == "DOSE-INSENSITIVE"


def test_a_DROP_is_never_read_as_an_effect():
    hi = paired(_series(4.0), _series(10.0))          # -6.0
    lo = paired(_series(10.0), _series(9.9))
    assert verdict(hi, lo) == "DOSE-INSENSITIVE"


def test_paired_refuses_to_silently_compare_different_item_sets():
    a = _series(10.0, n=60)
    b = _series(8.0, n=40)
    d = paired(a, b)
    assert d["n_common"] == 40 and d["n_a_only"] == 20 and d["n_b_only"] == 0, \
        "the mismatch must be reported in the result, not hidden"


def test_extract_preflight_accepts_the_banked_norms_and_rejects_drift(tmp_path):
    good = {"n_rows": 200000, "layers": {"7": {"injection_scale": BANKED_INJECTION_SCALE,
                                              "mean_av_train": BANKED_MEAN_AV_TRAIN}}}
    p = tmp_path / "norms.json"; p.write_text(json.dumps(good))
    assert check_extract(p)["passes"] is True
    for bad in ({"injection_scale": 5200, "mean_av_train": BANKED_MEAN_AV_TRAIN},
                {"injection_scale": BANKED_INJECTION_SCALE, "mean_av_train": BANKED_MEAN_AV_TRAIN + 3.0}):
        p.write_text(json.dumps({"n_rows": 200000, "layers": {"7": bad}}))
        assert check_extract(p)["passes"] is False, bad


def test_load_c3_reads_the_right_layer_and_fails_loudly_on_the_wrong_one(tmp_path):
    p = tmp_path / "gate_rows.jsonl"
    p.write_text("\n".join(json.dumps({"snippet_id": f"s{i}", "dG_S_c3_L7": 60.0 + i,
                                       "dG_S_c3_L22": 40.0}) for i in range(3)))
    got = load_c3(p, 7)
    assert got == {"s0": 60.0, "s1": 61.0, "s2": 62.0}
    with pytest.raises(SystemExit):
        load_c3(p, 33)
