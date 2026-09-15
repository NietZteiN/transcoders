"""CPU tests for the Phase-B gate's frozen rules and liveness checks (no model)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import nla_ml_gate as g  # noqa: E402

LIVE = [8, 22, 24]


def _cfg(tmp_path):
    cfg = g.load_cfg(g.CFG_PATH)
    cfg["n_boot"] = 400
    return cfg


def _rows(n_items, means: dict, self_val=0.0, seed=0):
    """Synthetic per-item rows. `means` maps arm-kind -> (single mean, multi mean)."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_items):
        row = {"snippet_id": f"s{i}", "n_tok": 100, "logp_U": -100.0, "logp_L0prompt": -50.0,
               "G_prompt_swap": 50.0 + rng.normal(0, 2), "n_spans": 4, "n_span_pos": 8}
        for K in LIVE:
            row[f"n_editable_L{K}"] = 3
            for kind in g.SINGLE_KINDS:
                arm = g.single_arm(kind, K)
                v = self_val if kind == "self" else means[kind][0] + rng.normal(0, 1)
                row[f"dG_{arm}"] = v; row[f"n_pos_{arm}"] = 8
        for name in ("primary", "depth", "all"):
            for kind in g.MULTI_KINDS + ("self",):
                arm = g.multi_arm(kind, name)
                v = self_val if kind == "self" else means[kind][1] + rng.normal(0, 1)
                row[f"dG_{arm}"] = v; row[f"n_pos_{arm}"] = 24
        rows.append(row)
    return rows


META = {"live_layers": LIVE, "dead_layers": [k for k in range(34) if k not in LIVE],
        "sets": {"primary": LIVE, "depth": LIVE, "all": LIVE}, "ignore_liveness": False}


def test_edit_live_verdict(tmp_path):
    cfg = _cfg(tmp_path)
    means = {"c3": (20, 30), "swap": (22, 32), "edit": (10, 25), "foreign": (5, 8),
             "rt": (3, 4), "random": (2, 3)}
    st = g.score(_rows(40, means), cfg, META)
    assert st["verdict"] == "M-EDIT-LIVE"
    p = st["gate"]["primary"]
    assert p["gate0"]["passes"] and p["H_M1"]["passes"] and p["H_M2"]["passes"] and p["specificity"]["passes"]
    assert st["identity_passes"] and st["reportable"]
    assert st["H_M3"]["verdict"] == "CONFIRM"          # 20/22 = 0.91 at every layer


def test_generic_verdict_and_hm3_refute(tmp_path):
    cfg = _cfg(tmp_path)
    means = {"c3": (10, 30), "swap": (22, 32), "edit": (10, 10), "foreign": (9, 10),
             "rt": (3, 4), "random": (2, 10)}
    st = g.score(_rows(40, means), cfg, META)
    assert st["verdict"] == "M-GENERIC"                 # no gain over single layer, no gain over random
    assert st["H_M3"]["verdict"] == "REFUTE"            # 10/22 = 0.45 median


def test_gain_but_generic(tmp_path):
    cfg = _cfg(tmp_path)
    means = {"c3": (20, 30), "swap": (22, 32), "edit": (10, 20), "foreign": (5, 8),
             "rt": (3, 4), "random": (2, 19)}
    st = g.score(_rows(40, means), cfg, META)
    assert st["verdict"] == "M-GAIN-BUT-GENERIC"


def test_gate0_fail(tmp_path):
    cfg = _cfg(tmp_path)
    means = {"c3": (2, 3), "swap": (22, 32), "edit": (10, 25), "foreign": (5, 8),
             "rt": (3, 4), "random": (2, 3)}
    st = g.score(_rows(40, means), cfg, META)
    assert st["verdict"] == "M-GATE0-FAIL"


def test_self_identity_failure_is_not_reportable(tmp_path):
    cfg = _cfg(tmp_path)
    means = {"c3": (20, 30), "swap": (22, 32), "edit": (10, 25), "foreign": (5, 8),
             "rt": (3, 4), "random": (2, 3)}
    st = g.score(_rows(40, means, self_val=2.5), cfg, META)
    assert not st["identity_passes"] and not st["reportable"]
    st0 = g.score(_rows(40, means, self_val=0.0), cfg, META)   # exact zeros are legitimate for SELF
    assert st0["identity_passes"]


def test_unwritten_arm_is_refused(tmp_path):
    cfg = _cfg(tmp_path)
    means = {"c3": (20, 30), "swap": (22, 32), "edit": (10, 25), "foreign": (5, 8),
             "rt": (3, 4), "random": (2, 3)}
    rows = _rows(10, means)
    for r in rows:
        r["n_pos_M_edit@primary"] = 0
    with pytest.raises(g.ArmNotWritten):
        g.score(rows, cfg, META)


def test_liveness_reads_trainer_outputs(tmp_path):
    cfg = _cfg(tmp_path)
    ld = tmp_path / "L5"; (ld / "av").mkdir(parents=True); (ld / "ar").mkdir()
    (ld / "av/eval.json").write_text(json.dumps({"holdout_gap_permuted_minus_real": 0.3}))
    (ld / "ar/eval.json").write_text(json.dumps({"holdout_fve": 0.4, "holdout_fve_shuffled": -0.1,
                                                 "holdout_fve_train_mean_baseline": 0.05}))
    (ld / "check.json").write_text(json.dumps({"n_reads": 96, "n_no_tags": 2, "cjk_rate": 0.0,
                                               "cos_cycle_mean": 0.5, "cos_other_mean": 0.1}))
    assert g.liveness(tmp_path, 5, cfg)["live"]
    assert not g.liveness(tmp_path, 6, cfg)["live"]                       # missing -> dead
    (ld / "ar/eval.json").write_text(json.dumps({"holdout_fve": 0.15, "holdout_fve_shuffled": -0.1,
                                                 "holdout_fve_train_mean_baseline": 0.05}))
    rep = g.liveness(tmp_path, 5, cfg)
    assert not rep["live"] and not rep["checks"]["b_ar_fve"]["pass"]
