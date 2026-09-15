"""H-R6 pre-GPU checks (CPU): the role tagger on a real renamed snippet, the reduced-rank ridge fit
recovering a planted map, the prototype fallback hierarchy, the test-set exclusion, and the runner's
vector-arm target arithmetic. Prereg: log/nla-harness/2026-09-14_better-vector-prereg.md."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
import ase_roles          # noqa: E402
import ase_vectors        # noqa: E402

JAVA_000 = """import java.util.*;
class Solution {
    public boolean socketconnection(List<Double> auths, double startupargs) {
        for (int i = 0; i < auths.size(); i++) {
            for (int j = i + 1; j < auths.size(); j++) {
                double fileName = Math.abs(auths.get(i) - auths.get(j));
                if (fileName < startupargs) return true;
            }
        }
        return false;
    }
    public int peers(int[] xs) {
        int attrs = 0;
        for (int x : xs) { attrs += x; }
        return attrs;
    }
}
"""


def test_role_tagger_java000():
    t = ase_roles.tag_source(JAVA_000, ["socketconnection", "auths", "startupargs", "fileName", "attrs", "xs", "peers"])
    assert (t["socketconnection"].kind, t["socketconnection"].type, t["socketconnection"].role) == ("method", "boolean", "method")
    assert (t["auths"].type, t["auths"].role) == ("List<Double>", "parameter")     # parameter outranks usage
    assert "called .size()" in t["auths"].usage and "called .get()" in t["auths"].usage
    assert (t["startupargs"].type, t["startupargs"].role) == ("double", "parameter")
    assert (t["fileName"].type, t["fileName"].role) == ("double", "local")
    assert (t["attrs"].type, t["attrs"].role) == ("int", "accumulator") and "returned" in t["attrs"].usage
    assert (t["xs"].type, t["xs"].role) == ("int[]", "parameter") and "iterated" in t["xs"].usage
    assert t["peers"].type == "int" and t["peers"].role == "method"
    line = t["auths"].prompt_line("- `{decoy}`: declared `{type}`, role: {role}")
    assert line == "- `auths`: declared `List<Double>`, role: parameter"
    assert "numbers" not in json.dumps({k: vars(v) for k, v in t.items()})   # true name never appears


def test_rrr_recovers_planted_map_and_rank0_is_mean():
    rng = np.random.default_rng(0)
    n, d, r = 400, 32, 3
    X = rng.normal(size=(n, d))
    A_true = rng.normal(size=(d, r)) @ rng.normal(size=(r, d)) * 0.3
    c = rng.normal(size=d)
    Y = X @ A_true + c + 0.01 * rng.normal(size=(n, d))
    mu = X.mean(0); cc = Y.mean(0)
    A = ase_vectors.rrr_fit(X - mu, Y - cc, lam=1.0, rank=r)
    assert np.linalg.matrix_rank(A, tol=1e-6) <= r
    pred = (X - mu) @ A + cc
    assert ase_vectors.cos_rows(pred, Y).mean() > 0.99
    A0 = ase_vectors.rrr_fit(X - mu, Y - cc, lam=1.0, rank=0)
    assert np.allclose(A0, 0.0)                       # rank 0 == the mean-difference (`erasure`) model


def _synthetic_pool(tmp_path, n_snip=12, d=16, seed=1):
    rng = np.random.default_rng(seed)
    A_true = rng.normal(size=(d, 2)) @ rng.normal(size=(2, d)) * 0.5
    spans = []
    roles = ["parameter", "local", "method"]
    for i in range(n_snip):
        sid = f"S_{i:03d}"
        for j in range(3):
            x = rng.normal(size=d)
            y = x + x @ A_true + 0.05 * rng.normal(size=d)
            spans.append({"snippet": sid, "span_idx": j, "name": f"n{j}", "decoy": f"d{j}",
                          "n_ren_tok": 1, "n_orig_tok": 1,
                          "h0_mean": torch.tensor(y, dtype=torch.float16),
                          "h1b_mean": torch.tensor(x, dtype=torch.float16),
                          "h0_tok": torch.zeros(1, d), "h1b_tok": torch.zeros(1, d),
                          "type": "int" if j < 2 else "boolean", "role": roles[j],
                          "kind": "method" if j == 2 else "variable", "usage": []})
    # one test span with a tag seen nowhere in the pool -> must fall back to `type`, then `kind`
    spans.append({**spans[0], "span_idx": 3, "type": "Map<K,V>", "role": "returned", "kind": "variable",
                  "h0_mean": spans[0]["h0_mean"], "h1b_mean": spans[0]["h1b_mean"]})
    pool = tmp_path / "pool.pt"
    torch.save({"layer": 7, "model": "toy", "config_sha": "x", "spans": spans, "excluded": {},
                "tags_missing": [], "n_snippets": n_snip}, pool)
    sub = tmp_path / "test_subset.jsonl"
    sub.write_text("\n".join(json.dumps({"snippet": f"S_{i:03d}"}) for i in range(3)) + "\n")
    cfg = {"seed": 20260724, "layer": 7,
           "paths": {"pool": str(pool), "test_subset": str(sub), "vectors": str(tmp_path / "vec.pt"),
                     "fit_report": str(tmp_path / "fit.json")},
           "split": {"min_pool_snippets": 5},
           "ridge_map": {"lambdas": [1e-1, 1.0, 10.0], "ranks": [0, 2, 8], "cv_folds": 3,
                         "apply_per_token": False, "gate_cos_margin": 0.05},
           "role_proto": {"tag_levels": ["type_role", "type", "kind"], "min_bucket": 2},
           "prompt_types": {"line": "- `{decoy}`: `{type}` {role}"}}
    cp = tmp_path / "cfg.yaml"
    import yaml
    cp.write_text(yaml.safe_dump(cfg))
    return cp, spans


def test_fit_excludes_test_snippets_and_resolves_prototypes(tmp_path, monkeypatch):
    cp, spans = _synthetic_pool(tmp_path)
    monkeypatch.setattr(sys, "argv", ["ase_vectors.py", "--config", str(cp)])
    assert ase_vectors.main() == 0
    rep = json.loads((tmp_path / "fit.json").read_text())
    assert rep["verdict"] == "OK"
    assert rep["n_test_snippets"] == 3 and rep["n_pool_snippets"] == 9
    assert rep["ridge_map"]["gate_pass"] and rep["ridge_map"]["best"]["rank"] >= 2   # planted rank-2 map found
    assert rep["role_proto"]["resolved_at"].get("type_role", 0) == 9        # 3 test snippets x 3 spans
    assert rep["role_proto"]["resolved_at"].get("kind", 0) == 1             # the Map<K,V>/returned span
    vec = torch.load(tmp_path / "vec.pt", weights_only=False)
    assert set(vec["arms"]["ridge_map"]) == {"S_000", "S_001", "S_002"}
    # exclusion: a test span's own clean state must not be its prototype (pool means only)
    own = spans[0]["h0_mean"].float()
    assert not torch.allclose(vec["arms"]["role_proto"]["S_000"][0], own, atol=1e-3)


def test_runner_vector_arm_targets():
    import ase_steer_run as R
    d = 8
    prep = {"S": {"spans": [{"decoy": "a", "ren_pos": [3, 4], "h0_tok": torch.zeros(1, d), "h0_mean": torch.zeros(d),
                             "h1b_tok": torch.zeros(2, d), "h1b_mean": torch.zeros(d)},
                            {"decoy": "b", "ren_pos": [9], "h0_tok": torch.zeros(1, d), "h0_mean": torch.zeros(d),
                             "h1b_tok": torch.zeros(1, d), "h1b_mean": torch.zeros(d)}]}}
    vectors = {"ridge_map": {"S": {0: torch.full((d,), 1.0), 1: torch.full((d,), 2.0)}}}
    T = R.residual_targets("ridge_map", "S", prep, np.random.default_rng(0), vectors)
    assert sorted(T) == [3, 4, 9] and float(T[3][0]) == 1.0 and float(T[9][0]) == 2.0
    with pytest.raises(RuntimeError):
        R.residual_targets("ridge_map", "S", prep, np.random.default_rng(0), {"ridge_map": {"S": {0: T[3]}}})
