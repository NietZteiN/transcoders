"""The B5 validation gate: does the NLA-patched runner still reproduce the banked numbers?

With `--nla-alpha 0` the residual write hook is a verified no-op (unit-tested byte-identity,
re-verified in the codesteer environment), so a run with attention steering ON and the belief
channel at zero MUST land on the accuracy the unpatched runner already produced for
`obf_steer:adversarial_rename`. If it does not, the patched runner is not the runner that
produced those numbers, and every composed result from it would measure the integration bug
rather than the levers.

This is a HARD STOP by design. The pre-registration says B5 is dropped rather than debugged
into existence, so this script exits non-zero on a miss and the autopilot runs nothing further.

Scoring re-derives correctness the same way `pipeline.analysis.aggregate` does — from each
case pack's `expected_bool` against the run's `predicted_labels.json` — rather than trusting a
summary written by the run under test, which would let a broken run vouch for itself.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

REPL = Path("/data/jvl210002/my_downloads/allocation_replication")
DEFAULT_GRID = REPL / "results/tables/grid.csv"


def pass_at_1_from_tree(result_root: Path, casepack_root: Path, run_tag: str,
                        dataset: str) -> tuple[float, int]:
    """P@1 over every (snippet, case) reachable under `run_tag`, first run only.

    Layout, confirmed against a completed grid rather than assumed:
      predictions  <result_root>/<model>/<snippet>/<technique>/<run_tag>/run_0001/predicted_labels.json
      case packs   <casepack_root>/<dataset>/<snippet>.json
    The two trees are siblings, not nested, which an earlier version of this function got
    wrong -- it searched for `case_pack.json` under the prediction path and would have found
    nothing, scored 0 cases, and failed the gate for a reason that had nothing to do with the
    integration under test.
    """
    covered: dict[tuple[str, str], bool] = {}
    for pl in result_root.rglob(f"*{run_tag}*/run_0001/predicted_labels.json"):
        # .../<snippet>/<technique>/<run_tag>/run_0001/predicted_labels.json
        snippet = pl.parents[3].name
        pack = casepack_root / dataset / f"{snippet}.json"
        if not pack.is_file():
            continue
        try:
            pred = json.loads(pl.read_text())
            cases = json.loads(pack.read_text()).get("cases", [])
        except Exception:
            continue
        for c in cases:
            cid, want = c.get("case_id"), c.get("expected_bool")
            if cid is None or want is None:
                continue
            got = str(pred.get(cid, "")).strip().upper()
            covered[(snippet, cid)] = (got == ("T" if bool(want) else "F"))
    n = len(covered)
    return (100.0 * sum(covered.values()) / n if n else float("nan")), n


def banked_reference(grid: Path, technique: str, arm: str) -> tuple[float, int]:
    """The already-published cell, recomputed from grid.csv rather than quoted from a doc."""
    seen: dict[tuple[str, str], bool] = {}
    with open(grid) as f:
        for r in csv.DictReader(f):
            if "Qwen2.5-Coder-7B" not in r["model"] or r["dataset"] != "humaneval":
                continue
            if r["arm"] != arm or r["technique"] != technique:
                continue
            if int(r["run_idx"]) != 1:
                continue
            seen[(r["snippet"], r["case_id"])] = (r["correct"] == "True")
    n = len(seen)
    return (100.0 * sum(seen.values()) / n if n else float("nan")), n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tol", type=float, default=0.02,
                    help="allowed |gate - banked| as a FRACTION of accuracy (0.02 = 2 points)")
    ap.add_argument("--run-tag", default="b5_gate_alpha0")
    ap.add_argument("--technique", default="adversarial_rename")
    ap.add_argument("--grid", default=str(DEFAULT_GRID))
    ap.add_argument("--dataset", default="humaneval")
    # NOTE the `artifacts/` segment. paths.DEFAULT_ARTIFACT_ROOT is "artifacts", and
    # resolve_artifact_root() only bypasses it when EYETRACKING_DATA_ROOT is set. The matrix
    # runner sets that per unit; a DIRECT invocation (which is how the gate and the six cells
    # run) does not, so its output lands under artifact/artifacts/. An earlier version of this
    # file pointed one level too high and would have scored 0 cases and failed the gate for a
    # reason unrelated to the integration. The known-answer test missed it because it was run
    # against a matrix-runner unit, whose layout differs precisely in this variable.
    ap.add_argument("--result-root",
                    default=str(REPL / "artifact/artifacts/obfuscation/result"))
    ap.add_argument("--casepack-root",
                    default=str(REPL / "artifact/artifacts/logs/eval_casepacks"))
    a = ap.parse_args()

    got, n_got = pass_at_1_from_tree(
        Path(a.result_root), Path(a.casepack_root), a.run_tag, a.dataset)
    ref, n_ref = banked_reference(Path(a.grid), a.technique, "obf_steer")
    delta = abs(got - ref) / 100.0 if (got == got and ref == ref) else float("nan")
    ok = bool(delta == delta and delta <= a.tol and n_got > 0 and n_ref > 0)

    verdict = {
        "pass": ok,
        "gate_p_at_1": round(got, 3), "gate_cases": n_got,
        "banked_p_at_1": round(ref, 3), "banked_cases": n_ref,
        "abs_delta_fraction": round(delta, 5) if delta == delta else None,
        "tolerance": a.tol,
        "rule": ("alpha=0 makes the belief channel a verified no-op, so the patched runner with "
                 "attention steering ON must reproduce the banked obf_steer number. A miss means "
                 "the integration changed the measurement."),
    }
    Path(a.out).write_text(json.dumps(verdict, indent=1))
    print(json.dumps(verdict, indent=1))
    if not ok:
        print("\nGATE FAILED — B5 stops here by pre-registration.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
