"""N4 — locate where a wrong reasoning trace FIRST goes wrong.

Detectors, in resolution precedence:
  D0  slice set-diff        20/20 wrong slice cases; no LLM, no execution. Slice errors are not
                            value errors — every intermediate is right and only the final line
                            selection is wrong — so localization is a set difference.
  D2  execution oracle      PRIMARY. Compares claimed values against a real run of the stimulus.
  D1  arithmetic self-check cheap cross-check only (fires on 0/140 cases here, kept for other corpora)
  D3  judge fallback        (separate module: first_error_judge.py — it sees ground truth and must
                            never share a process with N7's answer-blind judge)

Resolver: D0 outright for slices; else D2 > D1 > D3, with an EARLINESS OVERRIDE — a candidate
with confidence ≥0.60 sitting >200 chars earlier wins, because "first error" is a minimum over
positions, and a later high-confidence detection does not refute an earlier one.

Run (CPU only, no GPU, no server):
    python -m src.first_error --out data/nla/n4/<date>/first_errors.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from claims import Claim, arith_disagreements, extract_claims          # noqa: E402
from exec_oracle import OracleResult, probe_calls, trace_run, values_agree  # noqa: E402

PROJ = _HERE.parent.parent
SEED = 20260724
SRC = PROJ / "data" / "nla" / "overnight" / "2026-08-04"


# ------------------------------------------------------------------ D0: slices
def d0_slice(case: dict) -> dict | None:
    truth = case.get("truth")
    ans = case.get("model_answer")
    if truth is None or ans is None:
        return None
    try:
        claimed = set(json.loads(str(ans).replace("'", '"'))) if str(ans).startswith("[") else set()
        want = set(truth if isinstance(truth, list) else json.loads(str(truth)))
    except Exception:
        return None
    missing, spurious = sorted(want - claimed), sorted(claimed - want)
    if not missing and not spurious:
        return None
    line = min(missing + spurious)
    reply = case["model_reply"]
    m = re.search(rf"(?:^\s*|[Ll]ine\s*)({line})\b(?![\d.])", reply, re.M)
    pos = m.start() if m else int(0.5 * len(reply))
    # the known slice failure mode: applies "directly affects" instead of transitive dependency
    criterion = "direct_only" if missing and not spurious else "mixed"
    return {"position_char": pos, "kind": "slice_selection", "claimed": str(sorted(claimed)),
            "truth": str(sorted(want)), "detector": "D0", "confidence": 1.0,
            "slice": {"missing": missing, "spurious": spurious, "criterion": criterion}}


# ------------------------------------------------------------------ D2: execution
def d2_execution(case: dict, claims: list[Claim]) -> tuple[dict | None, OracleResult, int]:
    code, lang = case.get("code", ""), case.get("language", "python")
    call_claims = [c for c in claims if c.family == "call"]
    var_claims = [c for c in claims if c.family in ("var", "state")]
    checked = 0

    # (i) call-form — exact and cheap
    if call_claims:
        wanted = sorted({(c.name, int(c.arg)) for c in call_claims if c.arg is not None})
        res = probe_calls(code, lang, wanted[:60])
        if res.status == "ok":
            for c in sorted(call_claims, key=lambda c: c.span[0]):
                truth = res.calls.get((c.name, int(c.arg)))
                if truth in (None, "ERR"):
                    continue
                checked += 1
                if not values_agree(c.value, truth):
                    return ({"position_char": c.span[0], "kind": "wrong_value",
                             "claimed": str(c.value), "truth": str(truth),
                             "detector": "D2-call", "confidence": 0.90,
                             "detail": f"{c.name}({c.arg})"}, res, checked)
        else:
            return None, res, checked

    # (ii)/(iii) variable + container — occurrence-index alignment against a traced run
    if var_claims and case.get("call"):
        # enriched.json drops `meta`, so derive the entry-point name from the call string
        # ("xOrY(91, 56, 129)" -> "xOrY") and add every function the claims mention.
        fns = {re.match(r"\s*([A-Za-z_$][\w$]*)\s*\(", case["call"]).group(1)} \
            if re.match(r"\s*([A-Za-z_$][\w$]*)\s*\(", case["call"]) else set()
        fns |= {c.name for c in call_claims}
        tr = trace_run(code, case["call"], lang, sorted(fns))
        if tr.status == "ok" and tr.events:
            by_name: dict[str, list] = {}
            for e in tr.events:
                by_name.setdefault(e.get("name", ""), []).append(e)
            seen: dict[str, int] = {}
            for c in sorted(var_claims, key=lambda c: c.span[0]):
                seq = by_name.get(c.name)
                if not seq:
                    continue
                k = seen.get(c.name, 0)
                if k >= len(seq):
                    continue
                seen[c.name] = k + 1
                truth = seq[k].get("value")
                checked += 1
                if not values_agree(c.value, truth):
                    return ({"position_char": c.span[0], "kind": "wrong_state",
                             "claimed": str(c.value)[:120], "truth": str(truth)[:120],
                             "detector": "D2-state", "confidence": 0.60,
                             "detail": f"{c.name} (occurrence {k+1})"}, tr, checked)
            return None, tr, checked
        return None, tr, checked
    return None, OracleResult("unsupported"), checked


# ------------------------------------------------------------------ D1
def d1_arith(claims: list[Claim]) -> dict | None:
    bad = arith_disagreements(claims)
    if not bad:
        return None
    c = bad[0]
    lhs, rhs = c.value
    return {"position_char": c.span[0], "kind": "arithmetic", "claimed": str(rhs),
            "truth": str(lhs), "detector": "D1", "confidence": 0.95, "detail": c.raw}


# ------------------------------------------------------------------ resolver
def resolve(cands: list[dict]) -> dict | None:
    cands = [c for c in cands if c]
    if not cands:
        return None
    order = {"D0": 0, "D2-call": 1, "D2-state": 2, "D1": 3, "D3": 4}
    best = min(cands, key=lambda c: (order.get(c["detector"], 9), c["position_char"]))
    # earliness override: a confident candidate >200 chars earlier wins
    for c in cands:
        if c is best:
            continue
        if c["position_char"] < best["position_char"] - 200 and c["confidence"] >= 0.60:
            best = c
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(SRC / "enriched.json"))
    ap.add_argument("--out", default=str(PROJ / "data/nla/n4" /
                                        datetime.now().strftime("%Y-%m-%d") / "first_errors.jsonl"))
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cases = json.load(open(args.src))
    wrong = [c for c in cases if c.get("correct") is False and not c.get("truncated")]
    if args.limit:
        wrong = wrong[: args.limit]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_loc = 0
    stats = {"D0": 0, "D2-call": 0, "D2-state": 0, "D1": 0, "none": 0}
    with open(out_path, "w") as fh:
        for case in wrong:
            reply = case["model_reply"]
            claims = extract_claims(reply)
            cands: list[dict] = []
            oracle = OracleResult("unsupported")
            checked = 0
            if case["kind"] == "slice_prediction":
                cands.append(d0_slice(case))
            else:
                hit, oracle, checked = d2_execution(case, claims)
                cands.append(hit)
                cands.append(d1_arith(claims))
            best = resolve(cands)
            if best:
                n_loc += 1
                stats[best["detector"]] = stats.get(best["detector"], 0) + 1
                best["u_rel"] = best["position_char"] / max(len(reply), 1)
            else:
                stats["none"] += 1
            fh.write(json.dumps({
                "task_key": case["task_key"], "tier": case["tier"], "dataset": case["dataset"],
                "language": case["language"], "kind": case["kind"],
                "model_answer": case["model_answer"], "truth": case["truth"],
                "first_error": best,
                "alternatives": [c for c in cands if c and c is not best],
                "claims": {"n_total": len(claims), "n_checked": checked,
                           "by_family": {f: sum(1 for c in claims if c.family == f)
                                         for f in ("call", "var", "state", "arith")}},
                "oracle": {"status": oracle.status, "n_events": len(oracle.events),
                           "stderr": oracle.stderr[:200]},
                "seed": SEED,
            }) + "\n")
            fh.flush()
    print(f"[N4] {len(wrong)} wrong cases · localized {n_loc} ({n_loc/len(wrong):.0%})")
    print(f"     by detector: {stats}")
    print(f"     wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
