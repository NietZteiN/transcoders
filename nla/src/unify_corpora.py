"""Unify every NLA reading taken in this programme into one table.

The readings were captured in three separate passes with three different row shapes, which has
kept them un-comparable:

    overnight   380 cases  5,090 reads   token classes (fn_orig, adversarial, cot, ...)
    dense        91 pairs  4,653 reads   a denser second pass over reasoning traces
    malware     183 pkgs   5,108 reads   two framing arms over quarantined samples

They are the same measurement — a layer-20 vector at one token position, verbalized, with a
round-trip faithfulness score — so they belong in one table with a `corpus` column. That is what
makes cross-corpus questions askable at all: whether faithfulness differs on hostile code, whether
the token-class ordering holds outside the obfuscation set, whether `act_norm` behaves the same
in both.

TWO THINGS THIS FIXES ON THE WAY IN:

1. **Reused reads are not double-counted.** The dense pass deliberately re-used 232 readings from
   the overnight pass (verified byte-identical at the time). They carry `reused: true` and are
   dropped here, so 4,653 dense rows contribute 4,421 new ones. Counting them twice would inflate
   every pooled mean by a subset that is, by construction, drawn from one end of the trace.

2. **Malware rows are redacted at this boundary**, not downstream. Derived data inherits
   quarantine status, so if the unified table shipped raw it would be a quarantined file and
   useless for ordinary analysis. Redacting here means the unified table carries no live
   indicator and can sit with the rest of the data.

`act_norm` is present for the malware pass (captured natively) and for the 5,090 overnight
positions (recovered by N13 stage 2), and null for the dense pass, which never stored it.

Env: any. CPU, seconds.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
PROJ = _HERE.parent.parent

from redact import build_denylist, redact, verify  # noqa: E402

# one vocabulary for "what kind of token was read", across three naming schemes
CLASS_MAP = {
    # overnight `cls`
    "fn_orig": "function_name", "orig": "identifier_original", "adversarial": "identifier_decoy",
    "l1_neutral": "identifier_gibberish", "dispatcher": "dispatcher_var", "target": "target_var",
    "cot": "reasoning", "answer": "answer_line",
    # dense `kind` and malware `locus`
    "sweep": "reasoning", "strip": "reasoning", "burst": "reasoning",
    "COT": "reasoning", "CODE": "code_token",
}


def norm_class(raw: str | None) -> str:
    return CLASS_MAP.get(raw or "", (raw or "unknown").lower())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(PROJ / "data/nla/unified"))
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    cases: list[dict] = []
    stats: Counter = Counter()

    # ---- act_norm recovered by N13 stage 2, keyed (case, position) -------------
    an: dict[tuple[str, int], float] = {}
    p = PROJ / "data/nla/n13/act_norm.jsonl"
    if p.exists():
        for line in open(p):
            r = json.loads(line)
            if "error" in r:
                continue
            for x in r["norms"]:
                an[(r["case"], x["position"])] = x["act_norm"]

    # ---- pass 1: overnight ----------------------------------------------------
    for r in json.load(open(PROJ / "data/nla/overnight/2026-08-04/enriched.json")):
        cid = r["task_key"]
        cases.append({"corpus": "obfuscated", "case_id": cid, "pass": "overnight",
                      "tier": r.get("tier"), "dataset": r.get("dataset"),
                      "language": r.get("language"), "snippet_id": r.get("snippet_id"),
                      "correct": r.get("correct"), "malicious": False,
                      "code_chars": len(r.get("code") or ""),
                      "reply_tokens": r.get("reply_tokens")})
        for x in r["reads"]:
            a = x.get("anchor") or {}
            rows.append({
                "corpus": "obfuscated", "pass": "overnight", "case_id": cid,
                "position": int(x["position"]), "read_class": norm_class(x.get("cls")),
                "in_reply": a.get("in_reply"), "token": (a.get("tok") or "")[:40],
                "read": x.get("read") or "", "rt_cos": x.get("rt_cos"),
                "act_norm": an.get((cid, int(x["position"]))),
                "u_rel": None, "malice_margin": None,
                "tier": r.get("tier"), "correct": r.get("correct"), "malicious": False,
            })
        stats["overnight"] += len(r["reads"])

    # ---- pass 2: dense (drop reads re-used from pass 1) ------------------------
    for line in open(PROJ / "data/nla/n5/2026-08-06/dense_reads.jsonl"):
        rec = json.loads(line)
        for side in ("control", "wrong"):
            s = rec.get(side)
            if not isinstance(s, dict):
                continue
            cid = s["task_key"]
            cases.append({"corpus": "obfuscated", "case_id": cid, "pass": "dense",
                          "tier": s.get("tier"), "dataset": s.get("dataset"),
                          "language": s.get("language"), "snippet_id": None,
                          "correct": s.get("correct"), "malicious": False,
                          "code_chars": None, "reply_tokens": s.get("reply_tokens")})
            for x in s.get("reads", []):
                if x.get("reused"):
                    stats["dense_reused_dropped"] += 1
                    continue
                rows.append({
                    "corpus": "obfuscated", "pass": "dense", "case_id": cid,
                    "position": int(x["position"]), "read_class": norm_class(x.get("kind")),
                    "in_reply": True, "token": (x.get("tok") or "")[:40],
                    "read": x.get("read") or "", "rt_cos": x.get("rt_cos"),
                    "act_norm": an.get((cid, int(x["position"]))),
                    "u_rel": x.get("u_rel"), "malice_margin": None,
                    "tier": s.get("tier"), "correct": s.get("correct"), "malicious": False,
                })
                stats["dense_new"] += 1

    # ---- pass 3: malware, redacted at the boundary -----------------------------
    deny = build_denylist({json.loads(l).get("package")
                           for l in open(PROJ / "data/quarantine/malware/corpus.jsonl")
                           if l.strip()})
    for arm, fn in (("malware_security", "reads.jsonl"), ("malware_neutral", "reads_neutral.jsonl")):
        for line in open(PROJ / "data/quarantine/malware" / fn):
            r = json.loads(line)
            if "error" in r:
                continue
            cid = r["item_id"]                      # content hash; package name never emitted
            cases.append({"corpus": "malware", "case_id": cid, "pass": arm,
                          "tier": None, "dataset": "pypi-malware", "language": "python",
                          "snippet_id": None, "correct": None, "malicious": True,
                          "capability": r.get("capability_sig"),
                          "code_chars": r.get("code_chars"),
                          "reply_tokens": r.get("reply_tokens")})
            for x in r.get("reads", []):
                rows.append({
                    "corpus": "malware", "pass": arm, "case_id": cid,
                    "position": int(x["position"]), "read_class": norm_class(x.get("locus")),
                    "in_reply": x.get("locus") == "COT",
                    "token": redact(x.get("token_text") or "", deny)[:40],
                    "read": redact(x.get("read") or "", deny),
                    "rt_cos": x.get("rt_cos"), "act_norm": x.get("act_norm"),
                    "u_rel": x.get("u_rel"), "malice_margin": x.get("mm"),
                    "tier": None, "correct": None, "malicious": True,
                    "capability": r.get("capability_sig"),
                })
            stats[arm] += len(r.get("reads", []))

    # ---- verification gate: no live indicator may enter the unified table ------
    residual: Counter = Counter()
    for r in rows:
        if r["corpus"] != "malware":
            continue
        for k in verify(r["read"]):
            residual[k] += 1
        for k in verify(r["token"]):
            residual[k] += 1
    if residual:
        print(f"[unify] ABORT: residual indicators {dict(residual)}", file=sys.stderr)
        return 2

    rp, cp = out / "readings.jsonl", out / "cases.jsonl"
    with open(rp, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    with open(cp, "w") as fh:
        for c in cases:
            fh.write(json.dumps(c) + "\n")

    man = {
        "dataset": "nla_unified_readings", "built_utc": None,
        "n_readings": len(rows), "n_cases": len(cases),
        "by_pass": dict(Counter(r["pass"] for r in rows)),
        "by_corpus": dict(Counter(r["corpus"] for r in rows)),
        "by_class": dict(Counter(r["read_class"] for r in rows)),
        "dense_reused_dropped": stats["dense_reused_dropped"],
        "with_act_norm": sum(1 for r in rows if r["act_norm"] is not None),
        "redaction": "malware rows redacted at unification; 0 residual indicators verified",
        "note": ("Reused dense readings are dropped so no reading is counted twice. act_norm is "
                 "present for the malware pass (native) and the overnight pass (recovered by N13 "
                 "stage 2); the dense pass never stored it."),
    }
    (out / "manifest.json").write_text(json.dumps(man, indent=2))
    print(json.dumps(man, indent=2))
    print(f"[unify] wrote {rp} and {cp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
