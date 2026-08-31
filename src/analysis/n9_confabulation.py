"""N9 — how often do the readings invent a specific, and does faithfulness catch it?

EXPLORATORY. Not pre-registered, no FDR, no confirmatory claim. It exists because the project has
carried one qualitative statement since 2026-08-04 -- "the readings are reliable about themes and
confabulate specifics" -- that has never been measured. It turns out to be mechanically checkable.

Every case has a known ground-truth language (python or javascript, from the stimulus), and the
readings name a language unprompted in ~91% of cases. So "did the reading name the right language?"
is a per-reading correctness label that needs no judge, no annotation and no model -- a rare thing
in this project, and the reason this is worth running.

Two questions:
  1. HOW OFTEN is an invented specific wrong? (the confabulation rate)
  2. Does `rt_cos` -- round-trip faithfulness -- CATCH it? The page claims rt_cos measures
     completeness, not truth. If a confidently-reconstructed reading is just as likely to name the
     wrong language as a poorly-reconstructed one, that claim stops being an interpretive caveat and
     becomes a measured fact. This is the cleanest available test of it.

The verbalizer never sees the code -- only a 3,584-number activation -- so a named language is
always an inference from the residual stream, never a copy.

Env: transcoders-mi. No GPU.
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

PROJ = Path("/data/jvl210002/my_downloads/transcoders")
SEED = 20260724

# Word-boundary matches only. "Go" and "C" are excluded -- far too many false positives in prose
# ("go through the loop", "C = 3") to be worth their tiny signal.
LANG_PAT = {
    "python": r"\bPython\b",
    "javascript": r"\b(?:JavaScript|JS|Node\.?js|ECMAScript)\b",
    "java": r"\bJava\b(?!Script)",
    "csharp": r"(?:\bC#|\bCSharp\b)",
    "cpp": r"(?:\bC\+\+\b)",
    "typescript": r"\bTypeScript\b",
    "ruby": r"\bRuby\b",
    "rust": r"\bRust\b",
    "php": r"\bPHP\b",
    "swift": r"\bSwift\b",
    "kotlin": r"\bKotlin\b",
    "haskell": r"\bHaskell\b",
    "perl": r"\bPerl\b",
    "scala": r"\bScala\b",
}
# Libraries that pin a language, so naming one is an implicit language claim.
LIB_LANG = {
    "python": r"\b(?:numpy|pandas|itertools|collections\.|scipy|matplotlib|django|flask)\b",
    "javascript": r"\b(?:lodash|jQuery|React|Express|Array\.prototype)\b",
}


def named_languages(text: str) -> set[str]:
    return {k for k, p in LANG_PAT.items() if re.search(p, text, re.I if k != "java" else 0)}


def named_libs(text: str) -> set[str]:
    return {k for k, p in LIB_LANG.items() if re.search(p, text, re.I)}


def main() -> int:
    caps = {}
    for line in open(PROJ / "data/nla/overnight/2026-08-04/captures.jsonl"):
        r = json.loads(line)
        caps[r["task_key"]] = r
    dense = json.load(open(PROJ / "data/nla/n5/2026-08-06/dense_reads_anchored.json"))["reads"]

    rows = []
    for tk, c in caps.items():
        truth = (c.get("language") or "").lower()
        if truth not in ("python", "javascript"):
            continue
        seen = set()
        pool = [(rd, "banked") for rd in c.get("reads", [])]
        for rd in dense.get(tk, []):
            if rd["position"] not in {x.get("position") for x, _ in pool}:
                pool.append((rd, "dense"))
        for rd, src in pool:
            if rd.get("position") in seen:
                continue
            seen.add(rd.get("position"))
            langs = named_languages(rd["read"])
            libs = named_libs(rd["read"])
            claim = langs | libs
            where = rd.get("where", "")
            kind = ("cot" if where.startswith("cot@") else "answer" if where.startswith("ans@")
                    else "dispatcher" if where.startswith("disp:") else "code")
            rows.append({
                "case": tk, "truth": truth, "tier": c.get("tier") or "slice", "src": src,
                "kind": kind, "rt_cos": rd.get("rt_cos"), "u_rel": rd.get("u_rel"),
                "named": bool(claim),
                "correct_lang": (truth in claim) if claim else None,
                "wrong_only": bool(claim) and truth not in claim,
                "claims": sorted(claim),
                "n_langs": len(claim),
            })

    n = len(rows)
    named = [r for r in rows if r["named"]]
    wrong = [r for r in named if r["wrong_only"]]
    out: dict = {
        "n_readings": n,
        "named_a_language": len(named),
        "named_pct": round(100 * len(named) / n, 2),
        "of_those_wrong": len(wrong),
        "wrong_pct": round(100 * len(wrong) / max(len(named), 1), 2),
        "corpus_languages": dict(Counter(r["truth"] for r in rows)),
        "what_they_claim": dict(Counter(c for r in named for c in r["claims"]).most_common()),
    }

    # --- by the case's true language: the asymmetry is the finding ---
    by_truth = {}
    for t in ("python", "javascript"):
        g = [r for r in rows if r["truth"] == t]
        gn = [r for r in g if r["named"]]
        gw = [r for r in gn if r["wrong_only"]]
        by_truth[t] = {"n": len(g), "named": len(gn),
                       "named_pct": round(100 * len(gn) / max(len(g), 1), 2),
                       "wrong": len(gw),
                       "wrong_pct": round(100 * len(gw) / max(len(gn), 1), 2),
                       "top_claims": dict(Counter(c for r in gn for c in r["claims"]).most_common(4))}
    out["by_true_language"] = by_truth

    # --- does rt_cos catch it? quintiles of faithfulness vs the error rate ---
    fr = [r for r in named if r["rt_cos"] is not None]
    fr.sort(key=lambda r: r["rt_cos"])
    q = []
    for i in range(5):
        lo, hi = int(i * len(fr) / 5), int((i + 1) * len(fr) / 5)
        seg = fr[lo:hi]
        if not seg:
            continue
        w = sum(s["wrong_only"] for s in seg)
        q.append({"quintile": i + 1,
                  "rt_cos_lo": round(seg[0]["rt_cos"], 4), "rt_cos_hi": round(seg[-1]["rt_cos"], 4),
                  "n": len(seg), "wrong": w, "wrong_pct": round(100 * w / len(seg), 2)})
    out["by_faithfulness_quintile"] = q
    if len(q) >= 2:
        rt = np.array([r["rt_cos"] for r in fr])
        er = np.array([1.0 if r["wrong_only"] else 0.0 for r in fr])
        # point-biserial: does higher faithfulness predict naming the RIGHT language?
        out["corr_rt_vs_error"] = round(float(np.corrcoef(rt, er)[0, 1]), 4)
        # case-cluster bootstrap on the quintile-1 vs quintile-5 gap
        rng = np.random.default_rng(SEED)
        cases = sorted({r["case"] for r in fr})
        by_case = defaultdict(list)
        for r in fr:
            by_case[r["case"]].append(r)
        diffs = []
        for _ in range(2000):
            samp = [x for cse in rng.choice(cases, len(cases), True) for x in by_case[cse]]
            samp.sort(key=lambda r: r["rt_cos"])
            k = max(len(samp) // 5, 1)
            lo_e = np.mean([r["wrong_only"] for r in samp[:k]])
            hi_e = np.mean([r["wrong_only"] for r in samp[-k:]])
            diffs.append(hi_e - lo_e)
        out["q5_minus_q1_error_rate"] = {
            "mean": round(float(np.mean(diffs)), 4),
            "ci95": [round(float(np.percentile(diffs, 2.5)), 4),
                     round(float(np.percentile(diffs, 97.5)), 4)]}

    # --- by position in the trace ---
    # I predicted this would DECAY: the top confabulation examples all sat at u ~= 0.00-0.01, so the
    # natural story was "a prior that gets corrected as language-specific evidence accumulates".
    # It does the opposite. Kept because a failed prediction is worth as much as the by-kind split
    # that turned out to be the real explanation.
    dense_named = [r for r in rows if r["src"] == "dense" and r["named"]]
    pos = {}
    for t_lang in ("python", "javascript"):
        g = [r for r in dense_named if r["truth"] == t_lang]
        by_u = []
        for b in range(10):
            seg = [r for r in g if r.get("u_rel") is not None and int(r["u_rel"] * 10) == b]
            if seg:
                by_u.append({"decile": b, "n": len(seg),
                             "wrong_pct": round(100 * sum(s["wrong_only"] for s in seg) / len(seg), 2)})
        pos[t_lang] = by_u
    out["by_position"] = pos
    js = [r for r in dense_named if r["truth"] == "javascript" and r.get("u_rel") is not None]
    if len(js) > 50:
        u = np.array([r["u_rel"] for r in js])
        e = np.array([1.0 if r["wrong_only"] else 0.0 for r in js])
        out["position_corr_javascript"] = round(float(np.corrcoef(u, e)[0, 1]), 4)

    # --- by read kind and tier, for context ---
    for key in ("kind", "tier", "src"):
        d = {}
        for v in sorted({r[key] for r in rows}):
            g = [r for r in rows if r[key] == v and r["named"]]
            if not g:
                continue
            d[v] = {"named": len(g), "wrong": sum(r["wrong_only"] for r in g),
                    "wrong_pct": round(100 * sum(r["wrong_only"] for r in g) / len(g), 2)}
        out[f"by_{key}"] = d

    p = PROJ / "data/nla/n9/2026-08-07/confabulation.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({k: v for k, v in out.items() if k != "what_they_claim"}, indent=1, default=str))
    print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
