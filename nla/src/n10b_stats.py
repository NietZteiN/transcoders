"""N10b — can NLA reads discriminate malicious code from matched benign code?

THE QUESTION N10 COULD NOT ASK. N10's corpus was entirely malicious, so its foreign-read null
was a *within-malware* null: it tested whether a read identifies THIS sample's capability among
other malware, and that failed (own 0.874 vs foreign null 0.903). It could never test the prior
question — whether the residual stream represents malicious code differently from benign code at
all. That needs matched controls, which is what this scores.

**The primary statistic is item-level AUC in the NEUTRAL arm.** Two reasons it must be neutral:
N10 measured generic threat vocabulary at 2.5x under a security-framed preamble on identical
activations, so a security-framed AUC is partly measuring the prompt; and the neutral arm is the
deployable question anyway — does a model reading code *normally* represent malice.

**The hard-negative stratum is the real test.** Benign controls were matched on length and
surface-obfuscation bucket, and 77.7% of them import subprocess/base64/requests, read the
environment, or run install hooks. If the overall AUC is high but collapses toward 0.5 on hard
negatives alone, the reads are detecting an API surface, not malice — and that distinction is the
whole point of building the matched set.

Scores are per item: the fraction of its reads that name a specific malicious capability, and
separately the mean Malice Margin. Both are continuous, so AUC is the natural summary and no
threshold has to be invented.

Env `transcoders-mi` (scipy for the CI). CPU, seconds.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics as st
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import sys

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
PROJ = _HERE.parent.parent
SEED = 20260724

from malware_read_stats import MALICE_GENERIC, MALICE_SPECIFIC  # noqa: E402


def auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Rank-based AUC (Mann-Whitney), ties averaged. 0.5 = chance."""
    pairs = sorted(zip(scores, labels))
    ranks, i = {}, 0
    vals = [p[0] for p in pairs]
    while i < len(vals):
        j = i
        while j + 1 < len(vals) and vals[j + 1] == vals[i]:
            j += 1
        r = (i + j + 2) / 2
        for k in range(i, j + 1):
            ranks[k] = r
        i = j + 1
    pos = [ranks[k] for k, (_, l) in enumerate(pairs) if l == 1]
    n1, n0 = len(pos), len(pairs) - len(pos)
    if n1 == 0 or n0 == 0:
        return float("nan")
    return (sum(pos) - n1 * (n1 + 1) / 2) / (n1 * n0)


def auc_ci(labels: Sequence[int], scores: Sequence[float], n_boot: int = 2000):
    rng = random.Random(SEED)
    n = len(labels)
    out = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        a = auc([labels[i] for i in idx], [scores[i] for i in idx])
        if a == a:
            out.append(a)
    out.sort()
    return ([round(out[int(0.025 * len(out))], 4), round(out[int(0.975 * len(out))], 4)]
            if out else [None, None])


def item_scores(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        if "error" in r:
            continue
        reads = r.get("reads") or []
        if not reads:
            continue
        spec = [1 if MALICE_SPECIFIC.search(x["read"]) else 0 for x in reads]
        gen = [1 if MALICE_GENERIC.search(x["read"]) else 0 for x in reads]
        out.append({
            "item_id": r["item_id"],
            "label": 1 if r.get("label", "malicious") == "malicious" else 0,
            "specific_frac": st.mean(spec), "generic_frac": st.mean(gen),
            "any_specific": int(any(spec)),
            "mm_mean": st.mean(x["mm"] for x in reads),
            "code_chars": r.get("code_chars"),
            "hard_negative": r.get("hard_negative"),
            "n_reads": len(reads),
        })
    return out


def block(items: list[dict], name: str) -> dict[str, Any]:
    lab = [i["label"] for i in items]
    if not lab or len(set(lab)) < 2:
        return {"arm": name, "n": len(items), "note": "needs both classes"}
    res: dict[str, Any] = {"arm": name, "n": len(items),
                           "n_malicious": sum(lab), "n_benign": len(lab) - sum(lab)}
    for key in ("specific_frac", "generic_frac", "mm_mean"):
        sc = [i[key] for i in items]
        res[key] = {
            "auc": round(auc(lab, sc), 4),
            "auc_ci95": auc_ci(lab, sc),
            "mean_malicious": round(st.mean(s for s, l in zip(sc, lab) if l == 1), 4),
            "mean_benign": round(st.mean(s for s, l in zip(sc, lab) if l == 0), 4),
        }
    res["any_specific_rate"] = {
        "malicious": round(st.mean(i["any_specific"] for i in items if i["label"] == 1), 4),
        "benign": round(st.mean(i["any_specific"] for i in items if i["label"] == 0), 4),
    }
    # Confound check: matching should leave length uninformative. If length alone separates the
    # classes, any read-based AUC inherits that separation.
    lens = [i["code_chars"] or 0 for i in items]
    res["length_auc_confound_check"] = round(auc(lab, lens), 4)
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mal-security", default=str(PROJ / "data/quarantine/malware/reads.jsonl"))
    ap.add_argument("--mal-neutral",
                    default=str(PROJ / "data/quarantine/malware/reads_neutral.jsonl"))
    ap.add_argument("--ben-security",
                    default=str(PROJ / "data/nla/n10b/benign_reads_security.jsonl"))
    ap.add_argument("--ben-neutral",
                    default=str(PROJ / "data/nla/n10b/benign_reads_neutral.jsonl"))
    ap.add_argument("--benign-corpus", default=str(PROJ / "data/nla/n10b/benign_corpus.jsonl"))
    ap.add_argument("--out", default=str(PROJ / "data/nla/n10b/n10b_stats.json"))
    args = ap.parse_args()

    hard = {}
    for l in open(args.benign_corpus):
        r = json.loads(l)
        hard[r["item_id"]] = r.get("hard_negative", False)

    def load(p: str, label: str) -> list[dict]:
        f = Path(p)
        if not f.exists():
            return []
        rows = [json.loads(l) for l in open(f) if l.strip()]
        for r in rows:
            r.setdefault("label", label)
            if label == "benign":
                r["hard_negative"] = hard.get(r.get("item_id"), False)
        return rows

    rep: dict[str, Any] = {"experiment": "n10b_discrimination", "seed": SEED}
    for arm, mp, bp in (("neutral", args.mal_neutral, args.ben_neutral),
                        ("security", args.mal_security, args.ben_security)):
        items = item_scores(load(mp, "malicious") + load(bp, "benign"))
        rep[arm] = block(items, arm)
        # The stratum that separates "detects malice" from "detects an API surface".
        hn = [i for i in items if i["label"] == 1 or i["hard_negative"]]
        rep[arm]["hard_negative_stratum"] = block(hn, f"{arm}/hard-negatives-only")

    rep["primary"] = {
        "statistic": "item-level AUC of specific-capability read fraction, NEUTRAL arm",
        "auc": rep.get("neutral", {}).get("specific_frac", {}).get("auc"),
        "auc_ci95": rep.get("neutral", {}).get("specific_frac", {}).get("auc_ci95"),
        "auc_hard_negatives_only":
            rep.get("neutral", {}).get("hard_negative_stratum", {})
               .get("specific_frac", {}).get("auc"),
        "note": ("Neutral arm is primary: N10 measured 2.5x more generic threat vocabulary under "
                 "a security-framed preamble on IDENTICAL activations, so a security-framed AUC "
                 "is partly a measure of the prompt. A high overall AUC that collapses on the "
                 "hard-negative stratum means the reads track API surface, not malice."),
    }
    rep["finished_utc"] = datetime.now(timezone.utc).isoformat()
    Path(args.out).write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
