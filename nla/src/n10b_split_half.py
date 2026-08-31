"""N10b — split-half reliability: can the malice score rank ONE sample, or only separate crowds?

WHY THIS IS URGENT. N10b is the single positive result in the programme: neutral-arm item-level
AUC 0.764 for `specific_frac` (0.648 once the generic term `payload` is removed, and 0.856 for
`mm_mean`). An AUC is a statement about a *population* — the probability that a random malicious
item outranks a random benign one. It says nothing about whether the score attached to any
particular file means anything, and "this file scores 0.14, so it is probably malware" is the
claim a reader will actually take away.

Split-half reliability answers exactly that. Each item carries 14 reads. Split them into two
disjoint halves, score each half independently, and correlate the two scores across items. If the
halves disagree, the per-item score is noise that happens to have a population-level mean shift,
and only the crowd-level claim is licensed.

Spearman-Brown corrects the half-length correlation up to the full 14-read instrument:
r_full = 2*r_half / (1 + r_half).

PRE-STATED READING, so it is not chosen after seeing the number:
  r_full >= 0.70  the score is usable on a single sample; per-item claims are licensed.
  0.40-0.70       population claims only; any per-item statement carries the reliability.
  < 0.40          the item-level framing is not supported at all — report N10b as a population
                  separation and nothing more.
The 0.70 line is the conventional one for a research instrument and is arbitrary; it is written
here before the run for that reason.

ALSO REPORTED: AUC recomputed from each half alone. If the two halves' AUCs are far apart, the
headline AUC is itself unstable under which reads happened to be taken.

No GPU. Seconds.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics as st
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
PROJ = _HERE.parent.parent
SEED = 20260724
N_REPEATS = 200
RELIABLE, PARTIAL = 0.70, 0.40

from malware_read_stats import MALICE_GENERIC, MALICE_SPECIFIC  # noqa: E402
from n10b_stats import auc  # noqa: E402  (the same rank AUC the headline used)


def pearson(x: Sequence[float], y: Sequence[float]) -> float:
    n = len(x)
    if n < 3:
        return float("nan")
    mx, my = st.mean(x), st.mean(y)
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    return sxy / ((sxx * syy) ** 0.5) if sxx > 0 and syy > 0 else float("nan")


def rankify(v: Sequence[float]) -> list[float]:
    """Average ranks, so ties (very common here — most items score 0) are handled correctly."""
    order = sorted(range(len(v)), key=lambda i: v[i])
    out = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        r = (i + j + 2) / 2
        for k in range(i, j + 1):
            out[order[k]] = r
        i = j + 1
    return out


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    return pearson(rankify(x), rankify(y))


# The two regexes are compiled but still cost ~2 us per read, and a naive implementation
# re-matches every read on all 200 repeats x 3 metrics (~1.5 M matches, minutes). Each read's
# flags are fixed, so they are computed ONCE at load time and every split is then a mean over
# precomputed floats. Same numbers, ~200x faster.
def read_features(read: dict) -> tuple[float, float, float]:
    return (1.0 if MALICE_SPECIFIC.search(read["read"]) else 0.0,
            1.0 if MALICE_GENERIC.search(read["read"]) else 0.0,
            float(read["mm"]))


METRIC_IDX = {"specific_frac": 0, "generic_frac": 1, "mm_mean": 2}


def load(path: Path, label: int, hard: dict[str, bool] | None = None) -> list[dict]:
    out = []
    for l in open(path):
        if not l.strip():
            continue
        r = json.loads(l)
        if "error" in r:
            continue
        reads = r.get("reads") or []
        if len(reads) < 2:                      # a single read cannot be split
            continue
        out.append({"item_id": r["item_id"], "label": label,
                    "feats": [read_features(x) for x in reads],
                    # `hard_negative` is a property of the matched benign CORPUS, not of the
                    # reads file, so it is joined in by item_id. Reading it off the reads row
                    # silently yields None for every benign item and collapses the hard-negative
                    # stratum to one class — which is exactly what the first run did.
                    "hard_negative": (hard or {}).get(r["item_id"],
                                                      r.get("hard_negative"))})
    return out


def split_half(items: list[dict], metric: str, rng: random.Random,
               mode: str = "random") -> tuple[float, float, float, float]:
    """One split. Returns (pearson, spearman, auc_A, auc_B) across items."""
    k = METRIC_IDX[metric]
    a_vals, b_vals, labels = [], [], []
    for it in items:
        feats = it["feats"]
        if mode == "random":
            idx = list(range(len(feats)))
            rng.shuffle(idx)
            half = len(idx) // 2
            A, B = idx[:half], idx[half:2 * half]
        else:                                    # odd/even by position in the read sequence
            A, B = list(range(0, len(feats), 2)), list(range(1, len(feats), 2))
            n = min(len(A), len(B))
            A, B = A[:n], B[:n]
        if not A or not B:
            continue
        a_vals.append(st.mean(feats[i][k] for i in A))
        b_vals.append(st.mean(feats[i][k] for i in B))
        labels.append(it["label"])
    return (pearson(a_vals, b_vals), spearman(a_vals, b_vals),
            auc(labels, a_vals), auc(labels, b_vals))


def spearman_brown(r: float) -> float:
    return 2 * r / (1 + r) if r == r and r > -1 else float("nan")


def verdict(r_full: float) -> str:
    if r_full != r_full:
        return "NEEDS_DATA"
    if r_full >= RELIABLE:
        return "PER-ITEM LICENSED"
    return "POPULATION ONLY" if r_full >= PARTIAL else "ITEM-LEVEL FRAMING UNSUPPORTED"


def analyse(items: list[dict], name: str) -> dict[str, Any]:
    res: dict[str, Any] = {"stratum": name, "n_items": len(items),
                           "n_malicious": sum(i["label"] for i in items),
                           "reads_per_item_median": st.median(len(i["feats"]) for i in items)}
    for metric in ("specific_frac", "generic_frac", "mm_mean"):
        rng = random.Random(SEED)
        ps, ss, aa, ab = [], [], [], []
        for _ in range(N_REPEATS):
            p, s, x, y = split_half(items, metric, rng, "random")
            for acc, v in ((ps, p), (ss, s), (aa, x), (ab, y)):
                if v == v:
                    acc.append(v)
        r_half = st.mean(ps) if ps else float("nan")
        rho_half = st.mean(ss) if ss else float("nan")
        _, s_oe, _, _ = split_half(items, metric, random.Random(SEED), "oddeven")
        aucs = aa + ab
        res[metric] = {
            "pearson_half": round(r_half, 4),
            "spearman_half": round(rho_half, 4),
            "spearman_brown_full": round(spearman_brown(r_half), 4),
            "spearman_brown_full_rank": round(spearman_brown(rho_half), 4),
            "oddeven_spearman_half": round(s_oe, 4) if s_oe == s_oe else None,
            "half_auc_mean": round(st.mean(aucs), 4) if aucs else None,
            "half_auc_sd": round(st.pstdev(aucs), 4) if len(aucs) > 1 else None,
            "half_auc_range": [round(min(aucs), 4), round(max(aucs), 4)] if aucs else None,
            "verdict": verdict(spearman_brown(r_half)),
        }
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mal-neutral",
                    default=str(PROJ / "data/quarantine/malware/reads_neutral.jsonl"))
    ap.add_argument("--ben-neutral",
                    default=str(PROJ / "data/nla/n10b/benign_reads_neutral.jsonl"))
    ap.add_argument("--benign-corpus", default=str(PROJ / "data/nla/n10b/benign_corpus.jsonl"))
    ap.add_argument("--out", default=str(PROJ / "data/nla/n10b/n10b_split_half.json"))
    args = ap.parse_args()

    hard = {}
    for l in open(args.benign_corpus):
        if l.strip():
            r = json.loads(l)
            hard[r["item_id"]] = bool(r.get("hard_negative"))
    items = load(Path(args.mal_neutral), 1) + load(Path(args.ben_neutral), 0, hard)
    hard = [i for i in items if i["label"] == 1 or i.get("hard_negative")]

    rep = {"experiment": "n10b_split_half_reliability", "seed": SEED,
           "n_repeats": N_REPEATS, "thresholds": {"reliable": RELIABLE, "partial": PARTIAL},
           "arm": "neutral (the pre-registered primary arm)",
           "all": analyse(items, "all"),
           "hard_negatives_only": analyse(hard, "malicious + hard-negative benign")}
    rep["finished_utc"] = datetime.now(timezone.utc).isoformat()
    Path(args.out).write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
