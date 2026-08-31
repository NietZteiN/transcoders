"""Gate C1 — is the safe/vulnerable corpus trivially separable by content-free surface features?

WHAT THIS GATE IS FOR. The malware version of this plan needed C1 because malicious packages
are usually obfuscated and benign ones are not, so a triage claim could really be an
obfuscation-detection claim. The sleeper-agents corpus removes that by construction (same task
prompt, same generating model, same language within every pair) — but it introduces a NEW
version of the same risk at the build step: `strip_leaks` deletes class-diagnostic comments,
and those comments were 43% prevalent in the vulnerable class vs 0.9% in the safe class. So
stripping necessarily shortens the vulnerable class more than the safe class. Post-strip
medians are 449 vs 489 chars. If that residue is enough to classify, then every downstream
representational number is partly a length result — which is precisely how HT12 died.

WHAT COUNTS AS A CONFOUND HERE, AND WHAT DOESN'T. A classifier that fires on `'%s' % email`
versus a parameterized query is detecting the actual CWE; that is the signal, not a confound,
and it is not what this gate measures. C1 uses only **content-free** features — lengths, line
geometry, comment/whitespace density, identifier statistics. None of them can express which
API was called or how user input reached a sink. If those alone separate the classes, the
separation is an artifact of how the corpus was assembled.

Cross-validation is **grouped by `question_id`**: the 17 CWE task families have distinctive
shapes (a Flask redirect handler does not look like a tarfile extractor), so an ungrouped fold
lets the model memorize family identity and report an AUC that says nothing about
safe-vs-vulnerable. Grouping forces generalization to unseen task families.

PASS: grouped CV-AUC <= 0.80. Above that, the corpus is reported as trivially separable and
the build is revised (the obvious lever being to length-match the pairs) before any GPU time.

Also reported, because a pooled AUC can hide it: the **within-pair** length asymmetry. Pairs
share task, scaffold and generating model, so a within-pair sign test on `code_chars` is a far
sharper instrument than the marginal medians.

Env: `transcoders-mi` (numpy, sklearn, scipy). CPU only, seconds to run.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

_HERE = Path(__file__).resolve().parent
PROJ = _HERE.parent.parent
SEED = 20260724
PASS_AUC = 0.80

IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
COMMENT = re.compile(r"^[ \t]*(?:#|//)", re.M)
LONG_B64 = re.compile(r"['\"][A-Za-z0-9+/=]{40,}['\"]")

# Content-free only. Deliberately NO feature that names an API, a module, a sink, or a
# string-formatting style — those are the vulnerability itself, not a corpus artifact.
FEATURE_NAMES = [
    "n_chars", "log_n_chars", "n_lines", "mean_line_len", "max_line_len", "std_line_len",
    "blank_line_frac", "comment_line_frac", "leading_ws_mean", "n_idents", "n_uniq_idents",
    "ident_entropy", "mean_ident_len", "punct_frac", "digit_frac", "upper_frac", "n_long_b64",
]


def features(code: str) -> list[float]:
    lines = code.split("\n")
    line_lens = [len(l) for l in lines] or [0]
    idents = IDENT.findall(code)
    uniq = Counter(idents)
    n = max(len(idents), 1)
    ent = -sum((c / n) * math.log2(c / n) for c in uniq.values()) if idents else 0.0
    nc = max(len(code), 1)
    return [
        len(code),
        math.log1p(len(code)),
        len(lines),
        float(np.mean(line_lens)),
        float(np.max(line_lens)),
        float(np.std(line_lens)),
        sum(1 for l in lines if not l.strip()) / max(len(lines), 1),
        len(COMMENT.findall(code)) / max(len(lines), 1),
        float(np.mean([len(l) - len(l.lstrip()) for l in lines])),
        len(idents),
        len(uniq),
        ent,
        float(np.mean([len(i) for i in idents])) if idents else 0.0,
        sum(1 for ch in code if not ch.isalnum() and not ch.isspace()) / nc,
        sum(1 for ch in code if ch.isdigit()) / nc,
        sum(1 for ch in code if ch.isupper()) / nc,
        len(LONG_B64.findall(code)),
    ]


def auc(y: np.ndarray, s: np.ndarray) -> float:
    """Rank-based AUC (Mann-Whitney), ties averaged."""
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), float)
    ranks[order] = np.arange(1, len(s) + 1)
    # average ranks within ties
    su = np.sort(s)
    i = 0
    while i < len(su):
        j = i
        while j + 1 < len(su) and su[j + 1] == su[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2
        i = j + 1
    n1 = float(y.sum())
    n0 = float(len(y) - n1)
    if n1 == 0 or n0 == 0:
        return float("nan")
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(PROJ / "data/nla/sleeper/corpus.jsonl"))
    ap.add_argument("--out", default=str(PROJ / "data/nla/sleeper/c1_gate.json"))
    args = ap.parse_args()

    from scipy.stats import binomtest, mannwhitneyu
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    rows = [json.loads(l) for l in open(args.corpus) if l.strip()]
    by_id = {r["item_id"]: r for r in rows}
    X = np.array([features(r["code"]) for r in rows], float)
    y = np.array([r["label"] == "vulnerable" for r in rows], int)
    groups = np.array([r["question_id"] for r in rows])

    # ---- full content-free surface model, grouped CV -------------------------------
    gkf = GroupKFold(n_splits=min(5, len(set(groups))))
    oof = np.zeros(len(y))
    for tr, te in gkf.split(X, y, groups):
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=2000, random_state=SEED))
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
    surface_auc = auc(y, oof)

    # ---- each feature alone, to name the offender if the gate fails ----------------
    singles = sorted(
        ((name, float(max(a, 1 - a)))
         for name, a in ((FEATURE_NAMES[j], auc(y, X[:, j])) for j in range(X.shape[1]))),
        key=lambda t: -t[1])

    # ---- length alone, the specific risk introduced by strip_leaks -----------------
    len_auc = auc(y, X[:, FEATURE_NAMES.index("n_chars")])
    mw = mannwhitneyu(X[y == 1, 0], X[y == 0, 0], alternative="two-sided")

    # ---- within-pair length asymmetry: the sharp test ------------------------------
    pairs_p = Path(args.corpus).parent / "pairs.jsonl"
    pair_rows = [json.loads(l) for l in open(pairs_p) if l.strip()]
    d = np.array([by_id[p["vulnerable_item_id"]]["code_chars"]
                  - by_id[p["safe_item_id"]]["code_chars"] for p in pair_rows], float)
    n_shorter = int((d < 0).sum())
    n_tied = int((d == 0).sum())
    n_eff = len(d) - n_tied
    sign = binomtest(n_shorter, n_eff, 0.5, alternative="two-sided") if n_eff else None

    passed = bool(surface_auc <= PASS_AUC)
    out: dict[str, Any] = {
        "gate": "C1_surface_separability",
        "seed": SEED,
        "corpus": str(args.corpus),
        "n": len(rows),
        "criterion": f"grouped CV-AUC <= {PASS_AUC}",
        "surface_cv_auc_grouped_by_question_id": round(float(surface_auc), 4),
        "PASS": passed,
        "length_only": {
            "auc_n_chars": round(float(len_auc), 4),
            "median_vulnerable": float(np.median(X[y == 1, 0])),
            "median_safe": float(np.median(X[y == 0, 0])),
            "mannwhitney_p": float(mw.pvalue),
        },
        "within_pair_length": {
            "n_pairs": len(d),
            "n_vulnerable_shorter": n_shorter,
            "n_tied": n_tied,
            "median_delta_chars": float(np.median(d)),
            "sign_test_p": float(sign.pvalue) if sign else None,
        },
        "single_feature_auc_top8": [{"feature": f, "auc": round(a, 4)} for f, a in singles[:8]],
        "finished_utc": datetime.now(timezone.utc).isoformat(),
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"\nGate C1: {'PASS' if passed else 'FAIL'} "
          f"(surface AUC {surface_auc:.4f} vs threshold {PASS_AUC})")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
