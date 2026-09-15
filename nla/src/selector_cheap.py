"""H-A10a — can DEPLOYABLE features predict which items the NLA write helps, and which it harms?

Pre-registered in log/nla-harness/2026-09-13_selector-cheap-prereg.md. CPU only; reads banked artifacts.

WHY THIS IS NARROW ON PURPOSE. H-A8 found `c3_band` fixes 3 of 7 flippable items and breaks 0, while breaking
6 of 53 non-flippable -- pooled exactly 0.0000. `flippable` is defined from the ground-truth answer, so a
deployable claim needs a selector computable from the OBFUSCATED CODE ALONE. The frozen deployability rule
therefore excludes most of what is banked, and the exclusions matter more than the inclusions:

    dG_S_{swap,c3,edit,foreign,rt}  score the logp of the L0 REPLY          -> needs the clean run. OUT.
    cos_delta_E, ||Delta_s||        are built from h0                        -> needs the clean state. OUT.
    noop_rate, *_greedy_correct     are accuracy                            -> needs the truth. OUT.
    n_spans, n_editable, n_edits    come from the AV read on h1b            -> obfuscated only. IN.
    l1b_prompt_len                  is the obfuscated prompt                -> IN.
    noop_greedy_parsed, parsed_rate are parse events, not correctness        -> IN.

Statistics: per feature, the two-sided rank statistic |AUC - 0.5| against 10,000 label permutations, BH-FDR
across all feature x target tests.

WHY NOT LOO LOGISTIC (the prereg said LOO-CV logistic; this is the same test, stated exactly). A 1-D logistic is
MONOTONE in x, so its leave-one-out score is an affine function of x and its LOO AUC is either AUC(y, x) or
1 - AUC(y, x). The cross-validation was therefore doing exactly one thing: choosing the sign without peeking at
the labels. A two-sided statistic makes the sign free, so |AUC - 0.5| with a two-sided permutation null is the
identical hypothesis test with no fitting at all -- and it runs 10,000 permutations instead of 100. The reported
`auc` is max(a, 1 - a) with `direction` recording which way the feature points; the null statistic is two-sided
too, so selecting the sign costs nothing and inflates nothing.

With 7 and 6 positives only a large effect is detectable, so ABSENT is evidence that cheap features are
insufficient -- not that no selector exists.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

_PROJ = Path(__file__).resolve().parents[2]
TAG = "[SEL]"
SEED = 20260724
N_PERM = 10_000
AUC_BAR = 0.70
FEATURES = ("n_spans", "n_editable", "frac_editable", "sum_n_edits",
            "l1b_prompt_len", "noop_greedy_parsed", "noop_parsed_rate")


def auc(y: np.ndarray, s: np.ndarray) -> float:
    """Rank-based AUC with ties averaged; returns nan if a class is absent."""
    pos, neg = s[y == 1], s[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order), float)
    cat = np.concatenate([pos, neg])[order]
    i = 0
    while i < len(cat):                      # average ranks within tie groups
        j = i
        while j + 1 < len(cat) and cat[j + 1] == cat[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=str(_PROJ / "data/nla/ml/gemma4b"))
    ap.add_argument("--traces", default=str(_PROJ / "data/nla/p0/trace_llr/gemma4b/traces.jsonl"))
    ap.add_argument("--rows-2600", default="/scratch/juno/jvl210002/nla_a8_budget/b2600/accuracy_rows.jsonl")
    ap.add_argument("--out", default=str(_PROJ / "nla/data/selector_cheap"))
    args = ap.parse_args()
    rt = Path(args.root)

    spans: dict[str, list[dict]] = {}
    for r in map(json.loads, open(rt / "gate/vectors/L7_spans.jsonl")):
        spans.setdefault(r["snippet_id"], []).append(r)
    traces = {t["snippet_id"]: t for t in map(json.loads, open(args.traces))}
    acc = {r["snippet_id"]: r for r in map(json.loads, open(rt / "gate/accuracy/accuracy_rows.jsonl"))}
    p26 = Path(args.rows_2600)
    b26 = {r["snippet_id"]: r for r in map(json.loads, open(p26))} if p26.exists() else {}
    if not b26:
        print(f"{TAG} REFUSED: the 2600-token rows are required for the `harmed` target: {p26}")
        return 2

    rows = []
    for sid, sp in sorted(spans.items()):
        if sid not in acc or sid not in traces or sid not in b26:
            continue
        a, t, n = acc[sid], traces[sid], b26[sid]
        rows.append({
            "snippet_id": sid,
            "n_spans": float(len(sp)),
            "n_editable": float(sum(bool(q["editable"]) for q in sp)),
            "frac_editable": float(np.mean([bool(q["editable"]) for q in sp])),
            "sum_n_edits": float(sum(int(q.get("n_edits") or 0) for q in sp)),
            "l1b_prompt_len": float(len(t["l1b_prompt_ids"])),
            "noop_greedy_parsed": float(bool(a["noop_greedy_parsed"])),
            "noop_parsed_rate": float(a.get("noop_parsed_rate", float("nan"))),
            # targets
            "flippable": int(bool(a["flippable"])),
            "harmed": int(bool(n["noop_greedy_correct"]) and not bool(n["c3_band_greedy_correct"])),
        })
    print(f"{TAG} {len(rows)} items · flippable {sum(r['flippable'] for r in rows)} · "
          f"harmed {sum(r['harmed'] for r in rows)}", flush=True)

    rng = np.random.default_rng(SEED)
    st: dict = {"experiment": "A10a_cheap_selector", "seed": SEED, "n_perm": N_PERM, "auc_bar": AUC_BAR,
                "n_items": len(rows), "features": list(FEATURES), "tests": {}, "summary": []}
    tests = []
    for target in ("flippable", "harmed"):
        y = np.array([r[target] for r in rows])
        st[f"n_{target}"] = int(y.sum())
        for f in FEATURES:
            x = np.array([r[f] for r in rows])
            if not np.isfinite(x).all() or x.std() == 0:
                st["tests"][f"{target}:{f}"] = {"auc": float("nan"), "note": "constant or missing"}
                continue
            a_raw = auc(y, x)
            a_obs = max(a_raw, 1 - a_raw)          # two-sided: the sign is free (see module docstring)
            perm = np.array([auc(rng.permutation(y), x) for _ in range(N_PERM)])
            perm2 = np.maximum(perm, 1 - perm)
            p = float((perm2 >= a_obs).mean())
            d = {"auc": a_obs, "auc_raw": a_raw, "direction": ("higher->positive" if a_raw >= 0.5
                                                               else "lower->positive"),
                 "p_perm": p, "n_perm": int(N_PERM),
                 "perm_auc_mean": float(perm2.mean()), "perm_auc_p95": float(np.percentile(perm2, 95))}
            st["tests"][f"{target}:{f}"] = d
            tests.append((f"{target}:{f}", p, d))
    # BH-FDR across every test that produced a p-value
    tests.sort(key=lambda kv: kv[1])
    m = len(tests)
    for i, (k, p, d) in enumerate(tests, 1):
        d["q_bh"] = min(1.0, p * m / i)
    for i in range(m - 2, -1, -1):                      # enforce monotonicity
        tests[i][2]["q_bh"] = min(tests[i][2]["q_bh"], tests[i + 1][2]["q_bh"])
    hits = [(k, d) for k, _, d in tests if d["auc"] >= AUC_BAR and d["q_bh"] < 0.05]
    st["verdict"] = "SELECTOR-CHEAP-FOUND" if hits else "SELECTOR-CHEAP-ABSENT"
    st["hits"] = [k for k, _ in hits]
    st["note"] = ("Only a LARGE effect is detectable at 7/6 positives: ABSENT is evidence that cheap structural "
                  "features are insufficient, NOT that no selector exists. Licenses H-A10b (answer entropy on "
                  "the L1b prompt, this host, ~2-3 GPU-h).")
    for k, _, d in tests:
        st["summary"].append(f"{k:34s} AUC {d['auc']:.3f} ({d['direction']:16s})  p_perm {d['p_perm']:.4f}  "
                             f"q_BH {d['q_bh']:.4f}   (null mean {d['perm_auc_mean']:.3f}, "
                             f"p95 {d['perm_auc_p95']:.3f})")
    st["summary"].append(f"H-A10a {st['verdict']} (bar AUC >= {AUC_BAR}, q < 0.05) · hits {st['hits'] or 'none'}")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "selector_stats.json").write_text(json.dumps(st, indent=1))
    with open(out / "selector_rows.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print("\n".join(st["summary"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
