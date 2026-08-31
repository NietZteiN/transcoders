"""The behavioural predictors, consolidated — and how they degrade under obfuscation.

WHY. Every internal measure in this programme loses to a behavioural one. Reply length predicts
correctness at rho = +0.4770 (clean) / +0.3203 (renamed); N13's answer instability reaches
AUC 0.869; the residual stream at `last_prompt` adds +0.04-0.06 over a token count. If the readout
paper needs a positive, the honest one may be behavioural — and the interesting result is not that
these work, but that they work LESS WELL once identifiers are adversarial.

Answer instability is computable for free here. The ten independent unsteered draws on disk differ
only through the nondeterminism floor, so the spread of an item's answers across draws IS an
instability measure — and unlike N13's, it needs no extra sampling.

Three predictors, both tiers, each against correctness:
  reply length        mean chars across draws
  answer instability  1 - (modal answer's share of draws); 0 = same answer every time
  parse failure rate  fraction of draws emitting no `Output:` line

No GPU.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
PROJ = _HERE.parent.parent
from p1b_consensus_labels import SOURCES  # noqa: E402
from p1b_graded_labels import spearman  # noqa: E402


def auc(labels, scores) -> float:
    pairs = sorted(zip(scores, labels))
    ranks, i, vals = {}, 0, [p[0] for p in pairs]
    while i < len(vals):
        j = i
        while j + 1 < len(vals) and vals[j + 1] == vals[i]:
            j += 1
        for k in range(i, j + 1):
            ranks[k] = (i + j + 2) / 2
        i = j + 1
    pos = [ranks[k] for k, (_, l) in enumerate(pairs) if l == 1]
    n1, n0 = len(pos), len(pairs) - len(pos)
    return float("nan") if n1 == 0 or n0 == 0 else \
        (sum(pos) - n1 * (n1 + 1) / 2) / (n1 * n0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(PROJ / "data/nla/p0/p1b/behavioural.json"))
    args = ap.parse_args()

    # per-tier, per-item: list of (correct, answer, parsed, reply_chars) across draws
    per: dict[str, dict[str, list[dict]]] = {"L0": {}, "L1b": {}}
    n_src = 0
    for src in SOURCES:
        p = PROJ / "data/nla/p0" / src / "baseline.jsonl"
        if not p.exists():
            continue
        n_src += 1
        for l in open(p):
            if not l.strip():
                continue
            r = json.loads(l)
            sid = r["snippet_id"]
            per["L0"].setdefault(sid, []).append(
                {"correct": bool(r["l0_correct"]), "answer": r.get("l0_answer"),
                 "parsed": bool(r.get("l0_parsed")),
                 "chars": r.get("l0_reply_chars")})
            per["L1b"].setdefault(sid, []).append(
                {"correct": bool(r["l1b_correct"]), "answer": r.get("l1b_answer"),
                 "parsed": bool(r.get("l1b_parsed")),
                 "chars": r.get("l1b_reply_chars", r.get("reply_chars"))})

    rep = {"experiment": "p1b_behavioural_predictors", "n_draws": n_src,
           "instability": "1 - modal answer share across draws", "tiers": {}}

    for tier, d in per.items():
        sids = [s for s, v in d.items() if len(v) == n_src]
        y_graded = [st.mean(x["correct"] for x in d[s]) for s in sids]
        y_bin = [int(m > 0.5) for m in y_graded]
        # reply length is only recorded for L0 in the newest pass, so drop items missing it
        keep = [i for i, s in enumerate(sids)
                if all(x["chars"] is not None for x in d[s])]
        length = [st.mean(d[sids[i]][k]["chars"] for k in range(n_src)) for i in keep]
        instab = []
        parsefail = []
        for s in sids:
            answers = [x["answer"] for x in d[s]]
            c = Counter(answers)
            instab.append(1 - c.most_common(1)[0][1] / len(answers))
            parsefail.append(1 - st.mean(x["parsed"] for x in d[s]))
        block = {
            "n_items": len(sids), "n_with_length": len(keep),
            "mean_correct": round(st.mean(y_graded), 4),
            "mean_instability": round(st.mean(instab), 4),
            "mean_parse_failure": round(st.mean(parsefail), 4),
            "instability_rho_vs_correct": round(spearman(instab, y_graded), 4),
            "instability_auc": round(auc(y_bin, [-v for v in instab]), 4),
            "parsefail_rho_vs_correct": round(spearman(parsefail, y_graded), 4),
        }
        if keep:
            block["length_rho_vs_correct"] = round(
                spearman(length, [y_graded[i] for i in keep]), 4)
        rep["tiers"][tier] = block
        print(f"[beh] {tier:<4} n={len(sids):>3} acc {block['mean_correct']:.3f} · "
              f"instability {block['mean_instability']:.3f} "
              f"(rho {block['instability_rho_vs_correct']:+.4f}, "
              f"AUC {block['instability_auc']:.4f}) · "
              f"parse-fail {block['mean_parse_failure']:.3f} "
              f"(rho {block['parsefail_rho_vs_correct']:+.4f})", flush=True)

    # ── split-draw control ────────────────────────────────────────────────────
    # Instability and graded correctness above are computed from the SAME draws, and they are
    # mechanically entangled: y = 0.5 forces at least two distinct answers, hence instability
    # >= 0.5. The correlation reported is with y rather than |y - 0.5|, so it is not circular —
    # consistently WRONG items also have low instability — but the dependency should be removed
    # rather than argued about. Here instability comes from the first half of the draws and
    # correctness from the disjoint second half, so the two share no generation.
    half = n_src // 2
    for tier, d in per.items():
        sids = [s for s, v in d.items() if len(v) == n_src]
        y_out = [st.mean(x["correct"] for x in d[s][half:]) for s in sids]
        inst_in, pf_in = [], []
        for s in sids:
            answers = [x["answer"] for x in d[s][:half]]
            c = Counter(answers)
            inst_in.append(1 - c.most_common(1)[0][1] / len(answers))
            pf_in.append(1 - st.mean(x["parsed"] for x in d[s][:half]))
        y_bin = [int(m > 0.5) for m in y_out]
        rep["tiers"][tier]["split_draw"] = {
            "instability_draws": f"1-{half}", "correctness_draws": f"{half+1}-{n_src}",
            "instability_rho_vs_correct": round(spearman(inst_in, y_out), 4),
            "instability_auc": round(auc(y_bin, [-v for v in inst_in]), 4),
            "parsefail_rho_vs_correct": round(spearman(pf_in, y_out), 4),
        }
        sd = rep["tiers"][tier]["split_draw"]
        print(f"[beh:split] {tier:<4} instability(draws 1-{half}) vs correctness(draws "
              f"{half+1}-{n_src}): rho {sd['instability_rho_vs_correct']:+.4f} "
              f"AUC {sd['instability_auc']:.4f} · parse-fail rho "
              f"{sd['parsefail_rho_vs_correct']:+.4f}", flush=True)

    a, b = rep["tiers"]["L0"], rep["tiers"]["L1b"]
    rep["degradation"] = {
        "instability_rho": round(a["instability_rho_vs_correct"] - b["instability_rho_vs_correct"], 4),
        "instability_auc": round(a["instability_auc"] - b["instability_auc"], 4),
        "parsefail_rho": round(a["parsefail_rho_vs_correct"] - b["parsefail_rho_vs_correct"], 4),
        "mean_instability_increase": round(b["mean_instability"] - a["mean_instability"], 4),
    }
    rep["finished_utc"] = datetime.now(timezone.utc).isoformat()
    Path(args.out).write_text(json.dumps(rep, indent=2))
    print("\n" + json.dumps(rep["degradation"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
