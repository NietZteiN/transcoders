"""Is the replicated L2 read effect more than static dispatcher complexity?

Pre-registration: log/nla-harness/2026-08-31_l2-mechanism-prereg.md.

The deflationary hypothesis is specific and already documented: dispatcher complexity correlates
with accuracy at r = -0.196 in Papers 2-3, so a residual stream that merely encodes "this program
has N dispatcher sites" would reproduce the replicated effect without any state tracking. The test
is therefore incremental value over a baseline that already knows reply length AND the code's
static shape — the same discipline that killed the tier positive on 2026-08-30.

CPU only.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics as st
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
PROJ = _HERE.parent.parent
from p1b_ladder import read_draws, repetition_features, split_terminated  # noqa: E402
from p1b_graded_labels import oof_ridge, spearman  # noqa: E402

SEED, N_PERM, BAR = 20260724, 200, 0.10


def stimuli(tier: str) -> dict[str, dict]:
    rows = []
    for ds in ("dataset_a", "dataset_b"):
        p = PROJ / "data" / "stimuli" / ds / f"{ds}.jsonl"
        if p.exists():
            rows += [json.loads(l) for l in open(p) if l.strip()]
    return {r["snippet_id"]: r for r in rows if r["tier"] == tier}


def static_features(r: dict) -> list[float]:
    code = r["code"]
    return [
        float(len(r.get("dispatcher_spans") or [])),
        float(len(code)),
        float(code.count("\n") + 1),
        float(bool(re.search(r"\bwhile\b", code))),
        float(r.get("language") == "javascript"),
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiers", default="L2,L1b")
    ap.add_argument("--root", default=str(PROJ / "data/nla/p0/p1b/ladder"))
    # Replication scoring must exclude the discovery draws: a replication that includes its own
    # discovery data is not a replication.
    ap.add_argument("--draws", default=None, help="draw indices to score, e.g. '10-14'")
    ap.add_argument("--out", default=str(PROJ / "data/nla/p0/p1b/l2_mechanism.json"))
    args = ap.parse_args()
    root = Path(args.root)
    sel = None
    if args.draws:
        sel = set()
        for part in args.draws.split(","):
            if "-" in part:
                a, b_ = part.split("-"); sel.update(range(int(a), int(b_) + 1))
            else:
                sel.add(int(part))

    rep = {"experiment": "p1b_l2_mechanism", "seed": SEED, "bar": BAR,
           "prereg": "log/nla-harness/2026-08-31_l2-mechanism-prereg.md",
           "draws_scored": sorted(sel) if sel else "all",
           "static_features": ["n_dispatcher_spans", "code_chars", "n_lines",
                               "has_while", "is_javascript"],
           "tiers": {}}

    for tier in [t.strip() for t in args.tiers.split(",")]:
        d = root / tier
        acts = np.load(d / "acts.npy")
        sids = json.loads((d / "items.json").read_text())
        stim = stimuli(tier)
        corr: dict[str, list[int]] = {}
        chars: dict[str, list[int]] = {}
        rows_t = split_terminated(read_draws(d))[0]   # terminated rows only
        if sel is not None:
            rows_t = [r for r in rows_t if r["draw"] in sel]
        for r in rows_t:
            corr.setdefault(r["snippet_id"], []).append(int(r["correct"]))
            chars.setdefault(r["snippet_id"], []).append(int(r["reply_chars"]))
        keep = [i for i, s in enumerate(sids) if s in corr and s in stim]
        y = np.array([st.mean(corr[sids[i]]) for i in keep])
        g = np.array([sids[i] for i in keep])
        X = acts[keep]
        ln = np.array([[st.mean(chars[sids[i]])] for i in keep], float)
        S = np.array([static_features(stim[sids[i]]) for i in keep], float)
        # The repetition control, applied to THIS effect rather than only to the span probe.
        # On Qwen, five surface counts of the source text reached rho +0.8346 against the residual
        # stream's +0.8842 — that is what turned the analogous structural claim into a
        # surface-statistical one. If the Llama L2 effect is repetition in disguise, adding these
        # to the baseline removes it; if it survives, the effect is not text repetition.
        R = np.array([repetition_features(stim[sids[i]]["code"]) for i in keep], float)
        C = np.hstack([ln, S])
        CR = np.hstack([ln, S, R])

        rho_len = spearman(y, oof_ridge(ln, y, g))
        rho_stat = spearman(y, oof_ridge(S, y, g))
        rho_comb = spearman(y, oof_ridge(C, y, g))
        rho_rep = spearman(y, oof_ridge(R, y, g))
        rho_all = spearman(y, oof_ridge(CR, y, g))
        # The combined ridge can score WORSE out-of-fold than its own best component — at n = 60
        # with grouped folds it happens in 6 of 9 cells here, by as much as 0.12. When it does,
        # "beats the combined baseline" is measuring a degenerate baseline rather than signal, and
        # it is exactly the cells with the most degradation that produced positive verdicts. The
        # baseline is therefore the BEST of the three, which no amount of ridge misbehaviour can
        # deflate. Reported alongside so the degeneracy stays visible instead of being smoothed.
        rho_strict = max(rho_len, rho_stat, rho_comb, rho_rep, rho_all)
        curve = [spearman(y, oof_ridge(X[:, L], y, g)) for L in range(X.shape[1])]
        mean_rho = st.mean(curve)

        rng = np.random.default_rng(SEED)
        null = []
        for _ in range(N_PERM):
            yy = rng.permutation(y)
            null.append(st.mean(spearman(yy, oof_ridge(X[:, L], yy, g))
                                for L in range(X.shape[1])))
        p = (sum(1 for v in null if v >= mean_rho) + 1) / (N_PERM + 1)
        beats = mean_rho - rho_strict
        block = {
            "n": len(keep),
            "dispatcher_spans_mean": round(float(S[:, 0].mean()), 2),
            "rho_length_only": round(rho_len, 4),
            "rho_static_only": round(rho_stat, 4),
            "rho_combined_baseline": round(rho_comb, 4),
            "rho_repetition_only": round(rho_rep, 4),
            "rho_length_static_repetition": round(rho_all, 4),
            "rho_strict_baseline": round(rho_strict, 4),
            "combined_degraded_vs_best_single": round(rho_comb - max(rho_len, rho_stat), 4),
            "rho_residual_mean": round(mean_rho, 4),
            "beats_strict_by": round(beats, 4),
            "perm_p": round(p, 5),
            "perm_null_mean": round(st.mean(null), 4),
            "verdict": ("BEYOND LENGTH, STATIC SHAPE AND REPETITION"
                        if beats >= BAR and p < 0.05
                        else "REDUCES TO A SURFACE STATISTIC"),
        }
        rep["tiers"][tier] = block
        print(f"[l2] {tier:<4} n={len(keep):>3} · length {rho_len:+.4f} · static {rho_stat:+.4f} "
              f"· rep {rho_rep:+.4f} · combined {rho_comb:+.4f} · strict {rho_strict:+.4f} "
              f"· residual {mean_rho:+.4f} · beats {beats:+.4f} (p={p:.4f}) "
              f"-> {block['verdict']}", flush=True)

    rep["finished_utc"] = datetime.now(timezone.utc).isoformat()
    Path(args.out).write_text(json.dumps(rep, indent=2))
    print("\n" + json.dumps(rep["tiers"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
