"""Two controls that decide how the answer-line result should be described.

CONTROL A — is the answer itself decodable at `answer_line`?
The LATE READOUT ONLY reading says the correctness signal there (rho = 0.5702) is a corollary of
the model having already decided its answer: correctness is (answer == truth) and truth is fixed
per item, so decoding the answer decodes correctness. That predicts a specific, checkable thing —
properties of the ANSWER, which are not correctness, should also be decodable at `answer_line` and
not at `last_prompt`. Answer character length is used because it is a property of the emitted
string that carries no information about whether the string is right.

CONTROL B — does `last_prompt` ~ reply length hold on CLEAN code?
The headline negative is that at `last_prompt` a linear decoder ties a token count (+0.010). If
that is true of the L0 tier as well, it is a fact about the read site. If the residual stream
beats length on L0 and only ties under adversarial renaming, it is a fact about what obfuscation
does to the representation — which is this study's actual subject.

KNOWN LIMITATION, stated rather than worked around: `baseline.jsonl` records `reply_chars` for the
L1b generation only, so **no length baseline exists on disk for the L0 tier.** Control B therefore
compares residual-stream rho across tiers without one, and the L0 length baseline should be
recorded in future runs.

Graded k/n targets throughout, as established 2026-08-30. CPU only.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
PROJ = _HERE.parent.parent
from p1b_consensus_labels import SOURCES  # noqa: E402
from p1b_graded_labels import oof_ridge, spearman  # noqa: E402

SEED = 20260724


def load_votes(field: str) -> tuple[dict[str, list[int]], int]:
    votes: dict[str, list[int]] = {}
    n = 0
    for src in SOURCES:
        p = PROJ / "data/nla/p0" / src / "baseline.jsonl"
        if not p.exists():
            continue
        n += 1
        for l in open(p):
            if l.strip():
                r = json.loads(l)
                votes.setdefault(r["snippet_id"], []).append(int(bool(r[field])))
    return votes, n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos-acts", default=str(PROJ / "data/nla/p0/p1b/pos_acts.npz"))
    ap.add_argument("--tier-acts", default=str(PROJ / "data/nla/p0/p1b/acts.npz"))
    ap.add_argument("--posdep", default=str(PROJ / "data/nla/p0/p1b/position_depth.json"))
    ap.add_argument("--out", default=str(PROJ / "data/nla/p0/p1b/readout_controls.json"))
    args = ap.parse_args()

    rep = {"experiment": "p1b_readout_controls", "seed": SEED,
           "limitation": ("no L0 length baseline exists on disk — baseline.jsonl records "
                          "reply_chars for the L1b generation only"),
           "control_A_answer_decodability": {}, "control_B_tier_comparison": {}}

    # ── CONTROL A ─────────────────────────────────────────────────────────────
    z = np.load(args.pos_acts, allow_pickle=True)
    acts, valid, groups = z["acts"], z["valid"], z["groups"]
    lengths = z["lengths"].astype(float)
    positions = [str(x) for x in z["positions"]]
    replies = {r["snippet_id"]: r for r in json.load(open(args.posdep))["replies"]}

    ans_len, has_ans = [], []
    for sid in groups:
        r = replies.get(str(sid)) or {}
        a = r.get("answer")
        ans_len.append(len(a) if a else np.nan)
        has_ans.append(int(a is not None))
    ans_len = np.array(ans_len, float)
    has_ans = np.array(has_ans, float)

    A = rep["control_A_answer_decodability"]
    A["n_with_answer"] = int(np.nansum(~np.isnan(ans_len)))
    A["answer_len_mean"] = round(float(np.nanmean(ans_len)), 2)
    for tag in ("last_prompt", "answer_line"):
        pi = positions.index(tag)
        m = valid[:, pi] & ~np.isnan(ans_len)
        X, y, g = acts[m][:, pi], ans_len[m], np.array([str(x) for x in groups[m]])
        curve = [round(spearman(y, oof_ridge(X[:, L], y, g)), 4) for L in range(acts.shape[2])]
        best = int(np.argmax(curve))
        # length of the whole reply as the confound: answer length may just track reply length
        rl = round(spearman(y, oof_ridge(lengths[m].reshape(-1, 1), y, g)), 4)
        A[tag] = {"n": int(m.sum()), "rho_by_layer": curve, "argmax_layer": best,
                  "argmax_rho": curve[best], "mean_rho": round(st.mean(curve), 4),
                  "reply_length_baseline_rho": rl}
        print(f"[A] answer-length @ {tag:<12} argmax L{best} {curve[best]:+.4f} "
              f"· mean {st.mean(curve):+.4f} · reply-length baseline {rl:+.4f}", flush=True)
    d = A["answer_line"]["argmax_rho"] - A["last_prompt"]["argmax_rho"]
    A["delta_answer_line_minus_last_prompt"] = round(d, 4)
    A["reading"] = ("SUPPORTS LATE READOUT — the emitted answer is decodable at the answer line "
                    "and not at the prompt, so the correctness signal there is a corollary"
                    if d >= 0.15 else
                    "DOES NOT SUPPORT LATE READOUT — the answer is no more decodable at the "
                    "answer line than at the prompt")

    # ── CONTROL B ─────────────────────────────────────────────────────────────
    z2 = np.load(args.tier_acts, allow_pickle=True)
    tacts, y_tier, tgroups = z2["acts"], z2["y_tier"], z2["groups"]
    B = rep["control_B_tier_comparison"]
    for tier, field, flag in (("L0", "l0_correct", 0), ("L1b", "l1b_correct", 1)):
        votes, n_draws = load_votes(field)
        m = np.array([y_tier[i] == flag and len(votes.get(str(tgroups[i]), [])) == n_draws
                      for i in range(len(y_tier))])
        y = np.array([st.mean(votes[str(tgroups[i])]) for i in range(len(y_tier)) if m[i]])
        g = np.array([str(tgroups[i]) for i in range(len(y_tier)) if m[i]])
        X = tacts[m]
        curve = [round(spearman(y, oof_ridge(X[:, L], y, g)), 4) for L in range(tacts.shape[1])]
        best = int(np.argmax(curve))
        B[tier] = {"n": int(m.sum()), "n_draws": n_draws,
                   "target_mean": round(float(y.mean()), 4),
                   "rho_by_layer": curve, "argmax_layer": best, "argmax_rho": curve[best],
                   "mean_rho": round(st.mean(curve), 4),
                   "rho_L13": curve[13], "rho_L20": curve[20], "rho_L27": curve[27]}
        print(f"[B] {tier:<4} n={int(m.sum()):>3} mean-correct {y.mean():.3f} · "
              f"argmax L{best} {curve[best]:+.4f} · mean {st.mean(curve):+.4f}", flush=True)
    B["delta_L0_minus_L1b_argmax"] = round(B["L0"]["argmax_rho"] - B["L1b"]["argmax_rho"], 4)
    B["note"] = ("compared without a length baseline for L0 (see limitation); the L1b length "
                 "baseline is +0.3626 from 2026-08-30_p1b-graded-labels.md")

    rep["finished_utc"] = datetime.now(timezone.utc).isoformat()
    Path(args.out).write_text(json.dumps(rep, indent=2))
    print("\n[A] " + A["reading"])
    print(f"[B] L0 argmax {B['L0']['argmax_rho']:+.4f} vs L1b {B['L1b']['argmax_rho']:+.4f} "
          f"(delta {B['delta_L0_minus_L1b_argmax']:+.4f})")
    print(f"[ctl] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
