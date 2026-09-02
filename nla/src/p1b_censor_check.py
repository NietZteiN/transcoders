"""Does truncation censoring manufacture the reply-length baseline?

WHY THIS DECIDES SOMETHING EXPENSIVE. The read-side battery's central comparison is
"residual stream vs reply length". An item that never emits an `Output:` line is scored WRONG,
and it is long — that is why it hit the cap. So censoring makes long replies wrong *by
construction*, which inflates exactly the baseline the residual stream has to beat. If the
length baseline survives dropping censored items, parse rates of 0.90-0.97 are a nuisance and
the battery can run on what exists. If it collapses, the ladder needs regenerating at a bigger
budget before any of it means anything.

Reported per host and tier: the graded length baseline on ALL items vs on PARSED-ONLY items.
CPU only.
"""
from __future__ import annotations
import argparse, json, statistics as st, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
PROJ = _HERE.parent.parent
from p1b_graded_labels import oof_ridge, spearman  # noqa: E402
from p1b_ladder import read_draws, split_terminated  # noqa: E402

HOSTS = {"qwen7b": "ladder", "gemma12b": "ladder_gemma12b", "llama8b": "ladder_llama8b"}
TIERS = ("L0", "L1", "L1b", "L2", "L3")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(PROJ / "data/nla/p0/p1b/censor_check.json"))
    args = ap.parse_args()
    rep = {"experiment": "p1b_censoring_sensitivity", "hosts": {}}

    for host, sub in HOSTS.items():
        rep["hosts"][host] = {}
        for tier in TIERS:
            d = PROJ / "data/nla/p0/p1b" / sub / tier
            rows = read_draws(d)
            if not rows:
                continue
            by: dict[str, list[dict]] = {}
            for r in rows:
                by.setdefault(r["snippet_id"], []).append(r)
            sids = sorted(by)
            parse = st.mean(r.get("parsed", True) for r in rows)

            # all draws
            y_all = np.array([st.mean(r["correct"] for r in by[s]) for s in sids])
            l_all = np.array([[st.mean(r["reply_chars"] for r in by[s])] for s in sids], float)
            g = np.array(sids)
            rho_all = spearman(y_all, oof_ridge(l_all, y_all, g))

            # parsed draws only — an item with no parsed draw at all is dropped
            keep, y_p, l_p = [], [], []
            for s in sids:
                pr = [r for r in by[s] if r.get("parsed", True)]
                if not pr:
                    continue
                keep.append(s)
                y_p.append(st.mean(r["correct"] for r in pr))
                l_p.append([st.mean(r["reply_chars"] for r in pr)])
            y_p, l_p, g_p = np.array(y_p), np.array(l_p, float), np.array(keep)
            rho_parsed = spearman(y_p, oof_ridge(l_p, y_p, g_p))

            rep["hosts"][host][tier] = {
                "parse_rate": round(parse, 4), "n_items": len(sids),
                "n_items_parsed": len(keep),
                "mean_correct_all": round(float(y_all.mean()), 4),
                "mean_correct_parsed": round(float(y_p.mean()), 4),
                "length_rho_all": round(rho_all, 4),
                "length_rho_parsed_only": round(rho_parsed, 4),
                "delta": round(rho_parsed - rho_all, 4),
            }
            b = rep["hosts"][host][tier]
            print(f"  {host:<9}{tier:<5}parse={parse:.4f}  acc {b['mean_correct_all']:.4f}"
                  f"->{b['mean_correct_parsed']:.4f}  length-rho {rho_all:+.4f}"
                  f"->{rho_parsed:+.4f}  ({b['delta']:+.4f})", flush=True)

    deltas = [t["delta"] for h in rep["hosts"].values() for t in h.values()]
    rep["summary"] = {
        "mean_delta": round(st.mean(deltas), 4) if deltas else None,
        "max_abs_delta": round(max(abs(d) for d in deltas), 4) if deltas else None,
        "verdict": ("CENSORING DOES NOT DRIVE THE LENGTH BASELINE — battery can run as-is"
                    if deltas and max(abs(d) for d in deltas) < 0.10 else
                    "CENSORING MATERIALLY MOVES THE BASELINE — regenerate at a bigger budget"),
    }
    rep["finished_utc"] = datetime.now(timezone.utc).isoformat()
    Path(args.out).write_text(json.dumps(rep, indent=2))
    print("\n" + json.dumps(rep["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
