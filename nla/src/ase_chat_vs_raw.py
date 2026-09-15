"""H-R7e (descriptive) + per-case parse rates for the chat-template bake-off.

Pre-registered in log/nla-harness/2026-09-15_chat-template-rerun-prereg.md: for every arm present in BOTH
the raw-prompt dir (H-R2/H-R6 runs) and the chat-template dir (H-R7), report `c/n` accuracy under each
prompt form and the paired chat - raw difference (cluster bootstrap over snippets, N_BOOT 10 000, seed
20260724, same estimator as ase_bakeoff_stats.paired). Nothing here enters a verdict -- the H-R7 rules
read from ase_bakeoff_stats.py on the chat dir alone; this is the runtime-comparison table for the report.

Per-case parse (log/nla-harness/2026-09-15_parse-rate-correction.md): a case counts as parsed iff
`pred.get(case_id) is not None`, over cases x runs; the banked `parsed_frac` counts phantom keys and is NOT
used. acc|p = accuracy restricted to parsed cases.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ase_bakeoff_stats import load, load_truth, paired, N_BOOT, SEED  # noqa: E402  (same estimator)


def parse_stats(path: Path, truth: dict) -> dict:
    parsed = correct_p = total = 0
    for l in open(path):
        if not l.strip():
            continue
        r = json.loads(l); tr = truth[r["snippet"]]
        for run in r["runs"]:
            for c, t in tr.items():
                total += 1
                p = run["pred"].get(c)
                if p is not None:
                    parsed += 1
                    correct_p += int(p == t)
    return {"parse": parsed / total, "acc_given_parsed": (correct_p / parsed) if parsed else float("nan"),
            "n_case_runs": total}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", required=True, help="raw-prompt bakeoff dir")
    ap.add_argument("--chat", required=True, help="chat-template bakeoff dir")
    ap.add_argument("--packs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    truth = load_truth(args.packs)
    raw, chat = Path(args.raw), Path(args.chat)
    arms = sorted(p.stem for p in chat.glob("*.jsonl"))
    out = {"experiment": "H-R7e_chat_vs_raw", "n_boot": N_BOOT, "seed": SEED, "arms": {}}
    for i, arm in enumerate(arms):
        c = load(chat / f"{arm}.jsonl", "cn", truth)
        row = {"chat": {**parse_stats(chat / f"{arm}.jsonl", truth)}}
        row["chat"]["acc"] = float(sum(s * n for s, n in c.values()) / sum(n for _, n in c.values()))
        if (raw / f"{arm}.jsonl").exists():
            r = load(raw / f"{arm}.jsonl", "cn", truth)
            row["raw"] = {**parse_stats(raw / f"{arm}.jsonl", truth)}
            row["raw"]["acc"] = float(sum(s * n for s, n in r.values()) / sum(n for _, n in r.values()))
            row["chat_minus_raw"] = paired(c, r, 800 + i)
        out["arms"][arm] = row
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"{'arm':18s} {'chat acc':>8s} {'parse':>6s} {'acc|p':>6s}   {'raw acc':>8s} {'parse':>6s} {'acc|p':>6s}   chat-raw [95% CI]")
    for arm, row in out["arms"].items():
        ch = row["chat"]; s = f"{arm:18s} {ch['acc']:8.3f} {ch['parse']:6.2f} {ch['acc_given_parsed']:6.2f}"
        if "raw" in row:
            rw = row["raw"]; d = row["chat_minus_raw"]
            s += f"   {rw['acc']:8.3f} {rw['parse']:6.2f} {rw['acc_given_parsed']:6.2f}   {d['mean']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]"
        print(s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
