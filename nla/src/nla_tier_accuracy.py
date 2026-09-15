"""H-A1/H-A2/H-A3 — the PROMPT-SPACE tier ladder on the host: what does it score when the decoy is simply
absent from the prompt? Pre-registered in log/nla-harness/2026-09-12_tier-ladder-prereg.md.

L1 (nonsense names, original structure) is perfect decoy ERASURE done in token space, so
`rate_L1 - rate_L1b` is the ceiling for any erasure lever (NLA edit, erasure vector, attention masking or
reallocation), and `rate_L0 - rate_L1` is what only RECONSTRUCTION of the true meaning can add. L2/L3 size
the flattening route on this host before the dispatcher-read experiment is designed.

No steering and no autoencoder here -- forward generation only. The greedy / n=8 readouts, grader and
generation settings are kept identical to `nla_accuracy.py` so the two tables are directly comparable.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
_PROJ = _HERE.parents[1]

from nla_accuracy import boot_mean, mcnemar, wilson  # noqa: E402
from nla_ml_gate import load_cfg, load_host, root, select_pairs  # noqa: E402
from nla_tiers import tier_rows  # noqa: E402
from steer_run import MAX_NEW_GEN, build_user, graded  # noqa: E402
from task_bank import build_call  # noqa: E402

TAG = "[TIER]"
TIERS = ("L0", "L1", "L1b", "L2", "L3")
# Paper 3's L2 odds ratio, used only to state the H-A2 expectation in the stats file.
PAPER3_L2_OR = 0.57
BANKED_ACC = {"L0": 0.567, "L1b": 0.533}
CHURN_BAND = 9 / 60         # measured same-condition greedy churn, 6-9 of 60


def clears(d: dict, sign: int) -> bool:
    """CI excludes 0 on the side `sign` (+1: lower bound > 0; -1: upper bound < 0)."""
    lo, hi = d["ci95"]
    return bool(lo > 0) if sign > 0 else bool(hi < 0)


def verdicts(st: dict) -> dict:
    """The frozen H-A1/H-A2/H-A3 rules on the paired rate contrasts."""
    p = st["paired_rate"]
    return {
        "H_A1": "ERASURE-HEADROOM" if clears(p["L1_minus_L1b"], +1) else "ERASURE-FLOOR",
        "H_A2": "FLATTENING-PENALTY" if clears(p["L0_minus_L2"], +1) else "FLATTENING-FLAT",
        "H_A3": "ROUTES-COMPOUND" if clears(p["L3_minus_L2"], -1) else "ROUTES-NOT-ADDITIVE",
    }


def score(rows: list[dict], seed: int) -> dict:
    st: dict = {"experiment": "A_tier_ladder", "n_items": len(rows), "seed": seed, "tiers": list(TIERS),
                "greedy": {}, "sampled": {}, "paired_rate": {}, "vs_L1b_greedy": {}, "summary": []}
    for t in TIERS:
        v = [r[f"{t}_greedy_correct"] for r in rows if r.get(f"{t}_greedy_correct") is not None]
        if v:
            k, n = int(sum(v)), len(v)
            st["greedy"][t] = {"acc": k / n, "k": k, "n": n, "wilson95": list(wilson(k, n)),
                               "parsed": float(np.mean([r[f"{t}_greedy_parsed"] for r in rows
                                                        if r.get(f"{t}_greedy_parsed") is not None]))}
        rv = [r[f"{t}_rate"] for r in rows if f"{t}_rate" in r]
        if rv:
            st["sampled"][t] = boot_mean(rv, seed + 1)
    contrasts = (("L1_minus_L1b", "L1", "L1b"), ("L0_minus_L1", "L0", "L1"), ("L0_minus_L2", "L0", "L2"),
                 ("L3_minus_L2", "L3", "L2"), ("L0_minus_L1b", "L0", "L1b"), ("L1b_minus_L2", "L1b", "L2"))
    for i, (name, a, b) in enumerate(contrasts):
        d = [r[f"{a}_rate"] - r[f"{b}_rate"] for r in rows if f"{a}_rate" in r and f"{b}_rate" in r]
        st["paired_rate"][name] = boot_mean(d, seed + 10 + i)
    for t in TIERS:
        if t == "L1b":
            continue
        pair = [(r[f"{t}_greedy_correct"], r["L1b_greedy_correct"]) for r in rows
                if r.get(f"{t}_greedy_correct") is not None and r.get("L1b_greedy_correct") is not None]
        b = sum(1 for x, y in pair if x and not y); c = sum(1 for x, y in pair if y and not x)
        st["vs_L1b_greedy"][t] = {"n_pairs": len(pair), "tier_right_l1b_wrong": b,
                                  "l1b_right_tier_wrong": c, "mcnemar_p": mcnemar(b, c)}
    if all(k in st["paired_rate"] and st["paired_rate"][k]["n"] for k in
           ("L1_minus_L1b", "L0_minus_L2", "L3_minus_L2")):
        st["verdicts"] = verdicts(st)
    # sanity against the banked greedy numbers (reported, not gated)
    st["banked_check"] = {t: {"banked": BANKED_ACC[t], "here": st["greedy"].get(t, {}).get("acc"),
                              "within_churn": (abs(st["greedy"][t]["acc"] - BANKED_ACC[t]) <= CHURN_BAND
                                               if t in st["greedy"] else None)} for t in BANKED_ACC}
    st["H_A2_expectation"] = {"paper3_L2_or": PAPER3_L2_OR,
                              "predicted_rate_L2_from_L0": (lambda p: (p * PAPER3_L2_OR) / (1 - p + p * PAPER3_L2_OR))(
                                  st["sampled"]["L0"]["mean"]) if "L0" in st["sampled"] else None}
    g = st["greedy"]; s = st["sampled"]; p = st["paired_rate"]
    st["summary"] += [
        "greedy acc: " + " · ".join(f"{t} {g[t]['acc']:.3f}" for t in TIERS if t in g),
        "pass rate:  " + " · ".join(f"{t} {s[t]['mean']:.3f}" for t in TIERS if t in s),
        "paired rate: " + " · ".join(f"{k} {v['mean']:+.3f} [{v['ci95'][0]:+.3f},{v['ci95'][1]:+.3f}]"
                                      for k, v in p.items() if v["n"]),
    ]
    if "verdicts" in st:
        st["summary"].append("verdicts: " + " · ".join(f"{k} {v}" for k, v in st["verdicts"].items()))
    return st


def main() -> int:  # noqa: C901
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(_PROJ / "nla/configs/nla_ml_gate.yaml"))
    ap.add_argument("--root", default=None)
    ap.add_argument("--traces", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n-samples", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--no-sampled", action="store_true")
    ap.add_argument("--max-hours", type=float, default=7.0)
    args = ap.parse_args()

    cfg = load_cfg(Path(args.config))
    if "qwen" in cfg["train"]["host"]["model_id"].lower():
        print(f"{TAG} REFUSED: Chinese models are not run in this project."); return 2
    args._cfg = cfg
    args.repair = bool(cfg["repair"]); args.max_spans_per_item = int(cfg["max_spans_per_item"])
    rt = root(cfg, args.root)
    args.traces = args.traces or str(_PROJ / "data/nla/p0/trace_llr/gemma4b/traces.jsonl")
    out = Path(args.out or (rt / cfg["gate_subdir"] / "tier_ladder")); out.mkdir(parents=True, exist_ok=True)
    seed = int(cfg["seed"]); torch.manual_seed(seed)

    traces = {t["snippet_id"]: t for t in map(json.loads, open(args.traces))}
    tiers = tier_rows()
    model, tok = load_host(cfg, args.device)

    def prompt_ids(code: str, call: str) -> list[int]:
        # the exact templating of trace_llr.py, so L0/L1b reproduce the banked prompt ids bit-for-bit
        return list(tok.apply_chat_template([{"role": "user", "content": build_user(code, call)}],
                                            tokenize=True, add_generation_prompt=True, return_dict=False))

    @torch.no_grad()
    def gen(pids: list[int], n_seq: int, greedy: bool) -> list[str]:
        ids = torch.tensor([list(pids)], device=model.device)
        kw = dict(max_new_tokens=MAX_NEW_GEN, pad_token_id=tok.pad_token_id or 0)
        o = (model.generate(ids, do_sample=False, **kw) if greedy else
             model.generate(ids, do_sample=True, temperature=args.temperature, top_p=args.top_p,
                            num_return_sequences=n_seq, **kw))
        return [tok.decode(o[i, len(pids):], skip_special_tokens=True) for i in range(o.shape[0])]

    sids = [p["snippet_id"] for p in select_pairs(args, traces)]
    if args.limit:
        sids = sids[:args.limit]
    print(f"{TAG} {len(sids)} items · tiers {TIERS} · n_samples {args.n_samples}", flush=True)

    # ── identity gate: the rebuilt L0/L1b prompts must equal the banked trace ids on every item ──
    bad = []
    for sid in sids:
        tr = traces[sid]
        for t, key in (("L0", "l0_prompt_ids"), ("L1b", "l1b_prompt_ids")):
            row = tiers[sid][t]
            if prompt_ids(row["code"], build_call(row)) != list(tr[key]):
                bad.append((sid, t))
    if bad:
        print(f"{TAG} IDENTITY GATE FAILED on {len(bad)} prompts: {bad[:5]}", flush=True)
        (out / "identity_failure.json").write_text(json.dumps(bad))
        return 3
    print(f"{TAG} identity gate: {2*len(sids)}/{2*len(sids)} rebuilt prompts == banked trace ids", flush=True)

    t0 = time.time(); rows: list[dict] = []; skipped = {"no_call": 0, "truth_mismatch": 0, "no_tier": 0}
    for i, sid in enumerate(sids):
        tr = traces[sid]
        row = {"snippet_id": sid, "truth": tr["truth"],
               "banked_l0_correct": bool(tr["l0_correct"]), "banked_l1b_correct": bool(tr["l1b_correct"])}
        for t in TIERS:
            trow = tiers[sid].get(t)
            if not trow:
                skipped["no_tier"] += 1; continue
            call = build_call(trow)
            if not call:
                skipped["no_call"] += 1; continue
            if str(trow.get("expected_output")) != str(tr["truth"]):
                skipped["truth_mismatch"] += 1; continue
            pids = prompt_ids(trow["code"], call)
            g = gen(pids, 1, True)[0]
            got, ok = graded(g, tr["truth"])
            row[f"{t}_greedy_correct"] = bool(ok); row[f"{t}_greedy_parsed"] = got is not None
            row[f"{t}_prompt_len"] = len(pids)
            if not args.no_sampled:
                outs = gen(pids, args.n_samples, False)
                gr = [graded(o, tr["truth"]) for o in outs]
                row[f"{t}_rate"] = float(np.mean([ok for _, ok in gr]))
                row[f"{t}_parsed_rate"] = float(np.mean([g_ is not None for g_, _ in gr]))
        rows.append(row)
        print(f"{TAG} {i+1}/{len(sids)} {sid} · "
              + " ".join(f"{t}={'1' if row.get(f'{t}_greedy_correct') else '0'}"
                         f"/{row.get(f'{t}_rate', float('nan')):.2f}" for t in TIERS)
              + f" · {(time.time()-t0)/60:.1f} min", flush=True)
        if time.time() - t0 > args.max_hours * 3600:
            print(f"{TAG} wall-clock stop after {len(rows)} items", flush=True); break

    (out / "tier_rows.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    st = score(rows, seed); st["skipped"] = skipped
    (out / "tier_stats.json").write_text(json.dumps(st, indent=1))
    for line in st["summary"]:
        print(f"{TAG} {line}", flush=True)
    print(f"{TAG} wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
