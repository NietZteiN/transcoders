"""Does steering the NLA improve ACCURACY? Generated answers under the best steering configurations.

Run at the user's explicit request (2026-09-12: *"The result I want to see is if accuracy improved by
steering the nla"*), **notwithstanding H-S9's gate not firing** (+1.67 < +3.00 in
`beta_sweep_stats.json`). Pre-registered in log/nla-harness/2026-09-12_accuracy-prereg.md.

THE BOUND, STATED IN THE CODE SO IT TRAVELS WITH THE NUMBERS. This host scores acc_l0 0.567 ->
acc_l1b 0.533: 2 of 60 items of net headroom, 7 flippable, against a same-condition greedy churn of
6-9 items of 60. Exact McNemar at n=60: a PERFECT rescue of all 7 flippable items against the measured
churn gives p = 0.092 -- it cannot reach 0.05. So no arm here can support a positive significance
claim, and this module reports effect sizes with Wilson/bootstrap intervals and McNemar p-values
WITHOUT a verdict word. See docs/nla_flippable_corpus_scoping.md for the arithmetic.

Two readouts, because they answer different things:
  * GREEDY n=1 -- directly comparable with every banked accuracy number in this programme.
  * SAMPLED n=8 per item -> a per-item PASS RATE. ~2.8x smaller per-item SE than one greedy draw
    (0.177 vs 0.500 at p=0.5), the only cheap lever on the churn floor. Batch 1 with
    `num_return_sequences`, never left-padded batching: `answer_entropy.py:186-190` pads on the LEFT,
    which would shift every absolute position and silently break PositionReplacer's span targeting.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
_PROJ = _HERE.parents[1]

from nla_ml_gate import load_cfg, load_host, load_vectors, root, select_pairs  # noqa: E402
from steer import PositionReplacer  # noqa: E402
from steer_run import MAX_NEW_GEN, graded  # noqa: E402

TAG = "[ACC]"
# The best configuration measured on the content-specific effect (2026-09-12_beta-layerset-results.md:
# `band` L2-L13 at beta=0.35, SPEC +9.85) plus the banked single-layer standard for comparability.
BAND = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
BETA_BAND = 0.35
SINGLE = [2]          # best_single_edit_layer from the Phase-B gate
BETA_SINGLE = 1.0
VEC_KEY = {"c3": "c3", "edit": "edit", "foreign": "foreign", "rt": "rt", "swap": "h0"}
# "the identifiers may be misleading" -- the mandatory prompting baseline (CLAUDE.md §4).
WARN = ("Note: the identifiers in this code may be misleading. Reason about what the code actually "
        "does, not what the names suggest.\n\n")
ARMS = ("noop", "c3_band", "edit_band", "foreign_band", "edit_single", "prompt")


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    z, p = 1.959964, k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (max(0.0, c - h), min(1.0, c + h))


def mcnemar(b: int, c: int) -> float:
    """Exact two-sided McNemar: binomial test on b of b+c at p=0.5."""
    from math import comb
    n = b + c
    if n == 0:
        return 1.0
    k = max(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n)


def boot_mean(x, seed: int, n_boot: int = 10_000) -> dict:
    x = np.asarray(x, dtype=float)
    if not len(x):
        return {"mean": float("nan"), "ci95": [float("nan")] * 2, "n": 0}
    r = np.random.default_rng(seed)
    b = np.sort(x[r.integers(0, len(x), size=(n_boot, len(x)))].mean(1))
    return {"mean": float(x.mean()),
            "ci95": [float(b[int(.025 * n_boot)]), float(b[int(.975 * n_boot)])], "n": int(len(x))}


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
    ap.add_argument("--no-sampled", action="store_true", help="greedy pass only")
    ap.add_argument("--max-hours", type=float, default=5.0)
    # H-A8 (prereg 2026-09-13_budget-confound-prereg.md). The banked accuracy run (job 391968) generated at
    # steer_run.MAX_NEW_GEN = 1100 and ~22 % of its generations never emitted a parseable answer, on EVERY arm
    # (parsed rate 0.78-0.80). An unparsed generation scores as wrong, so that mass was inaccessible to every
    # steering arm; CLAUDE.md's own anchor finding is a ~2,048-token System-2 plateau. These two flags let the
    # budget be varied and the arm list trimmed so the instrument can be checked before any further steering.
    ap.add_argument("--max-new-gen", type=int, default=None,
                    help=f"generation budget in new tokens (default steer_run.MAX_NEW_GEN = {MAX_NEW_GEN})")
    ap.add_argument("--arms", default=None,
                    help="comma-separated subset of " + ",".join(ARMS))
    args = ap.parse_args()
    max_new = int(args.max_new_gen or MAX_NEW_GEN)
    arms = tuple(a.strip() for a in args.arms.split(",")) if args.arms else ARMS
    bad = [a for a in arms if a not in ARMS]
    if bad:
        print(f"{TAG} REFUSED: unknown arm(s) {bad}; known {list(ARMS)}"); return 2

    cfg = load_cfg(Path(args.config))
    if "qwen" in cfg["train"]["host"]["model_id"].lower():
        print(f"{TAG} REFUSED: Chinese models are not run in this project."); return 2
    args._cfg = cfg
    args.repair = bool(cfg["repair"]); args.max_spans_per_item = int(cfg["max_spans_per_item"])
    rt = root(cfg, args.root)
    args.traces = args.traces or str(_PROJ / "data/nla/p0/trace_llr/gemma4b/traces.jsonl")
    vdir = rt / cfg["gate_subdir"] / "vectors"
    out = Path(args.out or (rt / cfg["gate_subdir"] / "accuracy")); out.mkdir(parents=True, exist_ok=True)
    seed = int(cfg["seed"])
    torch.manual_seed(seed)

    traces = {t["snippet_id"]: t for t in map(json.loads, open(args.traces))}
    layers = sorted(set(BAND) | set(SINGLE))
    vec = {K: load_vectors(vdir, K) for K in layers}
    ref = vec[layers[0]][0]
    model, tok = load_host(cfg, args.device)
    reps = {K: PositionReplacer(model, K) for K in layers}

    def targets(K: int, sid: str, kind: str) -> dict:
        by, arr = vec[K]
        tg = {}
        for r in by[sid]:
            v = torch.from_numpy(arr[VEC_KEY[kind]][int(r["vec"])])
            for q in r["positions"]:
                tg[q] = v
        return tg

    def arm_setup(arm: str, sid: str) -> int:
        """Install the write for one arm; returns the expected written-position count."""
        if arm in ("noop", "prompt"):
            for r in reps.values():
                r.set_targets(None)
            return 0
        kind, where = arm.split("_")
        Ks, beta = (BAND, BETA_BAND) if where == "band" else (SINGLE, BETA_SINGLE)
        n = 0
        for K, r in reps.items():
            t = targets(K, sid, kind) if K in Ks else None
            r.set_beta(beta); r.set_targets(t); n += len(t or {})
        return n

    @torch.no_grad()
    def gen(pids, n_seq: int, greedy: bool, expect: int) -> list[str]:
        for r in reps.values():
            r.reset()
        ids = torch.tensor([list(pids)], device=model.device)
        kw = dict(max_new_tokens=max_new, pad_token_id=tok.pad_token_id or 0)
        if greedy:
            o = model.generate(ids, do_sample=False, **kw)
        else:
            # batch 1 + num_return_sequences: HF expands along dim 0 with NO padding, so absolute
            # positions stay aligned for PositionReplacer. Left-padded batching would not.
            o = model.generate(ids, do_sample=True, temperature=args.temperature, top_p=args.top_p,
                               num_return_sequences=n_seq, **kw)
        # The counter is per-FORWARD, not per-sequence: span positions all live in the prompt, so only
        # the prefill forward writes them, and `num_return_sequences` expands the batch dim without
        # changing `local`. Decode steps sit past every target position and early-return untouched.
        n_w = sum(r.n_positions_written for r in reps.values())
        if n_w != expect:
            raise RuntimeError(f"wrote {n_w} positions, expected {expect}")
        return [tok.decode(o[i, len(pids):], skip_special_tokens=True) for i in range(o.shape[0])]

    # ── the prompting baseline, with a round-trip guard ────────────────────────────────────────
    def prompt_ids_for(sid: str, pids: list[int]) -> list[int] | None:
        """Re-templated prompt carrying the warning. None if decode->encode does not round-trip."""
        txt = tok.decode(pids, skip_special_tokens=False)
        back = tok(txt, add_special_tokens=False)["input_ids"]
        if list(back) != list(pids):
            return None                      # refuse rather than silently malform the prompt
        anchor = "Here is the function"
        new = (txt.replace(anchor, WARN + anchor, 1) if anchor in txt else
               txt.rstrip() + "\n\n" + WARN)
        return tok(new, add_special_tokens=False)["input_ids"]

    sids = [p["snippet_id"] for p in select_pairs(args, traces) if p["snippet_id"] in ref]
    if args.limit:
        sids = sids[:args.limit]
    print(f"{TAG} {len(sids)} items · arms {arms} · max_new_gen {max_new} · band {BAND} b={BETA_BAND} · single {SINGLE} "
          f"b={BETA_SINGLE} · n_samples {args.n_samples}", flush=True)

    t0 = time.time()
    rows: list[dict] = []
    n_prompt_skipped = 0
    for i, sid in enumerate(sids):
        tr = traces[sid]
        row = {"snippet_id": sid, "truth": tr["truth"],
               "l0_correct": bool(tr["l0_correct"]), "l1b_correct": bool(tr["l1b_correct"]),
               "flippable": bool(tr["l0_correct"] and not tr["l1b_correct"])}
        for arm in arms:
            pids = tr["l1b_prompt_ids"]
            if arm == "prompt":
                pp = prompt_ids_for(sid, pids)
                if pp is None:
                    n_prompt_skipped += 1
                    row["prompt_greedy_correct"] = None
                    continue
                pids = pp
            exp = arm_setup(arm, sid)
            g = gen(pids, 1, True, exp)[0]
            got, ok = graded(g, tr["truth"])
            row[f"{arm}_greedy_correct"] = bool(ok)
            row[f"{arm}_greedy_parsed"] = got is not None
            if not args.no_sampled:
                outs = gen(pids, args.n_samples, False, exp)
                oks = [graded(o, tr["truth"])[1] for o in outs]
                row[f"{arm}_rate"] = float(np.mean(oks))
                row[f"{arm}_parsed_rate"] = float(np.mean([graded(o, tr["truth"])[0] is not None
                                                           for o in outs]))
        rows.append(row)
        print(f"{TAG} {i+1}/{len(sids)} {sid} · "
              + " ".join(f"{a}={'1' if row.get(f'{a}_greedy_correct') else '0'}"
                         f"/{row.get(f'{a}_rate', float('nan')):.2f}" for a in arms)
              + f" · {(time.time()-t0)/60:.1f} min", flush=True)
        if time.time() - t0 > args.max_hours * 3600:
            print(f"{TAG} wall-clock stop after {len(rows)} items", flush=True); break
    for r in reps.values():
        r.close()
    (out / "accuracy_rows.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    # ── score: descriptive only, no verdict word ───────────────────────────────────────────────
    st: dict = {"experiment": "S9_accuracy_at_user_request", "n_items": len(rows), "seed": seed,
                "band": BAND, "beta_band": BETA_BAND, "single": SINGLE, "beta_single": BETA_SINGLE,
                "n_samples": args.n_samples, "n_prompt_skipped": n_prompt_skipped,
                "bound": ("acc_l0 0.567 -> acc_l1b 0.533; 2/60 net headroom, 7 flippable; a PERFECT "
                          "rescue of all 7 against the measured 6-9/60 same-condition churn gives "
                          "McNemar p = 0.092. No positive significance claim is possible at n=60."),
                "greedy": {}, "sampled": {}, "vs_noop": {}, "summary": [],
                # H-A8: the budget is a first-class part of the instrument, so it is recorded in the stats.
                "config": {"max_new_gen": max_new, "arms": list(arms), "greedy_only": bool(args.no_sampled)}}
    flip = [r for r in rows if r["flippable"]]
    st["n_flippable"] = len(flip)
    for arm in arms:
        gk = f"{arm}_greedy_correct"
        v = [r[gk] for r in rows if r.get(gk) is not None]
        if v:
            k, n = int(sum(v)), len(v)
            lo, hi = wilson(k, n)
            st["greedy"][arm] = {"acc": k / n, "k": k, "n": n, "wilson95": [lo, hi]}
            # parsed rate: an unparsed generation scores as WRONG, so this is the share of the accuracy mass
            # the arm could even reach. Aggregated since H-A8 (it was only ever a per-row field before).
            pv = [r[f"{arm}_greedy_parsed"] for r in rows if r.get(f"{arm}_greedy_parsed") is not None]
            if pv:
                pk, pn = int(sum(pv)), len(pv)
                plo, phi = wilson(pk, pn)
                st["greedy"][arm]["parsed"] = pk / pn
                st["greedy"][arm]["parsed_wilson95"] = [plo, phi]
                st["greedy"][arm]["acc_given_parsed"] = (
                    float(np.mean([r[gk] for r in rows if r.get(f"{arm}_greedy_parsed")])) if pk else float("nan"))
            fv = [r[gk] for r in flip if r.get(gk) is not None]
            if fv:
                st["greedy"][arm]["acc_flippable"] = float(np.mean(fv))
                st["greedy"][arm]["k_flippable"] = int(sum(fv))
        rk = f"{arm}_rate"
        rv = [r[rk] for r in rows if rk in r]
        if rv:
            st["sampled"][arm] = boot_mean(rv, seed + 1)
            fr = [r[rk] for r in flip if rk in r]
            if fr:
                st["sampled"][arm]["rate_flippable"] = boot_mean(fr, seed + 2)
    for arm in arms:
        if arm == "noop":
            continue
        gk, nk = f"{arm}_greedy_correct", "noop_greedy_correct"
        pair = [(r[gk], r[nk]) for r in rows if r.get(gk) is not None and r.get(nk) is not None]
        b = sum(1 for a, n_ in pair if a and not n_)      # arm fixed it
        c = sum(1 for a, n_ in pair if n_ and not a)      # arm broke it
        d = {"n_pairs": len(pair), "arm_fixed": b, "arm_broke": c, "mcnemar_p": mcnemar(b, c)}
        rk = f"{arm}_rate"
        if any(rk in r for r in rows):
            dd = [r[rk] - r["noop_rate"] for r in rows if rk in r and "noop_rate" in r]
            d["paired_rate_delta"] = boot_mean(dd, seed + 3)
        st["vs_noop"][arm] = d
    for arm in arms:
        g = st["greedy"].get(arm, {})
        s = st["sampled"].get(arm, {})
        v = st["vs_noop"].get(arm, {})
        st["summary"].append(
            f"{arm:<13} greedy {g.get('acc', float('nan')):.4f} "
            f"parsed {g.get('parsed', float('nan')):.4f} acc|parsed {g.get('acc_given_parsed', float('nan')):.4f} "
            f"[{g.get('wilson95', [float('nan')]*2)[0]:.3f},{g.get('wilson95', [float('nan')]*2)[1]:.3f}] "
            f"flip {g.get('acc_flippable', float('nan')):.3f} · rate {s.get('mean', float('nan')):.4f} "
            + (f"· vs noop fixed {v['arm_fixed']} broke {v['arm_broke']} p={v['mcnemar_p']:.3f} "
               f"dRate {v.get('paired_rate_delta', {}).get('mean', float('nan')):+.4f} "
               f"{v.get('paired_rate_delta', {}).get('ci95', [])}" if v else ""))
    (out / "accuracy_stats.json").write_text(json.dumps(st, indent=1))
    print(f"{TAG} wrote {out / 'accuracy_stats.json'}")
    print(f"{TAG} BOUND: {st['bound']}")
    for line in st["summary"]:
        print(f"{TAG} {line}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
