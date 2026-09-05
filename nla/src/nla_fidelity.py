"""H-W7 — how much of an activation's CAUSAL power survives the round trip through language?

Pre-registered: `log/nla-harness/2026-09-04_nla-fidelity-prereg.md`.

Stage 1 established that the NLA channel delivers (`C3_ceiling` = +37.91 nats, 31.3% of a full
prompt swap, above prompting). What it could not say is how much is *lost* on the way, because it
had no un-verbalized comparison. This script supplies one:

    NLA causal fidelity  =  C3pure / P_patch

    C3pure  AR(AV(h_L0))   the clean activation, described in English, reconstructed, written back
    P_patch h_L0           the clean activation's own direction, written back directly
    C1r     AR(read_L1b)   the round-trip null on the SAME positions (banked stage-0 reads)

The released pair is scored on `MSE = 2(1-cos)` -- reconstruction fidelity. Nothing in the NLA
literature scores *causal* fidelity, and it is a different question: a direction can be recovered
to cos 0.9 and still carry a different effect on behaviour.

BOTH CEILINGS ARE DIRECTION-ONLY, and that is the design's load-bearing choice.
`PositionReplacer` writes `||h|| * unit(v)`, so `P_patch` supplies the clean direction at the L1b
position's own magnitude. Since the AR is trained on 2(1-cos) its output norm carries no
information at all, so a full-magnitude patch would beat it on a dimension the AR cannot even
represent -- the ratio would then measure the handicap rather than the fidelity.

POSITIONS. Only the 375 of 476 annotated spans whose decoy maps to a true term that actually
occurs in the L0 code. The other 101 are excluded by construction, not by preference: 88 map to
`?unpairedN` sentinels (the renaming INVENTED those identifiers -- there is no clean state to
patch) and 13 are builtins or one-character loop variables that were never renamed. Every arm
writes at exactly the same positions.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_NLA_ROOT = _HERE.parent
_PROJ = _NLA_ROOT.parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_NLA_ROOT / "vendor" / "nla-repo"))

from extract import ActivationExtractor  # noqa: E402
from span_positions import span_token_positions  # noqa: E402
from steer import PositionReplacer  # noqa: E402
from steer_run import HOSTS, AR_CHECKPOINTS, build_user, graded, load_pairs, SEED  # noqa: E402
from nla_cycle import AV_CHECKPOINTS, MAX_NEW_READ  # noqa: E402
from nla_writeback import MAX_NEW_GEN, SUPPORT_NATS, VETO_DROP, N_BOOT, term_spans  # noqa: E402

ARMS = ("C3pure", "P_patch", "C1r")
P_PROMPT_BANKED = 27.53      # banked prompting arm in G_sum nats, same 60 items (R2)
FIDELITY_BAR = 0.50          # frozen in the prereg


def score(rows: list[dict], baseline: dict | None, out_p: Path) -> dict:
    rng = np.random.default_rng(SEED)
    idx = np.arange(len(rows))

    def col(a: str) -> np.ndarray:
        return np.array([r[f"dG_{a}"] for r in rows], dtype=float)

    draws = rng.integers(0, len(rows), size=(N_BOOT, len(rows))) if rows else None

    def boot_mean(v: np.ndarray) -> tuple[float, float, float]:
        d = v[draws].mean(1)
        return float(v.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))

    per_arm = {}
    for a in ARMS:
        v = col(a)
        m, lo, hi = boot_mean(v)
        acc = [r.get(f"acc_{a}") for r in rows if r.get(f"acc_{a}") is not None]
        par = [r.get(f"parse_{a}") for r in rows if r.get(f"parse_{a}") is not None]
        e = {"dG_sum_mean": m, "ci95": [lo, hi], "n": len(v),
             "pct_of_prompt_swap": 100 * m / 121.127,
             "acc": float(np.mean(acc)) if acc else None,
             "parse": float(np.mean(par)) if par else None}
        if baseline and e["acc"] is not None:
            e["veto"] = bool((baseline["acc"] - e["acc"]) > VETO_DROP
                             or (baseline["parse"] - e["parse"]) > VETO_DROP)
        else:
            e["veto"] = None
        per_arm[a] = e

    c3, pp, c1 = col("C3pure"), col("P_patch"), col("C1r")
    # Ratio CI by resampling ITEMS and recomputing BOTH means on the same resample -- dividing two
    # independently bootstrapped means would throw away the pairing and mis-state the interval.
    rb = c3[draws].mean(1) / np.where(np.abs(pp[draws].mean(1)) < 1e-9, np.nan, pp[draws].mean(1))
    rb = rb[np.isfinite(rb)]
    ratio = {"point": float(c3.mean() / pp.mean()) if abs(pp.mean()) > 1e-9 else None,
             "ci95": [float(np.percentile(rb, 2.5)), float(np.percentile(rb, 97.5))] if rb.size else None}

    dm, dlo, dhi = boot_mean(c3 - c1)
    w7a = bool(per_arm["C3pure"]["ci95"][0] > P_PROMPT_BANKED)
    w7b_hi = bool(ratio["ci95"] and ratio["ci95"][0] >= FIDELITY_BAR)
    w7b_lo = bool(ratio["ci95"] and ratio["ci95"][1] < FIDELITY_BAR)
    w7c = bool(dm >= SUPPORT_NATS and dlo > 0)

    if w7a and w7b_hi:
        verdict = "W7-CHANNEL-STRONG-AND-FAITHFUL"
    elif w7a and w7b_lo:
        verdict = "W7-CHANNEL-STRONG-LANGUAGE-LOSSY"
    elif (not w7a) and w7b_hi:
        verdict = "W7-FAITHFUL-BUT-WEAK-SITE"
    elif (not w7a) and w7b_lo:
        verdict = "W7-RETRACT-CHANNEL-READING"
    else:
        verdict = "W7-INDETERMINATE-RATIO"

    stats = {"experiment": "W7_nla_causal_fidelity", "seed": SEED, "n_boot": N_BOOT,
             "n_items": len(rows),
             "n_positions": int(sum(r["n_positions"] for r in rows)),
             "baseline_behav": baseline, "per_arm": per_arm,
             "fidelity_ratio": ratio, "fidelity_bar": FIDELITY_BAR,
             "prompting_benchmark": P_PROMPT_BANKED,
             "C3pure_minus_C1r": {"mean": dm, "ci95": [dlo, dhi], "clears": w7c},
             "H_W7a_beats_prompting": w7a, "H_W7b_high_fidelity": w7b_hi,
             "H_W7b_low_fidelity": w7b_lo, "H_W7c_beats_roundtrip_null": w7c,
             "verdict": verdict}
    json.dump(stats, open(out_p, "w"), indent=2)
    return stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma12b", choices=sorted(HOSTS))
    ap.add_argument("--allow-banked-host", action="store_true")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-spans-per-item", type=int, default=8)
    ap.add_argument("--cycle-rows", default=None)
    ap.add_argument("--traces", default=None)
    ap.add_argument("--no-generate", action="store_true")
    ap.add_argument("--score-only", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--max-hours", type=float, default=8.0)
    args = ap.parse_args()

    if args.model == "qwen7b" and not args.allow_banked_host:
        print("[W7] REFUSED: qwen7b is a Chinese model; pass --allow-banked-host.")
        return 2
    model_name, layer = HOSTS[args.model]
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    if args.score_only:
        rp = out / "fidelity_rows.jsonl"
        if not rp.exists():
            print(f"[W7] REFUSED: no banked rows at {rp}"); return 2
        rows = [json.loads(l) for l in open(rp)]
        prev = out / "fidelity_stats.json"
        base = json.load(open(prev)).get("baseline_behav") if prev.exists() else None
        st = score(rows, base, prev)
        print(json.dumps(st, indent=2)); return 0

    t0 = time.time()
    cyc = Path(args.cycle_rows or _PROJ / f"data/nla/p0/nla_cycle/{args.model}/cycle_rows.jsonl")
    trc = Path(args.traces or _PROJ / f"data/nla/p0/trace_llr/{args.model}/traces.jsonl")
    for q in (cyc, trc):
        if not q.exists():
            print(f"[W7] REFUSED: missing prerequisite {q}"); return 2
    reads = {(r["snippet_id"], r["span_i"]): r["read"] for r in map(json.loads, open(cyc))}
    traces = {t["snippet_id"]: t for t in map(json.loads, open(trc))}

    from nla_inference import NLACritic  # noqa: E402
    from local_av import LocalAV  # noqa: E402

    ex = ActivationExtractor(model_name, layer, device=args.device)
    model, tokz = ex.model, ex.tokenizer
    av = LocalAV(AV_CHECKPOINTS[args.model], device=args.device)
    ar = NLACritic(AR_CHECKPOINTS[args.model], device=args.device)
    rep = PositionReplacer(model, layer)

    pairs = load_pairs(args.limit or (3 if args.smoke else None), random.Random(SEED))
    if args.smoke:
        pairs = pairs[:3]
    pairs = [p for p in pairs if p["snippet_id"] in traces]

    def logp(pids, rids) -> float:
        seq = torch.tensor([list(pids) + list(rids)], device=model.device)
        plen = len(pids); rep.reset()
        with torch.no_grad():
            lg = model(seq).logits[0, plen - 1:-1]
            tgt = seq[0, plen:]; tot = 0.0
            for s in range(0, lg.shape[0], 256):
                tot += float(torch.log_softmax(lg[s:s + 256].float(), -1)
                             .gather(1, tgt[s:s + 256, None])[:, 0].sum())
        return tot

    def gen(pids) -> str:
        rep.reset()
        with torch.no_grad():
            o = model.generate(torch.tensor([list(pids)], device=model.device),
                               max_new_tokens=MAX_NEW_GEN, do_sample=False,
                               pad_token_id=tokz.pad_token_id or 0)
        return tokz.decode(o[0, len(pids):], skip_special_tokens=True)

    sink = open(out / "fidelity_rows.jsonl", "w")
    rows, diag = [], {"locatable": 0, "unpaired": 0, "no_rename": 0, "no_read": 0}
    for pi, p in enumerate(pairs):
        if time.time() - t0 > args.max_hours * 3600:
            diag["wall_clock_stop"] = diag.get("wall_clock_stop", 0) + 1
            break
        rep.set_targets(None)                      # never extract under a stale write
        sid = p["snippet_id"]; tr = traces[sid]
        user1 = build_user(p["code_l1b"], p["call_l1b"])
        user0 = build_user(p["code_l0"], p["call_l0"])
        inv = {v: k for k, v in (p["rename_map"] or {}).items()}
        spans = (p.get("id_spans_l1b") or [])[: args.max_spans_per_item]
        if not spans:
            continue
        full0 = ex.extract_chat(user0, positions=None, text_id=sid)

        targets = {a: {} for a in ARMS}
        l0_cache: dict[str, np.ndarray] = {}
        for si, sp in enumerate(spans):
            decoy = p["code_l1b"][int(sp[0]):int(sp[1])].strip()
            true = inv.get(decoy)
            if not true:
                diag["no_rename"] += 1; continue
            occ = term_spans(p["code_l0"], true)
            if not occ:
                diag["unpaired"] += 1; continue     # `?unpairedN` — no clean state exists
            read = reads.get((sid, si))
            if read is None:
                diag["no_read"] += 1; continue
            pos, _ = span_token_positions(tokz, user1, p["code_l1b"], [sp])
            if not pos:
                continue
            if true not in l0_cache:
                pos0, _ = span_token_positions(tokz, user0, p["code_l0"], [occ[0]])
                if not pos0 or max(pos0) >= len(full0.activations):
                    continue
                l0_cache[true] = full0.activations[max(pos0)].astype(np.float32)
            h0 = l0_cache[true]
            diag["locatable"] += 1
            v_c3 = ar.reconstruct(av.generate(h0, temperature=0.0,
                                              max_new_tokens=MAX_NEW_READ)).numpy().astype(np.float32)
            v_c1 = ar.reconstruct(read).numpy().astype(np.float32)
            for arm, v in (("C3pure", v_c3), ("P_patch", h0), ("C1r", v_c1)):
                for q in pos:
                    targets[arm][q] = torch.from_numpy(np.asarray(v, dtype=np.float32))

        if not targets["C3pure"]:
            continue
        pids, rids = tr["l1b_prompt_ids"], tr["l0_reply_ids"]
        rep.set_targets(None)
        base = logp(pids, rids)
        row = {"snippet_id": sid, "n_positions": len(targets["C3pure"]),
               "logp_noop": base, "n_tok": len(rids)}
        for arm in ARMS:
            rep.set_targets(targets[arm])
            row[f"dG_{arm}"] = logp(pids, rids) - base
            if not args.no_generate:
                got, ok = graded(gen(pids), tr["truth"])
                row[f"acc_{arm}"], row[f"parse_{arm}"] = bool(ok), got is not None
        rep.set_targets(None)
        if not args.no_generate:
            got, ok = graded(gen(pids), tr["truth"])
            row["acc_noop"], row["parse_noop"] = bool(ok), got is not None
        rows.append(row); sink.write(json.dumps(row) + "\n"); sink.flush()
        print(f"[W7] {pi+1}/{len(pairs)} {sid} pos={row['n_positions']} "
              f"C3pure={row['dG_C3pure']:+.2f} P_patch={row['dG_P_patch']:+.2f} "
              f"C1r={row['dG_C1r']:+.2f}", flush=True)
    sink.close()

    baseline = None
    if not args.no_generate:
        a = [r["acc_noop"] for r in rows if "acc_noop" in r]
        if a:
            baseline = {"acc": float(np.mean(a)),
                        "parse": float(np.mean([r["parse_noop"] for r in rows])),
                        "n": len(a), "source": "this run (unsteered)"}
    st = score(rows, baseline, out / "fidelity_stats.json")
    st["diag"] = diag
    json.dump(st, open(out / "fidelity_stats.json", "w"), indent=2)
    for a, e in st["per_arm"].items():
        print(f"  {a:8s} {e['dG_sum_mean']:+8.2f} nats [{e['ci95'][0]:+.2f},{e['ci95'][1]:+.2f}]"
              f"  {e['pct_of_prompt_swap']:5.1f}% of swap  veto={e['veto']}", flush=True)
    print(f"  fidelity C3pure/P_patch = {st['fidelity_ratio']}", flush=True)
    print(f"[W7] VERDICT {st['verdict']}  diag={diag}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
