"""H-W13 / H-W14 — dose curve over patched spans, tier contrast, and a harness identity check.

Pre-registered: `log/nla-harness/2026-09-05_dose-and-tier-prereg.md`.

The null battery found the clean-state effect **item-specific but not span-specific** -- a sibling
span retains 87.4%. Two readings survive that, and they predict identical sibling-swap numbers:

  redundant     every span carries the SAME item-level signal, so one span restores most of it
  accumulative  each span carries a fraction and they sum, so swapping one of ~7.7 changes little

They differ in the DOSE CURVE, which is what `P_1 / P_2 / P_4 / P_all` measures.

k also sets how many positions are written, so dose is confounded with delivered perturbation --
and the battery showed how violent that dimension is (random replacement at ~25 positions costs
-365 nats). `F_1` and `F_all` are matched foreign-clean controls at the same k, so the quantity
that isolates content from operator at each dose is the CONTRAST `P_k - F_k`, never `P_k` alone.

`F_L1b_all` asks H-W14: a foreign item's OBFUSCATED span, against a foreign item's CLEAN span. If
they are equal, a foreign vector supplies generic program-ness regardless of tier and the measured
+15.18 floor says nothing about cleanliness.

SELF IS AN ASSERTION, NOT AN ARM. `PositionReplacer` writes `||h|| * unit(v)`, so writing a
position's own current activation is *exactly* the identity and must score 0. `|dG_SELF| <= 1.0`
nat is enforced; a violation means the write path, the position mapping or the scorer is faulty,
and the run is REFUSED rather than reported. It tests the whole pipeline end to end at every
written position, which no W run has had.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import zlib
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

ARMS = ("P_patch", "N_foreign_clean", "N_shuffled_clean", "N_random")
GAP_BAR = SUPPORT_NATS       # +12.11 nats, the frozen support threshold, reused as the gap bar


ARMS = ("SELF", "P_1", "P_2", "P_4", "P_all", "F_1", "F_all", "F_L1b_all")
GATED = ("P_1", "P_all", "F_1", "F_all", "F_L1b_all")
GAP_BAR = SUPPORT_NATS
SELF_TOL = 1.0               # nats; the identity assertion
DOSE_FRAC = 0.50             # H-W13a bar


def score(rows: list[dict], baseline: dict | None, out_p: Path) -> dict:
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, len(rows), size=(N_BOOT, len(rows))) if rows else None

    def col(a): return np.array([r[f"dG_{a}"] for r in rows], dtype=float)

    def boot(v):
        d = v[draws].mean(1)
        return float(v.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))

    per_arm = {}
    for a in ARMS:
        v = col(a); m, lo, hi = boot(v)
        acc = [r.get(f"acc_{a}") for r in rows if r.get(f"acc_{a}") is not None]
        par = [r.get(f"parse_{a}") for r in rows if r.get(f"parse_{a}") is not None]
        e = {"dG_sum_mean": m, "ci95": [lo, hi], "n": len(v),
             "acc": float(np.mean(acc)) if acc else None,
             "parse": float(np.mean(par)) if par else None}
        # An arm with no generation cannot clear the veto and therefore cannot carry a verdict --
        # the clause that made stage 1 refuse a verdict when no baseline existed.
        e["veto"] = (bool((baseline["acc"] - e["acc"]) > VETO_DROP
                          or (baseline["parse"] - e["parse"]) > VETO_DROP)
                     if baseline and e["acc"] is not None else None)
        per_arm[a] = e

    self_m = per_arm["SELF"]["dG_sum_mean"]
    self_ok = bool(abs(self_m) <= SELF_TOL)

    p1, pall = col("P_1"), col("P_all")
    f1, fall, fl1b = col("F_1"), col("F_all"), col("F_L1b_all")
    ratio = float(p1.mean() / pall.mean()) if abs(pall.mean()) > 1e-9 else None
    rb = p1[draws].mean(1) / np.where(np.abs(pall[draws].mean(1)) < 1e-9, np.nan,
                                      pall[draws].mean(1))
    rb = rb[np.isfinite(rb)]
    ratio_ci = [float(np.percentile(rb, 2.5)), float(np.percentile(rb, 97.5))] if rb.size else None

    contrasts = {}
    for name, v in (("content_at_k1", p1 - f1), ("content_at_kall", pall - fall),
                    ("H_W14_tier", fall - fl1b)):
        m, lo, hi = boot(v)
        contrasts[name] = {"mean": m, "ci95": [lo, hi],
                           "clears": bool(m >= GAP_BAR and lo > 0.0),
                           "positive_on": int((v > 0).sum()), "n": len(v)}

    curve = {k: per_arm[f"P_{k}"]["dG_sum_mean"] for k in ("1", "2", "4", "all")}
    monotone = bool(curve["1"] <= curve["2"] <= curve["4"] <= curve["all"])
    redundant = bool(ratio is not None and ratio_ci and ratio_ci[0] >= DOSE_FRAC)
    accumulative = bool(ratio is not None and ratio_ci and ratio_ci[1] < DOSE_FRAC and monotone)
    tier = contrasts["H_W14_tier"]["clears"]

    if not self_ok:
        verdict = "W13-HARNESS-FAULT"          # checked first; nothing else is readable if it fires
    elif redundant:
        verdict = "W13-REDUNDANT-TIER-MATTERS" if tier else "W13-REDUNDANT-TIER-GENERIC"
    elif accumulative:
        verdict = "W13-ACCUMULATIVE-TIER-MATTERS" if tier else "W13-ACCUMULATIVE-TIER-GENERIC"
    else:
        verdict = "W13-INDETERMINATE-DOSE"

    stats = {"experiment": "W13_dose_W14_tier", "seed": SEED, "n_boot": N_BOOT,
             "n_items": len(rows),
             "SELF_identity": {"mean": self_m, "tol": SELF_TOL, "passes": self_ok},
             "dose_curve": curve, "monotone": monotone,
             "P1_over_Pall": {"point": ratio, "ci95": ratio_ci, "bar": DOSE_FRAC},
             "contrasts": contrasts, "baseline_behav": baseline, "per_arm": per_arm,
             "H_W13a_redundant": redundant, "H_W13b_accumulative": accumulative,
             "H_W14_tier_matters": tier, "verdict": verdict}
    json.dump(stats, open(out_p, "w"), indent=2)
    return stats


GEN_ARMS = ("P_1", "F_L1b_all")     # veto generations; P_all is banked (job 378019, veto pass)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma12b", choices=sorted(HOSTS))
    ap.add_argument("--allow-banked-host", action="store_true")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-spans-per-item", type=int, default=8)
    ap.add_argument("--traces", default=None)
    ap.add_argument("--no-generate", action="store_true")
    ap.add_argument("--score-only", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--max-hours", type=float, default=8.0)
    args = ap.parse_args()

    if args.model == "qwen7b" and not args.allow_banked_host:
        print("[W13] REFUSED: qwen7b is a Chinese model; pass --allow-banked-host.")
        return 2
    model_name, layer = HOSTS[args.model]
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    if args.score_only:
        rp = out / "dose_rows.jsonl"
        if not rp.exists():
            print(f"[W13] REFUSED: no banked rows at {rp}"); return 2
        rows = [json.loads(l) for l in open(rp)]
        prev = out / "dose_stats.json"
        base = json.load(open(prev)).get("baseline_behav") if prev.exists() else None
        st = score(rows, base, prev)
        print(json.dumps(st, indent=2))
        return 0 if st["SELF_identity"]["passes"] else 3

    t0 = time.time()
    trc = Path(args.traces or _PROJ / f"data/nla/p0/trace_llr/{args.model}/traces.jsonl")
    if not trc.exists():
        print(f"[W13] REFUSED: missing prerequisite {trc}"); return 2
    traces = {t["snippet_id"]: t for t in map(json.loads, open(trc))}

    ex = ActivationExtractor(model_name, layer, device=args.device)
    model, tokz = ex.model, ex.tokenizer
    rep = PositionReplacer(model, layer)

    pairs = load_pairs(args.limit or (4 if args.smoke else None), random.Random(SEED))
    if args.smoke:
        pairs = pairs[:4]
    pairs = [p for p in pairs if p["snippet_id"] in traces]

    def logp(pids, rids) -> float:
        seq = torch.tensor([list(pids) + list(rids)], device=model.device)
        plen = len(pids); rep.reset()
        with torch.no_grad():
            lg = model(seq).logits[0, plen - 1:-1]
            tgt = seq[0, plen:]; tot = 0.0
            for st in range(0, lg.shape[0], 256):
                tot += float(torch.log_softmax(lg[st:st + 256].float(), -1)
                             .gather(1, tgt[st:st + 256, None])[:, 0].sum())
        return tot

    def gen(pids) -> str:
        rep.reset()
        with torch.no_grad():
            o = model.generate(torch.tensor([list(pids)], device=model.device),
                               max_new_tokens=MAX_NEW_GEN, do_sample=False,
                               pad_token_id=tokz.pad_token_id or 0)
        return tokz.decode(o[0, len(pids):], skip_special_tokens=True)

    # ── pass 1: pools. L0 clean states, L1b own states (for SELF and for the tier arm). ────────
    diag = {"locatable": 0, "unpaired": 0, "no_rename": 0, "no_span": 0}
    pool0: dict[str, dict[int, np.ndarray]] = {}      # clean (L0) vector per span
    pool1: dict[str, dict[int, np.ndarray]] = {}      # this item's own L1b vector per span
    posmap: dict[str, dict[int, list[int]]] = {}
    selfmap: dict[str, dict[int, np.ndarray]] = {}    # per-POSITION own activation, for SELF
    for p in pairs:
        if time.time() - t0 > args.max_hours * 3600:
            break
        rep.set_targets(None)
        sid = p["snippet_id"]
        user0 = build_user(p["code_l0"], p["call_l0"])
        user1 = build_user(p["code_l1b"], p["call_l1b"])
        inv = {v: k for k, v in (p["rename_map"] or {}).items()}
        spans = (p.get("id_spans_l1b") or [])[: args.max_spans_per_item]
        if not spans:
            continue
        full0 = ex.extract_chat(user0, positions=None, text_id=sid)
        full1 = ex.extract_chat(user1, positions=None, text_id=sid)
        v0, v1, pos_by, selfv, cache = {}, {}, {}, {}, {}
        for si, sp in enumerate(spans):
            decoy = p["code_l1b"][int(sp[0]):int(sp[1])].strip()
            true = inv.get(decoy)
            if not true:
                diag["no_rename"] += 1; continue
            occ = term_spans(p["code_l0"], true)
            if not occ:
                diag["unpaired"] += 1; continue
            pos, _ = span_token_positions(tokz, user1, p["code_l1b"], [sp])
            if not pos or max(pos) >= len(full1.activations):
                diag["no_span"] += 1; continue
            if true not in cache:
                pos0, _ = span_token_positions(tokz, user0, p["code_l0"], [occ[0]])
                if not pos0 or max(pos0) >= len(full0.activations):
                    diag["no_span"] += 1; continue
                cache[true] = full0.activations[max(pos0)].astype(np.float32)
            v0[si] = cache[true]
            v1[si] = full1.activations[max(pos)].astype(np.float32)
            pos_by[si] = pos
            # SELF is per POSITION, not per span: each written position gets back its own vector,
            # so the assertion covers the position mapping as well as the write arithmetic.
            for q in pos:
                selfv[q] = full1.activations[q].astype(np.float32)
            diag["locatable"] += 1
        if v0:
            pool0[sid], pool1[sid], posmap[sid], selfmap[sid] = v0, v1, pos_by, selfv
        print(f"[W13] pool {sid}: {len(v0)} spans", flush=True)

    sids = sorted(pool0)
    if len(sids) < 2:
        print("[W13] REFUSED: need >= 2 items with locatable spans"); return 2

    sink = open(out / "dose_rows.jsonl", "w")
    rows = []
    for pi, sid in enumerate(sids):
        if time.time() - t0 > args.max_hours * 3600:
            diag["wall_clock_stop"] = diag.get("wall_clock_stop", 0) + 1
            break
        tr = traces[sid]
        v0, v1, pos_by = pool0[sid], pool1[sid], posmap[sid]
        spans_here = sorted(v0)
        others = [q for q in sids if q != sid]
        # Seeded span ORDER, so P_1 subset P_2 subset P_4 subset P_all -- a nested ladder. Independent
        # draws per k would let the curve move because different spans were chosen rather than
        # because more were.
        order = sorted(spans_here, key=lambda si: zlib.crc32(f"{sid}#{si}".encode()))

        def foreign_vec(si: int, tier: int) -> np.ndarray:
            key = f"{sid}#{si}#f".encode()
            fs = others[zlib.crc32(key) % len(others)]
            src = pool0[fs] if tier == 0 else pool1[fs]
            fk = sorted(src)[zlib.crc32(key + b"k") % len(src)]
            return src[fk]

        targets = {a: {} for a in ARMS}
        targets["SELF"] = {q: torch.from_numpy(v) for q, v in selfmap[sid].items()}
        for k, name in ((1, "P_1"), (2, "P_2"), (4, "P_4"), (len(order), "P_all")):
            for si in order[:k]:
                for q in pos_by[si]:
                    targets[name][q] = torch.from_numpy(v0[si])
        for k, name in ((1, "F_1"), (len(order), "F_all")):
            for si in order[:k]:
                for q in pos_by[si]:
                    targets[name][q] = torch.from_numpy(foreign_vec(si, 0))
        for si in order:
            for q in pos_by[si]:
                targets["F_L1b_all"][q] = torch.from_numpy(foreign_vec(si, 1))

        pids, rids = tr["l1b_prompt_ids"], tr["l0_reply_ids"]
        rep.set_targets(None)
        base = logp(pids, rids)
        row = {"snippet_id": sid, "n_spans": len(spans_here),
               "n_positions_all": len(targets["P_all"]),
               "n_positions_1": len(targets["P_1"]), "logp_noop": base, "n_tok": len(rids)}
        for arm in ARMS:
            rep.set_targets(targets[arm])
            row[f"dG_{arm}"] = logp(pids, rids) - base
            if not args.no_generate and arm in GEN_ARMS:
                got, ok = graded(gen(pids), tr["truth"])
                row[f"acc_{arm}"], row[f"parse_{arm}"] = bool(ok), got is not None
        rep.set_targets(None)
        if not args.no_generate:
            got, ok = graded(gen(pids), tr["truth"])
            row["acc_noop"], row["parse_noop"] = bool(ok), got is not None
        rows.append(row); sink.write(json.dumps(row) + "\n"); sink.flush()
        print(f"[W13] {pi+1}/{len(sids)} {sid} SELF={row['dG_SELF']:+.3f} "
              f"P1={row['dG_P_1']:+.1f} P2={row['dG_P_2']:+.1f} P4={row['dG_P_4']:+.1f} "
              f"Pall={row['dG_P_all']:+.1f} F1={row['dG_F_1']:+.1f} "
              f"Fall={row['dG_F_all']:+.1f} FL1b={row['dG_F_L1b_all']:+.1f}", flush=True)
    sink.close()

    baseline = None
    if not args.no_generate:
        a = [r["acc_noop"] for r in rows if "acc_noop" in r]
        if a:
            baseline = {"acc": float(np.mean(a)),
                        "parse": float(np.mean([r["parse_noop"] for r in rows])),
                        "n": len(a), "source": "this run (unsteered)"}
    st = score(rows, baseline, out / "dose_stats.json")
    st["diag"] = diag
    json.dump(st, open(out / "dose_stats.json", "w"), indent=2)
    s_ = st["SELF_identity"]
    print(f"\n  SELF identity {s_['mean']:+.4f} nats (tol {s_['tol']}) -> "
          f"{'PASS' if s_['passes'] else 'FAIL'}", flush=True)
    for a, e in st["per_arm"].items():
        print(f"  {a:11s} {e['dG_sum_mean']:+8.2f} [{e['ci95'][0]:+.2f},{e['ci95'][1]:+.2f}] "
              f"veto={e['veto']}", flush=True)
    print(f"  dose curve {st['dose_curve']}  monotone={st['monotone']}")
    print(f"  P_1 / P_all = {st['P1_over_Pall']}")
    for n, c in st["contrasts"].items():
        print(f"  {n:16s} {c['mean']:+7.2f} [{c['ci95'][0]:+.2f},{c['ci95'][1]:+.2f}] "
              f"clears={c['clears']} ({c['positive_on']}/{c['n']})")
    print(f"[W13] VERDICT {st['verdict']}  diag={diag}", flush=True)
    if not s_["passes"]:
        print("[W13] REFUSED: SELF identity assertion failed — the write path, position mapping "
              "or scorer is faulty. Results are NOT reportable.", flush=True)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
