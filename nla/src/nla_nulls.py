"""Null battery — is the clean-state effect item-specific, span-specific, or just the operator?

Pre-registered: `log/nla-harness/2026-09-05_null-battery-prereg.md`.

H-W7 found that writing the clean activation's direction at the 375 locatable identifier spans
reaches 40% of a full prompt swap and beats prompting -- the programme's first arm to do so. What
that design could not rule out is that ANY clean-code direction, or any direction at all, does the
same thing at those positions. Residual streams are anisotropic (stage 0: every raw pairwise cosine
near 0.97), so a foreign clean activation is not obviously a different direction from the matched
one, and the worry is concrete rather than pedantic.

Four arms, identical positions, differing only in what is written:

    P_patch           h_L0 for THIS span            the anchor, re-run in-job so contrasts are paired
    N_foreign_clean   h_L0 from a DIFFERENT item     H-W9 -- item specificity
    N_shuffled_clean  h_L0 from another span, same item   H-W12 -- span specificity
    N_random          a seeded random unit vector    H-W6 -- the operator null

All are direction-only: `PositionReplacer` writes `||h|| * unit(v)`, so every arm delivers the same
magnitude at every position and the arms differ in direction alone. That is what makes the contrast
about content rather than about how hard each arm pushed.

Foreign and shuffled pairings and the random vectors are seeded functions of (snippet_id, span
index), so the arms are reproducible and cannot drift between runs.
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


def score(rows: list[dict], baseline: dict | None, out_p: Path) -> dict:
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, len(rows), size=(N_BOOT, len(rows))) if rows else None

    def col(a: str) -> np.ndarray:
        return np.array([r[f"dG_{a}"] for r in rows], dtype=float)

    def boot(v: np.ndarray) -> tuple[float, float, float]:
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
        e["veto"] = (bool((baseline["acc"] - e["acc"]) > VETO_DROP
                          or (baseline["parse"] - e["parse"]) > VETO_DROP)
                     if baseline and e["acc"] is not None else None)
        per_arm[a] = e

    # The three contrasts are PAIRED per item -- the arms share positions and the same clean trace,
    # so bootstrapping the difference is strictly tighter than comparing two marginal CIs.
    anchor = col("P_patch")
    gaps = {}
    for h, null in (("H_W9_item", "N_foreign_clean"),
                    ("H_W12_span", "N_shuffled_clean"),
                    ("H_W6_operator", "N_random")):
        d = anchor - col(null)
        m, lo, hi = boot(d)
        gaps[h] = {"null_arm": null, "gap_mean": m, "ci95": [lo, hi],
                   "clears": bool(m >= GAP_BAR and lo > 0.0),
                   "positive_on": int((d > 0).sum()), "n": len(d)}

    w9, w12, w6 = (gaps["H_W9_item"]["clears"], gaps["H_W12_span"]["clears"],
                   gaps["H_W6_operator"]["clears"])
    if not w6:
        verdict = "W9-OPERATOR-ONLY"          # checked FIRST: if content does not matter at all,
                                              # item- and span-specificity are not even askable.
    elif w9:
        verdict = "W9-ITEM-SPECIFIC"
    else:
        verdict = "W9-GENERIC-CLEAN"

    stats = {"experiment": "W9_null_battery", "seed": SEED, "n_boot": N_BOOT,
             "n_items": len(rows),
             "n_positions": int(sum(r["n_positions"] for r in rows)),
             "gap_bar": GAP_BAR, "baseline_behav": baseline,
             "per_arm": per_arm, "contrasts": gaps,
             "H_W9_item_specific": w9, "H_W12_span_specific": w12,
             "H_W6_beats_operator_null": w6, "verdict": verdict}
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
    ap.add_argument("--traces", default=None)
    ap.add_argument("--no-generate", action="store_true")
    ap.add_argument("--score-only", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--max-hours", type=float, default=8.0)
    args = ap.parse_args()

    if args.model == "qwen7b" and not args.allow_banked_host:
        print("[W9] REFUSED: qwen7b is a Chinese model; pass --allow-banked-host.")
        return 2
    model_name, layer = HOSTS[args.model]
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    if args.score_only:
        rp = out / "nulls_rows.jsonl"
        if not rp.exists():
            print(f"[W9] REFUSED: no banked rows at {rp}"); return 2
        rows = [json.loads(l) for l in open(rp)]
        prev = out / "nulls_stats.json"
        base = json.load(open(prev)).get("baseline_behav") if prev.exists() else None
        print(json.dumps(score(rows, base, prev), indent=2)); return 0

    t0 = time.time()
    trc = Path(args.traces or _PROJ / f"data/nla/p0/trace_llr/{args.model}/traces.jsonl")
    if not trc.exists():
        print(f"[W9] REFUSED: missing prerequisite {trc}"); return 2
    traces = {t["snippet_id"]: t for t in map(json.loads, open(trc))}

    # NO AV AND NO AR ARE LOADED. Every arm here is a raw activation or a random vector, so the
    # NLA models are not needed -- which frees ~48 GB and removes two failure modes from a run
    # whose only job is to decide whether H-W7's NLA result may be called item-specific.
    ex = ActivationExtractor(model_name, layer, device=args.device)
    model, tokz = ex.model, ex.tokenizer
    rep = PositionReplacer(model, layer)

    pairs = load_pairs(args.limit or (3 if args.smoke else None), random.Random(SEED))
    if args.smoke:
        pairs = pairs[:4]              # 4, so a foreign item always exists to draw from
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

    # ── pass 1: the pool of clean-state vectors ──────────────────────────────────────────────
    # Built before any writing so `N_foreign_clean` can draw from items processed later as well as
    # earlier. Drawing only from already-seen items would make the foreign arm depend on corpus
    # order -- a silent, seed-invisible dependence.
    diag = {"locatable": 0, "unpaired": 0, "no_rename": 0, "no_span": 0}
    pool: dict[str, dict[int, np.ndarray]] = {}
    posmap: dict[str, dict[int, list[int]]] = {}
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
        vecs, pos_by_span, cache = {}, {}, {}
        for si, sp in enumerate(spans):
            decoy = p["code_l1b"][int(sp[0]):int(sp[1])].strip()
            true = inv.get(decoy)
            if not true:
                diag["no_rename"] += 1; continue
            occ = term_spans(p["code_l0"], true)
            if not occ:
                diag["unpaired"] += 1; continue
            pos, _ = span_token_positions(tokz, user1, p["code_l1b"], [sp])
            if not pos:
                diag["no_span"] += 1; continue
            if true not in cache:
                pos0, _ = span_token_positions(tokz, user0, p["code_l0"], [occ[0]])
                if not pos0 or max(pos0) >= len(full0.activations):
                    diag["no_span"] += 1; continue
                cache[true] = full0.activations[max(pos0)].astype(np.float32)
            vecs[si] = cache[true]; pos_by_span[si] = pos
            diag["locatable"] += 1
        if vecs:
            pool[sid], posmap[sid] = vecs, pos_by_span
        print(f"[W9] pool {sid}: {len(vecs)} spans", flush=True)

    sids = sorted(pool)
    if len(sids) < 2:
        print("[W9] REFUSED: need >= 2 items with locatable spans for a foreign arm"); return 2
    d_model = ex.d_model

    # ── pass 2: write and score ──────────────────────────────────────────────────────────────
    sink = open(out / "nulls_rows.jsonl", "w")
    rows = []
    for pi, sid in enumerate(sids):
        if time.time() - t0 > args.max_hours * 3600:
            diag["wall_clock_stop"] = diag.get("wall_clock_stop", 0) + 1
            break
        tr = traces[sid]
        vecs, pos_by_span = pool[sid], posmap[sid]
        others = [q for q in sids if q != sid]
        targets = {a: {} for a in ARMS}
        for si, h0 in vecs.items():
            key = f"{sid}#{si}".encode()
            # Foreign: a different ITEM's clean span vector, chosen by a seeded hash of this span.
            fs = others[zlib.crc32(key) % len(others)]
            fk = sorted(pool[fs])[zlib.crc32(key + b"f") % len(pool[fs])]
            v_for = pool[fs][fk]
            # Shuffled: a different SPAN of THIS item where one exists; an item with a single
            # locatable span cannot support the arm and contributes its own vector, which makes
            # H-W12 conservative (the gap can only shrink) rather than undefined.
            sib = [k for k in vecs if k != si]
            v_shf = vecs[sib[zlib.crc32(key + b"s") % len(sib)]] if sib else h0
            g = torch.Generator().manual_seed(SEED + zlib.crc32(key + b"r"))
            v_rnd = torch.randn(d_model, generator=g).numpy().astype(np.float32)
            for arm, v in (("P_patch", h0), ("N_foreign_clean", v_for),
                           ("N_shuffled_clean", v_shf), ("N_random", v_rnd)):
                for q in pos_by_span[si]:
                    targets[arm][q] = torch.from_numpy(np.asarray(v, dtype=np.float32))

        pids, rids = tr["l1b_prompt_ids"], tr["l0_reply_ids"]
        rep.set_targets(None)
        base = logp(pids, rids)
        row = {"snippet_id": sid, "n_positions": len(targets["P_patch"]),
               "n_spans": len(vecs), "logp_noop": base, "n_tok": len(rids)}
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
        print(f"[W9] {pi+1}/{len(sids)} {sid} pos={row['n_positions']} "
              f"P={row['dG_P_patch']:+.1f} for={row['dG_N_foreign_clean']:+.1f} "
              f"shf={row['dG_N_shuffled_clean']:+.1f} rnd={row['dG_N_random']:+.1f}", flush=True)
    sink.close()

    baseline = None
    if not args.no_generate:
        a = [r["acc_noop"] for r in rows if "acc_noop" in r]
        if a:
            baseline = {"acc": float(np.mean(a)),
                        "parse": float(np.mean([r["parse_noop"] for r in rows])),
                        "n": len(a), "source": "this run (unsteered)"}
    st = score(rows, baseline, out / "nulls_stats.json")
    st["diag"] = diag
    json.dump(st, open(out / "nulls_stats.json", "w"), indent=2)
    for a, e in st["per_arm"].items():
        print(f"  {a:18s} {e['dG_sum_mean']:+8.2f} nats "
              f"[{e['ci95'][0]:+.2f},{e['ci95'][1]:+.2f}]  veto={e['veto']}", flush=True)
    for h, c in st["contrasts"].items():
        print(f"  {h:16s} P_patch - {c['null_arm']:17s} = {c['gap_mean']:+7.2f} "
              f"[{c['ci95'][0]:+.2f},{c['ci95'][1]:+.2f}]  clears={c['clears']} "
              f"({c['positive_on']}/{c['n']})", flush=True)
    print(f"[W9] VERDICT {st['verdict']}  diag={diag}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
