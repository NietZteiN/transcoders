"""H-W15 / H-W16 — what the item component is made of, and whether saturation is quantity or position.

Pre-registered: `log/nla-harness/2026-09-06_tier-source-prereg.md`.

The dose family decomposed the clean-state effect and left **56.2 % (+25.67 nats) as "it comes from
this item"**. All five obfuscation tiers exist for all 49 items, so an L1b span can be patched from
a DIFFERENT TIER OF THE SAME PROGRAM, which says what that 56 % is made of:

    T_L2   true identifiers, FLATTENED control flow   -> is it structure?
    T_L1   NONSENSE identifiers, original structure   -> is it meaning, or just the absence of a trap?

**L1 vs L1b is the distinction Papers 2-3 are built on**: L1 is merely uninformative, L1b is a
trap. So H-W16b asks whether the causal repair is *deleting the decoy* or *installing the truth* --
a question nothing in this programme has been able to pose until the tier ladder was used this way.

COVERAGE IS NOT EQUAL BETWEEN THE TWO, and the design reflects it rather than hiding it. L2
preserves identifiers so the true term is present verbatim at 375/375 spans. L1 anchors only
190/375 (50.7%): 131 through `meta.rename_map` and **59 through terms L1 never renamed**, which a
rename_map-only lookup would silently have dropped. The L1 arm therefore runs on its 190-position
subset **with a matched `T_L0_sub` arm at exactly those positions**; comparing a shrunk denominator
against the full-position anchor is the P0.3 failure mode.

H-W15 re-runs the dose ladder in REVERSE seeded order beside the forward one. The two select
different span subsets at each k, so if the yield depends only on *how many* spans are patched they
must agree; if they diverge, saturation is a property of *which* positions carry the signal.
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
from arm_guard import ArmNotWritten, arm_series, paired, record_positions  # noqa: E402
from repair_pairs import recover  # noqa: E402

ARMS_FULL = ("SELF", "T_L0_all", "T_L2_all", "T_L0_sub", "T_L1_sub",
             "P_1", "P_2", "P_4", "R_1", "R_2", "R_4")
# The ladder arms answer H-W15, which is settled (job 379819) and whose residual was explained
# offline on 2026-09-07 without GPU. They also cannot pass `arm_guard.paired`: the forward and
# reverse ladders select DIFFERENT spans at each k, spans differ in token length, so the two arms
# write different numbers of positions (equal on only 8/49 items at k=4). That mismatch is exactly
# what the guard exists to refuse, and refusing it here is correct rather than inconvenient.
ARMS_NO_LADDER = ("SELF", "T_L0_all", "T_L2_all", "T_L0_sub", "T_L1_sub")
ARMS = ARMS_FULL
GEN_ARMS = ("T_L2_all", "T_L1_sub")
GAP_BAR = SUPPORT_NATS
SELF_TOL = 1.0


def score(rows: list[dict], baseline: dict | None, out_p: Path) -> dict:
    rng = np.random.default_rng(SEED)
    def _draws(n):
        # Sized from the vector actually being resampled, not from len(rows): once the guard drops
        # unwritten items the two differ, and a fixed-size index would resample out of range.
        return rng.integers(0, n, size=(N_BOOT, n))

    guard_report = {}

    def col(a):
        """Values for `a` on items where it ACTUALLY WROTE. Bugs #8/#9: averaging the exact zeros
        an unwritten arm produces turned a real +21.82 into a sub-threshold +8.91, and turned an
        absent arm into a passing verdict."""
        v, rep = arm_series(rows, a, allow_exact_zero=(a == "SELF"))
        guard_report[a] = rep
        return np.array(v, dtype=float)

    def paircol(a, b):
        d, rep = paired(rows, a, b, allow_exact_zero=False)
        guard_report[f"{a}-{b}"] = rep
        return np.array(d, dtype=float)

    def boot(v):
        d = v[_draws(len(v))].mean(1)
        return float(v.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))

    per_arm, absent = {}, []
    for a in ARMS:
        try:
            v = col(a)
        except ArmNotWritten as e:
            absent.append(str(e)); continue
        m, lo, hi = boot(v)
        acc = [r.get(f"acc_{a}") for r in rows if r.get(f"acc_{a}") is not None]
        par = [r.get(f"parse_{a}") for r in rows if r.get(f"parse_{a}") is not None]
        e = {"dG_sum_mean": m, "ci95": [lo, hi], "n": len(v),
             "acc": float(np.mean(acc)) if acc else None,
             "parse": float(np.mean(par)) if par else None}
        e["veto"] = (bool((baseline["acc"] - e["acc"]) > VETO_DROP
                          or (baseline["parse"] - e["parse"]) > VETO_DROP)
                     if baseline and e["acc"] is not None else None)
        per_arm[a] = e

    if absent:
        stats = {"experiment": "W15_saturation_W16_tier_source", "verdict": "W16-ARM-ABSENT",
                 "absent_arms": absent, "guard": guard_report, "n_items": len(rows)}
        json.dump(stats, open(out_p, "w"), indent=2)
        return stats
    self_m = per_arm["SELF"]["dG_sum_mean"]
    self_ok = bool(abs(self_m) <= SELF_TOL)

    contrasts = {}
    for name, a, b in (("H_W16a_structure", "T_L0_all", "T_L2_all"),
                       ("H_W16b_meaning", "T_L0_sub", "T_L1_sub")):
        v = paircol(a, b); m, lo, hi = boot(v)
        contrasts[name] = {"minuend": a, "subtrahend": b, "mean": m, "ci95": [lo, hi],
                           "clears": bool(m >= GAP_BAR and lo > 0.0),
                           "positive_on": int((v > 0).sum()), "n": len(v)}
    # H-W15: forward vs reverse at each k. "Quantity" requires every |R_k - P_k| to be small AND
    # its CI to contain 0 -- a small mean with a CI excluding 0 is still a position effect.
    ladder = {}
    for k in ([] if not any(a.startswith("R_") for a in ARMS) else ("1", "2", "4")):
        v = paircol(f"R_{k}", f"P_{k}"); m, lo, hi = boot(v)
        ladder[k] = {"diff_mean": m, "ci95": [lo, hi],
                     "matches": bool(abs(m) < GAP_BAR and lo <= 0.0 <= hi)}
    quantity = bool(ladder) and bool(all(x["matches"] for x in ladder.values()))

    structure, meaning = contrasts["H_W16a_structure"]["clears"], contrasts["H_W16b_meaning"]["clears"]
    if not self_ok:
        verdict = "W16-HARNESS-FAULT"
    elif meaning and structure:
        verdict = "W16-MEANING-AND-STRUCTURE"
    elif meaning:
        verdict = "W16-MEANING-CARRIES-IT"
    elif structure:
        verdict = "W16-STRUCTURE-CARRIES-IT"
    else:
        verdict = "W16-NEITHER-REMOVAL-SUFFICES"   # nonsense ~ truth AND flattened ~ original:
                                                   # the repair is removing the decoy, not restoring
    stats = {"experiment": "W15_saturation_W16_tier_source", "seed": SEED, "n_boot": N_BOOT,
             "n_items": len(rows),
             "SELF_identity": {"mean": self_m, "tol": SELF_TOL, "passes": self_ok},
             "per_arm": per_arm, "contrasts": contrasts,
             "H_W15_ladder": ladder, "H_W15_quantity_not_position": quantity,
             "H_W16a_structure_matters": structure, "H_W16b_meaning_matters": meaning,
             "baseline_behav": baseline, "guard": guard_report, "verdict": verdict}
    json.dump(stats, open(out_p, "w"), indent=2)
    return stats


def tier_rows() -> dict:
    """snippet_id -> {tier: row} across every stimulus file."""
    out: dict = {}
    for ds in ("dataset_a", "dataset_b"):
        p = _PROJ / "data" / "stimuli" / ds / f"{ds}.jsonl"
        if p.exists():
            for r in map(json.loads, open(p)):
                out.setdefault(r["snippet_id"], {})[r["tier"]] = r
    return out


def tier_anchor(tier_row: dict, true_term: str, tier: str,
                strict: bool = False) -> str | None:
    """The term that stands for `true_term` in this tier's code, or None.

    L2 keeps identifiers, so the true term appears verbatim. L1 renames to nonsense, so it must be
    mapped through that tier's own rename_map -- EXCEPT where L1 left the identifier alone, which
    is 59 of 375 spans and would be silently dropped by a rename_map-only lookup.
    """
    m = (tier_row.get("meta") or {}).get("rename_map") or {}
    cand = m.get(true_term)
    if cand and term_spans(tier_row["code"], cand):
        return cand
    if strict:
        # H-W17: the fallback below lifts coverage by admitting spans that CANNOT express the
        # nonsense-vs-truth contrast -- an unrenamed span's L1 anchor is the same identifier as its
        # L0 anchor, so the two arms are identical there by construction and contribute an exact
        # zero to the gap. That diluted H-W16b's pooled estimate with 29 items of structural nulls.
        return None
    if term_spans(tier_row["code"], true_term):
        return true_term
    return None


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
    ap.add_argument("--no-ladder", action="store_true",
                    help="drop the P_k/R_k dose-ladder arms. H-W15 is settled and the two ladders "
                         "write different position counts, which arm_guard.paired correctly "
                         "refuses; keeping them would block the run for no new information.")
    ap.add_argument("--repair", action="store_true",
                    help="H-W23: fill anchors the stimulus pipeline lost, using repair_pairs. The "
                         "recorded rename_map still takes precedence wherever it has a real key; "
                         "recovery is used only where a '?unpaired' sentinel was written. Lifts "
                         "coverage 375->471 (L0) and 131->471 (L1), JavaScript 7->240.")
    ap.add_argument("--strict-anchor", action="store_true",
                    help="H-W17: require a rename_map entry, so only spans a tier genuinely "
                         "RENAMED qualify. Drops coverage 190 -> 131 spans and 49 -> 20 items, "
                         "and is the only way the L0-vs-L1 contrast means anything.")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--max-hours", type=float, default=8.0)
    args = ap.parse_args()

    if args.model == "qwen7b" and not args.allow_banked_host:
        print("[W16] REFUSED: qwen7b is a Chinese model; pass --allow-banked-host.")
        return 2
    global ARMS
    if args.no_ladder:
        ARMS = ARMS_NO_LADDER
    model_name, layer = HOSTS[args.model]
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    if args.score_only:
        rp = out / "tiers_rows.jsonl"
        if not rp.exists():
            print(f"[W16] REFUSED: no banked rows at {rp}"); return 2
        rows = [json.loads(l) for l in open(rp)]
        prev = out / "tiers_stats.json"
        base = json.load(open(prev)).get("baseline_behav") if prev.exists() else None
        st = score(rows, base, prev); print(json.dumps(st, indent=2))
        return 0 if st["SELF_identity"]["passes"] else 3

    t0 = time.time()
    trc = Path(args.traces or _PROJ / f"data/nla/p0/trace_llr/{args.model}/traces.jsonl")
    if not trc.exists():
        print(f"[W16] REFUSED: missing prerequisite {trc}"); return 2
    traces = {t["snippet_id"]: t for t in map(json.loads, open(trc))}
    tiers = tier_rows()

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

    sink = open(out / "tiers_rows.jsonl", "w")
    rows, diag = [], {"spans": 0, "L2_ok": 0, "L1_ok": 0, "no_L0_anchor": 0}
    for pi, p in enumerate(pairs):
        if time.time() - t0 > args.max_hours * 3600:
            diag["wall_clock_stop"] = diag.get("wall_clock_stop", 0) + 1
            break
        rep.set_targets(None)
        sid = p["snippet_id"]
        if sid not in traces or "L1" not in tiers.get(sid, {}):
            continue
        tr = traces[sid]
        user1 = build_user(p["code_l1b"], p["call_l1b"])
        inv = {v: k for k, v in (p["rename_map"] or {}).items()}
        lang = p.get("language") or "python"
        # Recovered {original: renamed}; inverted for the decoy->original direction. Computed once
        # per item and consulted ONLY where the recorded map wrote a '?unpaired' sentinel.
        rec_inv: dict[str, str] = {}
        rec_tier: dict[str, dict[str, str]] = {}
        if args.repair:
            rec_inv = {v: k for k, v in
                       recover(p["code_l0"], p["code_l1b"], lang)[0].items()}
        spans = (p.get("id_spans_l1b") or [])[: args.max_spans_per_item]
        if not spans:
            continue

        # One forward per tier, reused across every span of this item.
        acts, ok = {}, True
        for t in ("L0", "L1", "L2"):
            row_t = tiers[sid].get(t)
            if row_t is None:
                ok = False; break
            ut = build_user(row_t["code"], p["call_l0"] if t == "L0" else p["call_l0"])
            acts[t] = (ex.extract_chat(ut, positions=None, text_id=f"{sid}:{t}"), row_t, ut)
        if not ok:
            continue
        full1 = ex.extract_chat(user1, positions=None, text_id=sid)

        per_span = {}          # si -> {"pos":[...], "L0":vec, "L2":vec, "L1":vec|None}
        selfv = {}
        for si, sp in enumerate(spans):
            diag["spans"] += 1
            decoy = p["code_l1b"][int(sp[0]):int(sp[1])].strip()
            true = inv.get(decoy)
            if (not true or str(true).startswith("?unpaired")) and args.repair:
                true = rec_inv.get(decoy)
            if not true or str(true).startswith("?unpaired"):
                continue
            pos, _ = span_token_positions(tokz, user1, p["code_l1b"], [sp])
            if not pos or max(pos) >= len(full1.activations):
                continue
            got = {}
            for t in ("L0", "L1", "L2"):
                fa, row_t, ut = acts[t]
                if t == "L0":
                    term = true
                else:
                    term = tier_anchor(row_t, true, t, strict=args.strict_anchor)
                    if term is None and args.repair:
                        if t not in rec_tier:
                            rec_tier[t] = recover(p["code_l0"], row_t["code"], lang)[0]
                        term = rec_tier[t].get(true)
                if term is None:
                    continue
                occ = term_spans(row_t["code"], term)
                if not occ:
                    continue
                pt, _ = span_token_positions(tokz, ut, row_t["code"], [occ[0]])
                if pt and max(pt) < len(fa.activations):
                    got[t] = fa.activations[max(pt)].astype(np.float32)
            if "L0" not in got:
                diag["no_L0_anchor"] += 1
                continue
            if "L2" in got: diag["L2_ok"] += 1
            if "L1" in got: diag["L1_ok"] += 1
            per_span[si] = {"pos": pos, **got}
            for q in pos:
                selfv[q] = full1.activations[q].astype(np.float32)

        if not per_span:
            continue
        order = sorted(per_span, key=lambda si: zlib.crc32(f"{sid}#{si}".encode()))
        sub = [si for si in order if "L1" in per_span[si]]      # the matched 190-position subset

        targets = {a: {} for a in ARMS}
        targets["SELF"] = {q: torch.from_numpy(v) for q, v in selfv.items()}
        for si in order:                                        # full-position arms
            for q in per_span[si]["pos"]:
                targets["T_L0_all"][q] = torch.from_numpy(per_span[si]["L0"])
                # H-W24: this arm is CONDITIONAL, so it can hold fewer positions than T_L0_all
                # when a span has an L0 anchor but no L2 one. In job 379819 the counts happened to
                # match (L2 anchored all 375 L0-anchored spans), which is why H-W16a's +0.66 is
                # sound -- but that was luck of coverage, not construction. record_positions now
                # stamps both counts and arm_guard.paired refuses the contrast if they diverge.
                if "L2" in per_span[si]:
                    targets["T_L2_all"][q] = torch.from_numpy(per_span[si]["L2"])
        for si in sub:                                          # matched L0/L1 subset
            for q in per_span[si]["pos"]:
                targets["T_L0_sub"][q] = torch.from_numpy(per_span[si]["L0"])
                targets["T_L1_sub"][q] = torch.from_numpy(per_span[si]["L1"])
        if not args.no_ladder:
            rev = order[::-1]
            for k in (1, 2, 4):
                for name, seq_ in ((f"P_{k}", order[:k]), (f"R_{k}", rev[:k])):
                    for si in seq_:
                        for q in per_span[si]["pos"]:
                            targets[name][q] = torch.from_numpy(per_span[si]["L0"])

        pids, rids = tr["l1b_prompt_ids"], tr["l0_reply_ids"]
        rep.set_targets(None)
        base = logp(pids, rids)
        row = {"snippet_id": sid, "n_spans": len(per_span), "n_sub": len(sub),
               "n_positions_all": len(targets["T_L0_all"]),
               "n_positions_sub": len(targets["T_L0_sub"]),
               "logp_noop": base, "n_tok": len(rids)}
        for arm in ARMS:
            rep.set_targets(targets[arm])
            row[f"dG_{arm}"] = logp(pids, rids) - base
            if not args.no_generate and arm in GEN_ARMS:
                got_, ok_ = graded(gen(pids), tr["truth"])
                row[f"acc_{arm}"], row[f"parse_{arm}"] = bool(ok_), got_ is not None
        rep.set_targets(None)
        if not args.no_generate:
            got_, ok_ = graded(gen(pids), tr["truth"])
            row["acc_noop"], row["parse_noop"] = bool(ok_), got_ is not None
        record_positions(row, targets)
        rows.append(row); sink.write(json.dumps(row) + "\n"); sink.flush()
        print(f"[W16] {pi+1}/{len(pairs)} {sid} SELF={row['dG_SELF']:+.3f} "
              f"L0={row['dG_T_L0_all']:+.1f} L2={row['dG_T_L2_all']:+.1f} | "
              f"sub L0={row['dG_T_L0_sub']:+.1f} L1={row['dG_T_L1_sub']:+.1f} "
              f"({len(sub)}/{len(per_span)} spans)", flush=True)
    sink.close()

    baseline = None
    if not args.no_generate:
        a = [r["acc_noop"] for r in rows if "acc_noop" in r]
        if a:
            baseline = {"acc": float(np.mean(a)),
                        "parse": float(np.mean([r["parse_noop"] for r in rows])),
                        "n": len(a), "source": "this run (unsteered)"}
    st = score(rows, baseline, out / "tiers_stats.json")
    st["diag"] = diag
    json.dump(st, open(out / "tiers_stats.json", "w"), indent=2)
    s_ = st["SELF_identity"]
    print(f"\n  SELF identity {s_['mean']:+.4f} (tol {s_['tol']}) -> "
          f"{'PASS' if s_['passes'] else 'FAIL'}")
    for a, e in st["per_arm"].items():
        print(f"  {a:11s} {e['dG_sum_mean']:+8.2f} [{e['ci95'][0]:+.2f},{e['ci95'][1]:+.2f}] "
              f"veto={e['veto']}")
    for n, c in st["contrasts"].items():
        print(f"  {n:18s} {c['minuend']} - {c['subtrahend']} = {c['mean']:+7.2f} "
              f"[{c['ci95'][0]:+.2f},{c['ci95'][1]:+.2f}] clears={c['clears']} "
              f"({c['positive_on']}/{c['n']})")
    print(f"  H-W15 ladder {st['H_W15_ladder']}  quantity_not_position="
          f"{st['H_W15_quantity_not_position']}")
    print(f"[W16] VERDICT {st['verdict']}  diag={diag}", flush=True)
    if not s_["passes"]:
        print("[W16] REFUSED: SELF identity failed — results are NOT reportable.", flush=True)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
