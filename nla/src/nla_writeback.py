"""Experiment W, stage 1 + H-W2′ — run the NLA's own protocol and see whether it steers.

Pre-registered: `log/nla-harness/2026-09-03_nla-writeback-prereg.md`, amended by
`2026-09-04_cycle-gate-amendment.md` (stage-0 gate) and `2026-09-04_stage2-vacuous-correction.md`
(H-W2 → H-W2′). Licensed to run by stage 0's `NLA-LIVE`.

THE PROTOCOL, as the NLA paper states it and as this project has never run it:

    read  ->  minimally edit the read  ->  reconstruct  ->  write back

Everything banked before this fed the AR a *template string* instead of a verbalized read
(`steer_run.py:480` defers the real variant as "a different question"), which is why `V1 ≡ V3`
with CI [0,0] was close to guaranteed by construction rather than informative.

Three properties of the released AR shape the operator, and none of them is a preference:
  * it emits a WHOLE STATE at one position, not a delta -> the native operation is REPLACE;
  * it is trained on MSE = 2(1-cos), so its output norm carries no information -> the LOCAL norm
    is kept (`steer.PositionReplacer`), which is what removes ALPHA from this design entirely;
  * R2 measured the only positions with a positive CI to be identifier spans, while `V5_replace`
    at `last_prompt` moved the clean half by +0.0014 of G -> the write goes to the SPANS.

ARMS. All four write at the same positions with the same operator, differing only in the text the
AR reconstructs. Equal positions matter: an arm that wrote at fewer positions would differ from
the others in delivered perturbation as well as in content, and the comparison would be void.

  W1_edit      AR(read with the decoy term replaced by the true term)   the protocol
  C1_roundtrip AR(read, unedited)                                       round-trip null
  C2_foreign   AR(read with a FOREIGN item's true term substituted)     same edit shape, wrong content
  C3_ceiling   AR(AV(h at the matched L0 span))                         this channel's own ceiling

C1 is the arm that decides whether stage 1 is interpretable at all: if merely passing a position
through AV->AR and writing it back moves G_sum, then every W1 number is round-trip artifact and
the family is UNINFORMATIVE rather than null.

A span whose read does not contain its decoy term CANNOT be edited. Those spans receive the
UNEDITED reconstruction in W1 and C2 -- identical to C1 there -- so all arms keep identical write
positions and W1 differs from C1 exactly at the spans where an edit was possible. The editable
count is reported, because a shrinking denominator looks identical to a null (the P0.3 lesson).

READOUT. `G_sum` in nats, per item: because every arm scores the SAME banked clean trace with the
same token count, the per-token normalisation cancels and the arm contrast is simply the
difference of log-prob sums. Support = +12.11 nats (10% of mean G_sum = 121.127). Accuracy is
secondary and underpowered by construction (6/60 flippable), and enters only as the veto.
"""
from __future__ import annotations

import argparse
import json
import random
import re
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
from steer_vectors import substitute_terms  # noqa: E402
from steer_run import HOSTS, AR_CHECKPOINTS, build_user, graded, load_pairs, SEED  # noqa: E402
from nla_cycle import AV_CHECKPOINTS, MAX_NEW_READ, cluster_boot  # noqa: E402

MAX_NEW_GEN = 1100          # matches every banked generation run
SUPPORT_NATS = 12.1127      # 10% of mean G_sum = 121.127, frozen in the prereg
VETO_DROP = 0.05            # R2's behavioural veto, inherited verbatim
N_BOOT = 10_000
ARMS = ("W1_edit", "C1_roundtrip", "C2_foreign", "C3_ceiling")
WORD = "[A-Za-z_][A-Za-z_0-9]*"


def term_spans(code: str, term: str) -> list[tuple[int, int]]:
    """Word-boundary occurrences of `term` in `code`.

    Word-boundary rather than substring: the L0 side routinely uses one- and two-character names
    (`f`, `ls`, `i`), and a substring search for `f` would match inside every identifier in the
    file and hand C3 a target built from arbitrary positions.
    """
    return [(m.start(), m.end())
            for m in re.finditer(rf"(?<![A-Za-z_0-9]){re.escape(term)}(?![A-Za-z_0-9])", code)]


def score(rows: list[dict], baseline: dict | None, out_p: Path) -> dict:
    """Apply the frozen rules. No GPU; re-runnable from banked rows."""
    rng = np.random.default_rng(SEED)

    def boot(vals: list[float]) -> tuple[float, float, float]:
        return cluster_boot([np.array([v]) for v in vals], rng, n=N_BOOT)

    per_arm: dict[str, dict] = {}
    for arm in ARMS:
        d = [r[f"dG_{arm}"] for r in rows if f"dG_{arm}" in r]
        if not d:
            continue
        m, lo, hi = boot(d)
        acc = [r[f"acc_{arm}"] for r in rows if f"acc_{arm}" in r]
        par = [r[f"parse_{arm}"] for r in rows if f"parse_{arm}" in r]
        per_arm[arm] = {
            "dG_sum_mean": m, "ci95": [lo, hi], "n": len(d),
            "frac_of_G": m / 121.127,
            "clears_support": bool(m >= SUPPORT_NATS and lo > 0.0),
            "acc": (float(np.mean(acc)) if acc else None),
            "parse": (float(np.mean(par)) if par else None),
        }

    # Behavioural veto, R2's clause verbatim. Its absence is what produced the S-LIVE artifact.
    base_acc = baseline.get("acc") if baseline else None
    base_par = baseline.get("parse") if baseline else None
    for arm, a in per_arm.items():
        if a["acc"] is None or base_acc is None:
            a["veto"] = None                      # not measured => cannot clear the veto
        else:
            a["veto"] = bool((base_acc - a["acc"]) > VETO_DROP
                             or (base_par - a["parse"]) > VETO_DROP)

    # ── H-W2': did the write reach the answer-production site, in the edit's direction? ──
    reg = [r["cos_dh_dedit"] for r in rows if r.get("cos_dh_dedit") is not None]
    reg_f = [r["cos_dh_dedit_foreign"] for r in rows if r.get("cos_dh_dedit_foreign") is not None]
    reg_c1 = [r["cos_dh_c1_dedit"] for r in rows if r.get("cos_dh_c1_dedit") is not None]

    def rate(v: list[float]) -> dict:
        if not v:
            return {"rate": None, "ci95": [None, None], "n": 0, "mean_cos": None}
        pos = [float(x > 0) for x in v]
        m, lo, hi = boot(pos)
        return {"rate": m, "ci95": [lo, hi], "n": len(v), "mean_cos": float(np.mean(v))}

    w2 = {"W1_vs_edit": rate(reg), "W1_vs_foreign_edit": rate(reg_f), "C1_vs_edit": rate(reg_c1)}
    w2["registers"] = bool(w2["W1_vs_edit"]["rate"] is not None
                           and w2["W1_vs_edit"]["ci95"][0] > 0.50)

    a = per_arm.get("W1_edit", {})
    c1 = per_arm.get("C1_roundtrip", {})
    c2 = per_arm.get("C2_foreign", {})
    # C1 decides interpretability BEFORE H-W1 is read: if the bare round trip moves G_sum, every
    # W1 number is round-trip artifact and this is UNINFORMATIVE, not a null.
    c1_contaminated = bool(c1 and c1.get("clears_support"))
    veto_missing = any(v.get("veto") is None for v in per_arm.values())
    w1_moves = bool(a.get("clears_support") and a.get("veto") is False
                    and not (c2.get("clears_support") and c2.get("dG_sum_mean", 0)
                             >= a.get("dG_sum_mean", 0)))

    if c1_contaminated:
        verdict = "W-UNINFORMATIVE-ROUNDTRIP"
    elif veto_missing:
        verdict = "W-VETO-MISSING"
    elif w1_moves and w2["registers"]:
        verdict = "W-STEERS"
    elif (not w1_moves) and w2["registers"]:
        verdict = "W-READOUT-NOT-MECHANISM"
    elif (not w1_moves) and not w2["registers"]:
        verdict = "W-UNINFORMATIVE-NO-DELIVERY"
    else:
        verdict = "W-GENERIC-PERTURBATION"

    stats = {
        "experiment": "W1_writeback", "seed": SEED, "n_boot": N_BOOT,
        "n_items": len(rows),
        "n_positions": int(sum(r["n_positions"] for r in rows)),
        "n_editable_spans": int(sum(r["n_editable"] for r in rows)),
        "support_nats": SUPPORT_NATS, "veto_drop": VETO_DROP,
        "baseline_behav": baseline,
        "per_arm": per_arm, "H_W2prime": w2,
        "verdict": verdict,
    }
    json.dump(stats, open(out_p, "w"), indent=2)
    return stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma12b", choices=sorted(HOSTS))
    ap.add_argument("--allow-banked-host", action="store_true")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-spans-per-item", type=int, default=8,
                    help="MUST match nla_cycle.py's cap or span_i indices no longer line up "
                         "with the banked reads")
    ap.add_argument("--cycle-rows", default=None, help="banked stage-0 reads (cycle_rows.jsonl)")
    ap.add_argument("--traces", default=None, help="banked clean traces from trace_llr.py")
    ap.add_argument("--no-generate", action="store_true",
                    help="skip the behavioural veto's generations (scoring only). The veto is "
                         "REQUIRED by the prereg, so a run with this flag cannot yield a verdict "
                         "and is marked VETO-MISSING.")
    ap.add_argument("--score-only", action="store_true",
                    help="re-apply the frozen rules to banked writeback_rows.jsonl. No GPU.")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--max-hours", type=float, default=5.0)
    args = ap.parse_args()

    if args.model == "qwen7b" and not args.allow_banked_host:
        print("[W1] REFUSED: qwen7b is a Chinese model; pass --allow-banked-host to override.")
        return 2

    model_name, layer = HOSTS[args.model]
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    if args.score_only:
        rp = out / "writeback_rows.jsonl"
        if not rp.exists():
            print(f"[W1] REFUSED: no banked rows at {rp}")
            return 2
        rows = [json.loads(l) for l in open(rp)]
        prev = out / "writeback_stats.json"
        base = json.load(open(prev)).get("baseline_behav") if prev.exists() else None
        st = score(rows, base, out / "writeback_stats.json")
        print(json.dumps({k: v for k, v in st.items() if k != "per_arm"}, indent=2))
        print(f"[W1] VERDICT {st['verdict']}")
        return 0
    t0 = time.time()
    cyc = Path(args.cycle_rows or _PROJ / f"data/nla/p0/nla_cycle/{args.model}/cycle_rows.jsonl")
    trc = Path(args.traces or _PROJ / f"data/nla/p0/trace_llr/{args.model}/traces.jsonl")
    for p in (cyc, trc):
        if not p.exists():
            print(f"[W1] REFUSED: missing prerequisite {p}")
            return 2

    reads = {(r["snippet_id"], r["span_i"]): r["read"]
             for r in map(json.loads, open(cyc))}
    traces = {t["snippet_id"]: t for t in map(json.loads, open(trc))}
    print(f"[W1] {len(reads)} banked reads, {len(traces)} traces", flush=True)

    from nla_inference import NLACritic  # noqa: E402
    from local_av import LocalAV  # noqa: E402

    # The AR load prints `model.norm.weight | MISSING ... newly initialized`, which looks alarming
    # and is not. The checkpoint is a truncated K+1 backbone (33 layers, 0..32, 430 tensors) and
    # genuinely omits the final norm; transformers initialises Gemma3RMSNorm's weight to ZEROS and
    # its forward is `out * (1.0 + weight)`, i.e. exactly unit scale. So the value is deterministic
    # and the same across runs -- checked 2026-09-04 rather than assumed, because a randomly
    # initialised tensor here would make every AR call irreproducible and silently invalidate
    # stage 0's cosines.
    ex = ActivationExtractor(model_name, layer, device=args.device)
    model, tokz = ex.model, ex.tokenizer
    av = LocalAV(AV_CHECKPOINTS[args.model], device=args.device)
    ar = NLACritic(AR_CHECKPOINTS[args.model], device=args.device)

    # Hook order is load-bearing. Forward hooks fire in registration order, so the replacer must
    # be registered BEFORE the read-back hook or H-W2' would capture the pre-write activation and
    # silently report that nothing propagated.
    rep = PositionReplacer(model, layer)
    grab: dict[str, torch.Tensor] = {}

    def _grab(_m, _i, output):
        h = output[0] if isinstance(output, tuple) else output
        grab["h"] = h.detach()
        return output
    ex._layers()[layer].register_forward_hook(_grab)

    pairs = load_pairs(args.limit or (3 if args.smoke else None), random.Random(SEED))
    if args.smoke:
        pairs = pairs[:3]
    pairs = [p for p in pairs if p["snippet_id"] in traces]
    rng = random.Random(SEED)

    def logp(prompt_ids, reply_ids) -> float:
        """sum_t log P(reply_t | prompt, reply_<t) under whatever the replacer currently holds."""
        seq = torch.tensor([list(prompt_ids) + list(reply_ids)], device=model.device)
        plen = len(prompt_ids)
        rep.reset()
        with torch.no_grad():
            logits = model(seq).logits[0, plen - 1:-1]
            tgt = seq[0, plen:]
            tot = 0.0
            for s in range(0, logits.shape[0], 256):
                lg = logits[s:s + 256].float()
                tot += float(torch.log_softmax(lg, -1).gather(1, tgt[s:s + 256, None])[:, 0].sum())
        return tot

    def gen(prompt_ids) -> str:
        rep.reset()
        with torch.no_grad():
            o = model.generate(torch.tensor([list(prompt_ids)], device=model.device),
                               max_new_tokens=MAX_NEW_GEN, do_sample=False,
                               pad_token_id=tokz.pad_token_id or 0)
        return tokz.decode(o[0, len(prompt_ids):], skip_special_tokens=True)

    # Rows are appended as they complete: a wall-clock stop or a crash must not throw away hours
    # of GPU. (B5 ran without per-case rows and its six cells can never be re-analysed.)
    sink = open(out / "writeback_rows.jsonl", "w")
    rows, diag = [], {"editable": 0, "spans": 0, "c3_missing": 0, "no_read": 0}
    for pi, p in enumerate(pairs):
        if time.time() - t0 > args.max_hours * 3600:
            diag["wall_clock_stop"] = diag.get("wall_clock_stop", 0) + 1
            break
        sid = p["snippet_id"]; tr = traces[sid]
        # Clear first. The replacer persists its targets across calls, so without this the L0
        # extraction below runs under the PREVIOUS item's write and C3's "clean state" is read
        # out of a steered forward pass -- a silent contamination of the ceiling arm.
        rep.set_targets(None)
        user1 = build_user(p["code_l1b"], p["call_l1b"])
        user0 = build_user(p["code_l0"], p["call_l0"])
        inv = {v: k for k, v in (p["rename_map"] or {}).items()}   # decoy -> true
        spans = (p.get("id_spans_l1b") or [])[: args.max_spans_per_item]

        # L0 span vectors for C3, keyed by TRUE term. The stimuli carry identifier_spans only on
        # the L1b side, so the clean-side locations are derived by word-boundary search. One
        # vector per term (its FIRST occurrence's last token) rather than per occurrence: the
        # decoy and true sides have different occurrence counts on most items, so a per-occurrence
        # pairing would silently mismatch -- the same trap `load_pairs` documents for zipping the
        # two term lists by index.
        l0_vec: dict[str, np.ndarray] = {}
        if spans:
            full0 = ex.extract_chat(user0, positions=None, text_id=sid)
            for t_true in {inv.get(p["code_l1b"][int(a):int(b)].strip()) for a, b in spans}:
                if not t_true:
                    continue
                occ = term_spans(p["code_l0"], t_true)
                if not occ:
                    continue
                pos0, _ = span_token_positions(tokz, user0, p["code_l0"], [occ[0]])
                if pos0 and max(pos0) < len(full0.activations):
                    l0_vec[t_true] = full0.activations[max(pos0)].astype(np.float32)

        targets = {a: {} for a in ARMS}
        d_edit: list[np.ndarray] = []
        d_edit_foreign: list[np.ndarray] = []
        n_edit = 0
        for si, sp in enumerate(spans):
            diag["spans"] += 1
            read = reads.get((sid, si))
            if read is None:
                diag["no_read"] += 1
                continue
            pos, _ = span_token_positions(tokz, user1, p["code_l1b"], [sp])
            if not pos:
                continue
            decoy = p["code_l1b"][int(sp[0]):int(sp[1])].strip()
            true = inv.get(decoy)
            edited, n1 = (substitute_terms(read, {decoy: true}) if true else (read, 0))
            # C2's edit has the same SHAPE as W1's (one identifier swapped for another in the
            # same read) and the wrong CONTENT: the true term of a different item, chosen by a
            # seeded hash of this span so the arm is reproducible and cannot drift between runs.
            others = [q for q in pairs if q["snippet_id"] != sid and q["terms_true"]]
            ft = (others[zlib.crc32(f"{sid}#{si}".encode()) % len(others)]["terms_true"][0]
                  if others else "value")
            foreign, _ = (substitute_terms(read, {decoy: ft}) if true else (read, 0))
            if n1:
                n_edit += 1; diag["editable"] += 1
            v_w1 = ar.reconstruct(edited if n1 else read).numpy().astype(np.float32)
            v_c1 = ar.reconstruct(read).numpy().astype(np.float32)
            v_c2 = ar.reconstruct(foreign if n1 else read).numpy().astype(np.float32)
            h0 = l0_vec.get(true) if true else None
            if h0 is None:
                diag["c3_missing"] += 1
                v_c3 = v_c1                       # C3 falls back to the round trip at that span,
                                                  # keeping positions equal; counted above.
            else:
                v_c3 = ar.reconstruct(av.generate(h0, temperature=0.0,
                                                  max_new_tokens=MAX_NEW_READ)
                                      ).numpy().astype(np.float32)
            # H-W2' needs the direction the edit actually carried, per span: AR(edited) minus
            # AR(unedited). Only editable spans contribute -- an unedited span contributes exactly
            # zero by construction and would dilute the mean toward the origin.
            if n1:
                d_edit.append(v_w1 - v_c1)
                d_edit_foreign.append(v_c2 - v_c1)
            for arm, v in (("W1_edit", v_w1), ("C1_roundtrip", v_c1),
                           ("C2_foreign", v_c2), ("C3_ceiling", v_c3)):
                for q in pos:
                    targets[arm][q] = torch.from_numpy(v)

        if not targets["W1_edit"]:
            continue
        pids, rids = tr["l1b_prompt_ids"], tr["l0_reply_ids"]
        rep.set_targets(None)
        base = logp(pids, rids)
        h_unsteered = grab["h"][0, len(pids) - 1].float().cpu().numpy().copy()
        row = {"snippet_id": sid, "n_positions": len(targets["W1_edit"]), "n_editable": n_edit,
               "logp_noop": base, "n_tok": len(rids)}
        dh: dict[str, np.ndarray] = {}
        for arm in ARMS:
            rep.set_targets(targets[arm])
            row[f"dG_{arm}"] = logp(pids, rids) - base
            if arm in ("W1_edit", "C1_roundtrip"):
                # Read AFTER the write, at `last_prompt` -- the answer-production site. Reading
                # the WRITTEN position would return the written vector by construction; that is
                # the vacuity corrected in 2026-09-04_stage2-vacuous-correction.md.
                dh[arm] = (grab["h"][0, len(pids) - 1].float().cpu().numpy() - h_unsteered)
            if not args.no_generate:
                reply = gen(pids)
                got, ok = graded(reply, tr["truth"])
                row[f"acc_{arm}"], row[f"parse_{arm}"] = bool(ok), got is not None
        rep.set_targets(None)
        if not args.no_generate:
            reply = gen(pids)
            got, ok = graded(reply, tr["truth"])
            row["acc_noop"], row["parse_noop"] = bool(ok), got is not None
        # H-W2': did the write move the answer site IN THE DIRECTION OF THE EDIT? Both sides are
        # differences, so they are centred by construction and stage 0's anisotropy -- which put
        # every raw cosine near 0.97 -- cannot inflate this statistic.
        def _cos(a, b):
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            return None if na < 1e-9 or nb < 1e-9 else float(np.dot(a, b) / (na * nb))
        if d_edit:
            de = np.mean(d_edit, 0); dfe = np.mean(d_edit_foreign, 0)
            row["cos_dh_dedit"] = _cos(dh.get("W1_edit"), de) if "W1_edit" in dh else None
            row["cos_dh_dedit_foreign"] = _cos(dh.get("W1_edit"), dfe) if "W1_edit" in dh else None
            # A write carrying no edit must not register against the edit direction.
            row["cos_dh_c1_dedit"] = (_cos(dh["C1_roundtrip"], de)
                                      if "C1_roundtrip" in dh else None)
        rows.append(row)
        sink.write(json.dumps(row) + "\n"); sink.flush()
        print(f"[W1] {pi+1}/{len(pairs)} {sid} pos={row['n_positions']} edit={n_edit} "
              f"dG_W1={row['dG_W1_edit']:+.2f} dG_C1={row['dG_C1_roundtrip']:+.2f}", flush=True)

    sink.close()
    print(f"[W1] {len(rows)} rows; diag={diag}", flush=True)

    base_p = _PROJ / f"data/nla/p0/steerv2/{args.model}/run/baseline.jsonl"
    baseline = None
    if not args.no_generate:
        acc = [r["acc_noop"] for r in rows if "acc_noop" in r]
        par = [r["parse_noop"] for r in rows if "parse_noop" in r]
        if acc:
            baseline = {"acc": float(np.mean(acc)), "parse": float(np.mean(par)),
                        "n": len(acc), "source": "this run (unsteered)"}
    if baseline is None and base_p.exists():
        b = [json.loads(l) for l in open(base_p)]
        if b:
            baseline = {"acc": float(np.mean([x.get("correct", False) for x in b])),
                        "parse": float(np.mean([x.get("parsed", False) for x in b])),
                        "n": len(b), "source": str(base_p)}
    st = score(rows, baseline, out / "writeback_stats.json")
    st["diag"] = diag
    json.dump(st, open(out / "writeback_stats.json", "w"), indent=2)
    print(json.dumps({k: v for k, v in st.items() if k != "per_arm"}, indent=2), flush=True)
    for arm, a in st["per_arm"].items():
        print(f"  {arm:14s} dG_sum {a['dG_sum_mean']:+8.2f} nats "
              f"[{a['ci95'][0]:+.2f}, {a['ci95'][1]:+.2f}]  "
              f"support={a['clears_support']}  veto={a['veto']}  acc={a['acc']}", flush=True)
    print(f"[W1] VERDICT {st['verdict']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
