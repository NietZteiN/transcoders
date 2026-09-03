"""R — re-score the banked steering arms on a continuous readout (pre-registered 2026-09-03).

Greedy accuracy on 60 items had a rescue ceiling of 6 items (+0.10) — the support threshold
itself. This replaces the readout, not the interventions.

METRIC M (clean-trace log-likelihood ratio). For item i with corrupt (L1b) prompt x_i, let
y_clean be the model's own greedy L0 reply and y_corr its own greedy L1b reply. Under an
intervention theta applied exactly as in the banked run (same vector, alpha, positions, layer),

    M_i(theta) = [ log P(y_clean | x_i, theta) - log P(y_corr | x_i, theta) ] / |y_clean|

teacher-forced, per token. Positive = the write makes the model more willing to produce its own
clean trace than its corrupt one. Defined on every item, not just the six flippable ones.
dM = M(theta) - M(no-op). Both replies are stored as TOKEN IDS so the teacher-forced sequence is
exactly what the model generated — re-encoding decoded text does not round-trip under BPE.

Why the model's OWN replies, not the ground-truth answer: the answer is emitted after hundreds
of reasoning tokens, and a belief edit that changes the reasoning shows up long before the
`Output:` line. Scoring the trace asks the belief question; scoring the answer token alone would
re-import the bottleneck this metric exists to remove.

Stages (all resumable, all rows keyed):
  1. traces.jsonl   — per item: L0/L1b prompt ids, greedy reply ids, grades; last-token acts.
  2. llr_rows.jsonl — per (item, arm, reply, replicate): sum log P, n tokens.
     Arms: every banked Gemma arm (single-layer L32 last_prompt; id_spans at the two matched
     alphas; multilayer 0..32 at the two matched alphas; V5 replacement; P_prompt), the no-op
     TWICE (bf16 floor), and the L0 prompt (sanity: the clean prompt must prefer its own trace).
  3. llr_stats.json — per-arm mean dM, paired bootstrap CI, floor, and the frozen R verdict.

Hash note: steer_run seeds R_random / picks F_foreign with Python's hash(str), which is salted
per process unless PYTHONHASHSEED is set — it was not. Those controls in the banked rows are
therefore not bit-reproducible; here they use a stable sha1 so THIS file is. Same construction,
different draw; both are controls, neither carries a claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_NLA_ROOT = _HERE.parent
_PROJ = _NLA_ROOT.parent
sys.path.insert(0, str(_NLA_ROOT / "vendor" / "nla-repo"))
sys.path.insert(0, str(_HERE))

from steer_run import (AR_CHECKPOINTS, HOSTS, SEED, MAX_NEW_GEN, build_user, graded,  # noqa: E402
                       load_pairs)

STEERV2 = _PROJ / "data" / "nla" / "p0" / "steerv2"
FLOOR_DEFAULT = 0.01          # nats/token, used only if the two no-op replicates agree exactly
SUPPORT = 0.05                # nats/token — the pre-registered support threshold for dM


def stable_hash(s: str) -> int:
    return int(hashlib.sha1(s.encode()).hexdigest()[:8], 16)


# ── the arm table ────────────────────────────────────────────────────────────────────────────
# Mirrors the banked Gemma corpus arm for arm. `positions` and `ml` are the intervention's shape;
# alpha values for the matched arms are READ from the artifacts the banked runs used, so a typo
# here cannot silently score a different intervention than the one that generated.
def banked_arms(host_dir: Path) -> list[dict]:
    em = json.loads((host_dir / "energy_match.json").read_text())["matched_alphas"]
    cm = json.loads((host_dir / "coverage_match.json").read_text())
    a_ml_primary = em["1.0"]["multi_alpha"]                       # 0.1 on gemma
    # coverage_match records the DELIVERED-matched alpha, which the rule REFUSED (ratio 0.49);
    # the run used the energy convention alpha_ref/sqrt(n_tokens) -- 0.149071 for median 45 --
    # rounded to 6 dp exactly as coverage_alpha.py wrote it, plus the max-delivery arm at 8.0.
    a_cov = round(1.0 / (cm["tokens_per_item_median"] ** 0.5), 6)
    arms: list[dict] = []
    LP = "last_prompt"
    for a in (0.25, 1.0, 4.0):
        arms += [dict(cond="V1_gloss", alpha=a, positions=LP, ml=False),
                 dict(cond="V3_taskvec", alpha=a, positions=LP, ml=False)]
    for c in ("V2_wordedit", "V4_oracle", "R_random", "F_foreign", "A_antipodal"):
        arms.append(dict(cond=c, alpha=1.0, positions=LP, ml=False))
    # R_random at every alpha a belief arm uses, so the sensitivity clause always has a
    # same-alpha control to compare against (the banked corpus had it at 0.1 and 1.0 only).
    for a in (0.1, 0.25, 4.0):
        arms.append(dict(cond="R_random", alpha=a, positions=LP, ml=False))
    for a in (float(a_cov), 8.0):
        for c in ("V1_gloss", "V3_taskvec", "R_random"):
            arms.append(dict(cond=c, alpha=a, positions="id_spans", ml=False))
    for a in (float(a_ml_primary), 1.0):
        for c in ("V3_taskvec", "V4_oracle", "R_random"):
            arms.append(dict(cond=c, alpha=a, positions=LP, ml=True))
    arms.append(dict(cond="V5_replace", alpha=1.0, positions=LP, ml=True))
    arms.append(dict(cond="P_prompt", alpha=0.0, positions=None, ml=False))
    return arms


def arm_key(a: dict) -> str:
    return f"{a['cond']}|{a['alpha']}|{a['positions']}|{'ML' if a['ml'] else 'SL'}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma12b", choices=sorted(HOSTS))
    ap.add_argument("--allow-banked-host", action="store_true")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--only-arms", default=None, help="comma list of arm keys (see arm_key)")
    ap.add_argument("--smoke", action="store_true",
                    help="3 items, V3@1.0 + V1@1.0 + no-op x2 + L0 sanity; prints the floor")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-new-gen", type=int, default=MAX_NEW_GEN)
    ap.add_argument("--max-hours", type=float, default=6.0)
    args = ap.parse_args()
    if args.model == "qwen7b" and not args.allow_banked_host:
        raise SystemExit("qwen7b is not a permitted host (2026-09-02 constraint)")

    import torch
    from capture_core import JsonlSink, WallGuard
    from extract import ActivationExtractor
    from nla_inference import NLACritic
    from steer import ActivationSteerer, SteerSpec
    from steer_multilayer import MultiLayerSpec, MultiLayerSteerer
    from steer_vectors import (ActivationPair, TaskVectorBank, antipodal, random_direction,
                               substitute_terms, word_edit_direction)
    from span_positions import span_token_positions

    model_name, layer = HOSTS[args.model]
    host_dir = STEERV2 / args.model
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    wall = WallGuard(args.max_hours)
    rng = random.Random(SEED)
    pairs = load_pairs(3 if args.smoke else args.limit, rng)
    print(f"[R] {len(pairs)} pairs · host {args.model} · layer {layer}", flush=True)

    arms = banked_arms(host_dir)
    if args.smoke:
        arms = [a for a in arms if arm_key(a) in {"V3_taskvec|1.0|last_prompt|SL",
                                                  "V1_gloss|1.0|last_prompt|SL"}]
    if args.only_arms:
        want = {k.strip() for k in args.only_arms.split(",")}
        arms = [a for a in arms if arm_key(a) in want]
    print(f"[R] {len(arms)} arms: {[arm_key(a) for a in arms]}", flush=True)
    need_ar = any(a["cond"] in {"V1_gloss", "V2_wordedit", "F_foreign", "A_antipodal"}
                  for a in arms)

    ex = ActivationExtractor(model_name, layer, device=args.device)
    model, tokz = ex.model, ex.tokenizer
    model.eval()

    def prompt_ids(code: str, call: str, extra: str = "") -> list[int]:
        return list(tokz.apply_chat_template([{"role": "user", "content": build_user(code, call) + extra}],
                                             tokenize=True, add_generation_prompt=True,
                                             return_dict=False))

    # ── stage 1: traces ──────────────────────────────────────────────────────────────────────
    tr_p = out / "traces.jsonl"
    sink = JsonlSink(tr_p, key_field="snippet_id")
    done = sink.done_keys()
    traces: dict[str, dict] = {}
    if tr_p.exists():
        for l in open(tr_p):
            r = json.loads(l); traces[r["snippet_id"]] = r
    acts: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for i, p in enumerate(pairs, 1):
        sid = p["snippet_id"]
        h0 = ex.extract_chat(build_user(p["code_l0"], p["call_l0"]), None, text_id="x").activations[-1]
        h1 = ex.extract_chat(build_user(p["code_l1b"], p["call_l1b"]), None, text_id="x").activations[-1]
        acts[sid] = (h0, h1)
        if sid in done:
            continue
        row = {"snippet_id": sid, "truth": p["truth"]}
        for tier, code, call in (("l0", p["code_l0"], p["call_l0"]), ("l1b", p["code_l1b"], p["call_l1b"])):
            ids = prompt_ids(code, call)
            with torch.no_grad():
                o = model.generate(torch.tensor([ids], device=model.device),
                                   max_new_tokens=args.max_new_gen, do_sample=False,
                                   pad_token_id=tokz.eos_token_id)
            rep_ids = o[0][len(ids):].tolist()
            rep = tokz.decode(rep_ids, skip_special_tokens=True)
            got, ok = graded(rep, p["truth"])
            row.update({f"{tier}_prompt_ids": ids, f"{tier}_reply_ids": rep_ids,
                        f"{tier}_reply": rep, f"{tier}_answer": got, f"{tier}_correct": ok,
                        f"{tier}_parsed": got is not None})
        sink.append(row); traces[sid] = row
        print(f"[R] trace {i}/{len(pairs)} {sid} L0={row['l0_correct']} L1b={row['l1b_correct']}",
              flush=True)
    sink.close()
    usable = [p for p in pairs if p["snippet_id"] in traces]
    flip = [p["snippet_id"] for p in usable
            if traces[p["snippet_id"]]["l0_correct"] and not traces[p["snippet_id"]]["l1b_correct"]]
    print(f"[R] {len(usable)} usable · flippable {len(flip)}: {flip}", flush=True)

    # ── vectors, built exactly as steer_run builds them ──────────────────────────────────────
    bank = TaskVectorBank([ActivationPair(p["snippet_id"], torch.tensor(acts[p["snippet_id"]][0]),
                                          torch.tensor(acts[p["snippet_id"]][1])) for p in usable])
    v1_cache: dict[str, torch.Tensor] = {}
    v2_cache: dict[str, torch.Tensor] = {}
    if need_ar:
        ar = NLACritic(AR_CHECKPOINTS[args.model], device=args.device)
        for p in usable:
            v1_cache[p["snippet_id"]] = (ar.reconstruct(p["gloss_true"]).float()
                                         - ar.reconstruct(p["gloss_decoy"]).float()).cpu()
            # V2: one decoy->true substitution, same deterministic tie-break as steer_run.
            cands = []
            for true_name, decoy_name in (p.get("rename_map") or {}).items():
                if not decoy_name or true_name.lower() == decoy_name.lower():
                    continue
                _, n = substitute_terms(p["gloss_decoy"], {decoy_name: true_name})
                if n == 1:
                    cands.append((p["gloss_decoy"].lower().find(decoy_name.lower()), decoy_name, true_name))
            if cands:
                cands.sort()
                try:
                    v2_cache[p["snippet_id"]] = word_edit_direction(
                        ar, p["gloss_decoy"], {cands[0][1]: cands[0][2]}, require_single=True).cpu()
                except ValueError:
                    pass
        del ar; torch.cuda.empty_cache()
        print(f"[R] AR directions: V1 {len(v1_cache)} · V2 {len(v2_cache)}", flush=True)

    sids = [p["snippet_id"] for p in usable]
    ml_bank = np.load(host_dir / "multilayer_bank.npz", allow_pickle=True) \
        if any(a["ml"] for a in arms) else None
    if ml_bank is not None:
        Dml = ml_bank["deltas"].astype(np.float32); Cml = ml_bank["clean"].astype(np.float32)
        ml_idx = {str(k): i for i, k in enumerate(ml_bank["item_ids"])}
        assert Dml.shape[2] == ex.d_model and Dml.shape[1] == ex.n_layers, "wrong host's bank"
        Dsum = Dml.sum(axis=0); n_ml = Dml.shape[0]

    def single_delta(cond: str, p: dict):
        sid = p["snippet_id"]
        if cond == "V1_gloss":   return v1_cache.get(sid)
        if cond == "V2_wordedit": return v2_cache.get(sid)
        if cond == "V3_taskvec": return bank.direction_for(exclude=sid)
        if cond == "V4_oracle":  return bank.oracle_for(sid)
        if cond == "R_random":   return random_direction(ex.d_model, seed=SEED + stable_hash(sid) % 9973,
                                                         like=v1_cache.get(sid))
        if cond == "F_foreign":
            others = [s for s in sids if s != sid]
            return v1_cache.get(others[stable_hash(sid) % max(len(others), 1)])
        if cond == "A_antipodal":
            v = v1_cache.get(sid); return None if v is None else antipodal(v)
        raise KeyError(cond)

    def multi_spec(cond: str, p: dict, alpha: float, pos: list[int]) -> MultiLayerSpec:
        i = ml_idx[p["snippet_id"]]
        L = range(layer + 1)
        if cond == "V5_replace":
            return MultiLayerSpec(deltas={}, absolute={l: torch.from_numpy(Cml[i][l].copy()) for l in L},
                                  positions="explicit")
        if cond == "V3_taskvec":
            V = (Dsum - Dml[i]) / (n_ml - 1)
        elif cond == "V4_oracle":
            V = Dml[i]
        elif cond == "R_random":
            g = torch.Generator().manual_seed(SEED + stable_hash(p["snippet_id"]) % 9973)
            return MultiLayerSpec(deltas={l: torch.randn(Dml.shape[2], generator=g) for l in L},
                                  alpha=alpha, positions="explicit")
        else:
            raise KeyError(cond)
        return MultiLayerSpec(deltas={l: torch.from_numpy(V[l].copy()) for l in L},
                              alpha=alpha, positions="explicit")

    span_pos: dict[str, list[int]] = {}
    if any(a["positions"] == "id_spans" for a in arms):
        for p in usable:
            pos, _ = span_token_positions(tokz, build_user(p["code_l1b"], p["call_l1b"]),
                                          p["code_l1b"], p.get("id_spans_l1b") or [])
            if pos:
                span_pos[p["snippet_id"]] = pos
        print(f"[R] id_spans mapped {len(span_pos)}/{len(usable)}", flush=True)

    # ── the teacher-forced likelihood ────────────────────────────────────────────────────────
    single = ActivationSteerer(model, layer)
    multi = MultiLayerSteerer(model, range(layer + 1))

    def logp_reply(p_ids: list[int], r_ids: list[int], spec=None, ml: bool = False,
                   positions: list[int] | None = None) -> tuple[float, int]:
        """sum_t log P(r_t | p, r_<t) under the intervention. One forward, no generation."""
        seq = torch.tensor([p_ids + r_ids], device=model.device)
        plen = len(p_ids)
        single.set_spec(None); multi.set_spec(None)
        if spec is not None:
            if ml:
                multi.set_spec(spec, positions=positions if positions is not None else [plen - 1])
            else:
                single.set_spec(spec, prompt_len=plen, max_total=seq.shape[1])
        with torch.no_grad():
            logits = model(seq).logits[0, plen - 1:-1]          # predicts r_0 .. r_{n-1}
            # log-softmax in fp32, chunked: 1.1k x 262k logits is 1.2 GB in fp32 at once.
            tgt = seq[0, plen:]
            tot = 0.0
            for s in range(0, logits.shape[0], 256):
                lg = logits[s:s + 256].float()
                lp = torch.log_softmax(lg, dim=-1).gather(1, tgt[s:s + 256, None])[:, 0]
                tot += float(lp.sum())
        single.reset(); multi.reset()
        n_written = single.n_positions_written + multi.n_positions_written
        return tot, int(tgt.shape[0])

    # ── stage 2: rows ────────────────────────────────────────────────────────────────────────
    rows_p = out / "llr_rows.jsonl"
    rsink = JsonlSink(rows_p, key_field="key")
    rdone = rsink.done_keys()

    def emit(key: str, **kw) -> None:
        if key in rdone:
            return
        rsink.append({"key": key, **kw}); rdone.add(key)

    jobs: list[tuple[str, dict | None, str]] = []          # (tag, arm|None, prompt kind)
    jobs += [("noop#1", None, "l1b"), ("noop#2", None, "l1b"), ("l0prompt", None, "l0")]
    jobs += [(arm_key(a), a, "l1b") for a in arms]
    total = len(jobs) * len(usable) * 2
    n = 0; t0 = time.time()
    for tag, arm, kind in jobs:
        for p in usable:
            if wall.expired():
                print("[R] wall budget reached", flush=True); break
            sid = p["snippet_id"]; tr = traces[sid]
            try:
                extra = ""
                if arm is not None and arm["cond"] == "P_prompt":
                    extra = f"\n\nNote: the identifiers are misleading. This is {p['gloss_true']}."
                p_ids = (tr["l0_prompt_ids"] if kind == "l0"
                         else prompt_ids(p["code_l1b"], p["call_l1b"], extra) if extra
                         else tr["l1b_prompt_ids"])
                spec = None; ml = False; positions = None; skip = None
                if arm is not None and arm["cond"] != "P_prompt":
                    if arm["positions"] == "id_spans":
                        positions = span_pos.get(sid)
                        if not positions:
                            skip = "id_spans unmapped"
                    if arm["ml"]:
                        ml = True
                        spec = multi_spec(arm["cond"], p, arm["alpha"], positions or [len(p_ids) - 1])
                    else:
                        d = single_delta(arm["cond"], p)
                        if d is None:
                            skip = f"{arm['cond']} unavailable for item"
                        else:
                            spec = SteerSpec(delta=d, alpha=arm["alpha"],
                                             positions=positions if positions else "last_prompt")
                for which in ("clean", "corr"):
                    key = f"{sid}|{tag}|{which}"
                    if key in rdone:
                        n += 1; continue
                    if skip:
                        emit(key, snippet_id=sid, arm=tag, reply=which, skipped=True, reason=skip)
                        n += 1; continue
                    r_ids = tr["l0_reply_ids"] if which == "clean" else tr["l1b_reply_ids"]
                    lp, nt = logp_reply(p_ids, r_ids, spec, ml=ml, positions=positions)
                    emit(key, snippet_id=sid, arm=tag, reply=which, logp_sum=lp, n_tok=nt,
                         alpha=None if arm is None else arm["alpha"],
                         positions=None if arm is None else arm["positions"],
                         multilayer=ml, n_positions=len(positions) if positions else 1)
                    n += 1
            except Exception as e:                       # noqa: BLE001
                for which in ("clean", "corr"):
                    emit(f"{sid}|{tag}|{which}", snippet_id=sid, arm=tag, reply=which,
                         error=repr(e)[:200])
                n += 2
            if n % 50 == 0:
                rate = n / max(time.time() - t0, 1e-6)
                print(f"[R] {n}/{total} · {rate * 60:.0f} rows/min · ETA {(total - n) / rate / 60:.0f} min",
                      flush=True)
    rsink.close()
    single.close(); multi.close()

    # ── stage 3: stats + frozen verdict ──────────────────────────────────────────────────────
    stats = score(rows_p, traces, arms, out / "llr_stats.json", flip)
    (out / "run_manifest.json").write_text(json.dumps({
        "experiment": "R_trace_llr", "seed": SEED, "argv": sys.argv, "model": model_name,
        "host": args.model, "layer": layer, "n_items": len(usable), "flippable": flip,
        "arms": [arm_key(a) for a in arms], "finished_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_hours": round(wall.elapsed_h(), 3)}, indent=2))
    print(json.dumps({k: stats[k] for k in ("floor", "verdict", "sanity")}, indent=2), flush=True)
    return 0


def score(rows_p: Path, traces: dict, arms: list[dict], out_p: Path, flip: list[str]) -> dict:
    """Pure function of the rows file — re-runnable on its own via --score-only."""
    rows = [json.loads(l) for l in open(rows_p)]
    by: dict[tuple[str, str], dict[str, dict]] = {}
    for r in rows:
        if r.get("skipped") or r.get("error"):
            continue
        by.setdefault((r["arm"], r["reply"]), {})[r["snippet_id"]] = r

    def M(tag: str) -> dict[str, float]:
        c, k = by.get((tag, "clean"), {}), by.get((tag, "corr"), {})
        return {s: (c[s]["logp_sum"] - k[s]["logp_sum"]) / max(c[s]["n_tok"], 1)
                for s in c if s in k}

    m1, m2, m0l0 = M("noop#1"), M("noop#2"), M("l0prompt")
    common = sorted(set(m1) & set(m2))
    d_rep = np.array([m1[s] - m2[s] for s in common])
    floor = float(2 * d_rep.std()) if len(common) > 1 and d_rep.std() > 0 else FLOOR_DEFAULT
    # sanity: the clean prompt must prefer its own trace more than the corrupt prompt does
    san = sorted(set(m1) & set(m0l0))
    san_gap = np.array([m0l0[s] - m1[s] for s in san])
    san_flip = np.array([m0l0[s] - m1[s] for s in san if s in set(flip)])
    rng = np.random.default_rng(SEED)

    def boot_ci(x: np.ndarray, n: int = 5000) -> tuple[float, float]:
        if len(x) < 2:
            return (float("nan"), float("nan"))
        idx = rng.integers(0, len(x), size=(n, len(x)))
        means = x[idx].mean(1)
        return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))

    per_arm: dict[str, dict] = {}
    for a in arms:
        tag = arm_key(a); ma = M(tag)
        keys = sorted(set(ma) & set(m1))
        dm = np.array([ma[s] - m1[s] for s in keys])
        lo, hi = boot_ci(dm)
        per_arm[tag] = {"n": len(keys), "mean_dM": float(dm.mean()) if len(dm) else None,
                        "ci95": [lo, hi], "sd": float(dm.std()) if len(dm) else None,
                        "mean_dM_flippable": float(np.mean([ma[s] - m1[s] for s in keys if s in set(flip)]))
                        if any(s in set(flip) for s in keys) else None,
                        "cond": a["cond"], "alpha": a["alpha"], "positions": a["positions"],
                        "multilayer": a["ml"]}

    # ── the frozen rule (2026-09-03_site-alive-prereg.md, Experiment R) ──
    belief = {"V1_gloss", "V2_wordedit", "V3_taskvec", "V4_oracle", "V5_replace"}
    rand_by_alpha = {(v["alpha"], v["positions"], v["multilayer"]): v for k, v in per_arm.items()
                     if v["cond"] == "R_random" and v["mean_dM"] is not None}

    def clears(v: dict, thr: float) -> bool:
        return v["mean_dM"] is not None and v["mean_dM"] >= thr and v["ci95"][0] > 0

    r1 = []
    for k, v in per_arm.items():
        if v["cond"] in belief and clears(v, SUPPORT):
            rr = rand_by_alpha.get((v["alpha"], v["positions"], v["multilayer"]))
            if rr is None or rr["mean_dM"] < v["mean_dM"] - floor:
                r1.append(k)
    any_random_moves = any(v["mean_dM"] is not None and abs(v["mean_dM"]) > floor
                           and (v["ci95"][0] > 0 or v["ci95"][1] < 0)
                           for v in per_arm.values() if v["cond"] == "R_random")
    if r1:
        verdict = "R-1"
    elif any_random_moves:
        verdict = "R-0"
    else:
        verdict = "R-UNINF"
    res = {"floor": floor, "n_floor_items": len(common), "support_threshold": SUPPORT,
           "sanity": {"n": len(san), "mean_gap_l0_minus_l1b_prompt": float(san_gap.mean()) if len(san) else None,
                      "frac_positive": float((san_gap > 0).mean()) if len(san) else None,
                      "flippable_mean_gap": float(san_flip.mean()) if len(san_flip) else None,
                      "flippable_all_positive": bool((san_flip > 0).all()) if len(san_flip) else None},
           "M_noop_mean": float(np.mean([m1[s] for s in m1])) if m1 else None,
           "verdict": verdict, "supporting_arms": r1, "random_moves": any_random_moves,
           "per_arm": per_arm, "flippable": flip}
    out_p.write_text(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    if "--score-only" in sys.argv:
        # re-score an existing rows file without touching the GPU
        i = sys.argv.index("--score-only"); d = Path(sys.argv[i + 1])
        traces = {json.loads(l)["snippet_id"]: json.loads(l) for l in open(d / "traces.jsonl")}
        flip = [s for s, t in traces.items() if t["l0_correct"] and not t["l1b_correct"]]
        host = "gemma12b"
        r = score(d / "llr_rows.jsonl", traces, banked_arms(STEERV2 / host), d / "llr_stats.json", flip)
        print(json.dumps({k: r[k] for k in ("floor", "verdict", "sanity", "supporting_arms")}, indent=2))
        sys.exit(0)
    sys.exit(main())
