"""S — optimised-vector site test (pre-registered 2026-09-03, blocked on R != R-UNINF).

The question B4/B5/KV-bypass/coverage could not separate: "there is no belief to edit" versus
"this site cannot deliver one". If a direction learned by gradient ascent on M at the banked
site (L32, last prompt token, prefill-only, alpha 1.0) generalises to held-out items, the site
can carry belief information and every null so far was a direction problem. If the best
direction the optimiser can find fits its training items and nothing else, the site is closed
regardless of recipe.

It is a DIRECTION search, not a magnitude search: the hook normalises delta and scales by
alpha*||h|| exactly as every banked arm did, so ||delta|| is pinned to the V3-at-alpha-1 norm
by construction and the optimiser can only choose where to point.

Per split (5 splits, 30/30, split seed = optimiser seed):
  delta      -- ascent on  sum_train M_i(delta)
  delta_shuf -- ascent with y_clean / y_corr SWAPPED on train (the label-shuffle control: an
                optimiser that "generalises" this too is fitting reply style, not belief)
Both evaluated on the 30 held-out items: dM per item, paired bootstrap CI; on the held-out L0
prompts: change in log P(y_clean | x_l0) (specificity — a belief edit must not damage the
clean prompt's own trace); cos(delta, V1/V3/V4); optionally greedy accuracy + rescue count.

Frozen rule (site-alive-prereg, Experiment S):
  S-LIVE  held-out mean dM >= +0.05, CI excl 0, on >= 4/5 splits, and delta_shuf does not.
  S-DEAD  train dM rises but held-out dM inside the floor on >= 3/5 splits.
  S-UNINF train dM does not rise -> optimiser failure; fix before reading anything.
"""
from __future__ import annotations

import argparse
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

from steer_run import AR_CHECKPOINTS, HOSTS, SEED, MAX_NEW_GEN, build_user, graded, load_pairs  # noqa: E402

SUPPORT = 0.05


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma12b", choices=sorted(HOSTS))
    ap.add_argument("--allow-banked-host", action="store_true")
    ap.add_argument("--traces", required=True, help="R's traces.jsonl (replies as token ids)")
    ap.add_argument("--llr-stats", required=True, help="R's llr_stats.json (for the floor)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--with-greedy", action="store_true", help="also generate on held-out items")
    ap.add_argument("--smoke", action="store_true", help="1 split, 6/6 items, 4 steps")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-hours", type=float, default=8.0)
    args = ap.parse_args()
    if args.model == "qwen7b" and not args.allow_banked_host:
        raise SystemExit("qwen7b is not a permitted host (2026-09-02 constraint)")

    import torch
    from capture_core import WallGuard
    from extract import ActivationExtractor
    from nla_inference import NLACritic
    from steer import ActivationSteerer, SteerSpec
    from steer_vectors import cosine

    model_name, layer = HOSTS[args.model]
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    wall = WallGuard(args.max_hours)
    floor = float(json.loads(Path(args.llr_stats).read_text())["floor"])
    traces = {json.loads(l)["snippet_id"]: json.loads(l) for l in open(args.traces)}
    rng = random.Random(SEED)
    pairs = [p for p in load_pairs(None, rng) if p["snippet_id"] in traces]
    if args.smoke:
        pairs = pairs[:12]; args.splits = 1; args.steps = 4
    sids = [p["snippet_id"] for p in pairs]
    flip = {s for s in sids if traces[s]["l0_correct"] and not traces[s]["l1b_correct"]}
    print(f"[S] {len(pairs)} items · flippable {len(flip)} · floor {floor:.4f}", flush=True)

    ex = ActivationExtractor(model_name, layer, device=args.device)
    model, tokz = ex.model, ex.tokenizer
    model.eval()
    for prm in model.parameters():
        prm.requires_grad_(False)
    steerer = ActivationSteerer(model, layer)
    d_model = ex.d_model

    def logp(p_ids: list[int], r_ids: list[int], delta: torch.Tensor | None, grad: bool) -> torch.Tensor:
        """sum log P(reply | prompt) with delta written at the last prompt token, alpha fixed."""
        seq = torch.tensor([p_ids + r_ids], device=model.device)
        plen = len(p_ids)
        steerer.set_spec(None if delta is None else SteerSpec(delta=delta, alpha=args.alpha, positions="last_prompt"),
                         prompt_len=plen, max_total=seq.shape[1])
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            logits = model(seq).logits[0, plen - 1:-1]
            tgt = seq[0, plen:]
            tot = torch.zeros((), device=model.device, dtype=torch.float32)
            for s in range(0, logits.shape[0], 256):
                lp = torch.log_softmax(logits[s:s + 256].float(), dim=-1).gather(1, tgt[s:s + 256, None])[:, 0]
                tot = tot + lp.sum()
        steerer.reset()
        return tot

    def M(sid: str, delta: torch.Tensor | None, grad: bool = False, swap: bool = False) -> torch.Tensor:
        t = traces[sid]
        yc, yk = t["l0_reply_ids"], t["l1b_reply_ids"]
        if swap:
            yc, yk = yk, yc
        return (logp(t["l1b_prompt_ids"], yc, delta, grad) - logp(t["l1b_prompt_ids"], yk, delta, grad)) / max(len(yc), 1)

    # reference directions for the cosine report (V1 via the AR, V3/V4 from last-token acts)
    acts = {}
    for p in pairs:
        h0 = ex.extract_chat(build_user(p["code_l0"], p["call_l0"]), None, text_id="x").activations[-1]
        h1 = ex.extract_chat(build_user(p["code_l1b"], p["call_l1b"]), None, text_id="x").activations[-1]
        acts[p["snippet_id"]] = torch.tensor(h0).float() - torch.tensor(h1).float()
    v1 = {}
    if AR_CHECKPOINTS.get(args.model):
        ar = NLACritic(AR_CHECKPOINTS[args.model], device=args.device)
        for p in pairs:
            v1[p["snippet_id"]] = (ar.reconstruct(p["gloss_true"]).float() - ar.reconstruct(p["gloss_decoy"]).float()).cpu()
        del ar; torch.cuda.empty_cache()

    def optimise(train: list[str], seed: int, swap: bool) -> tuple[torch.Tensor, list[float]]:
        g = torch.Generator().manual_seed(seed)
        delta = torch.randn(d_model, generator=g).to(model.device).float()
        delta = (delta / delta.norm()).requires_grad_(True)
        opt = torch.optim.Adam([delta], lr=args.lr)
        r = random.Random(seed)
        curve = []
        for step in range(args.steps):
            batch = r.sample(train, min(args.batch, len(train)))
            opt.zero_grad()
            tot = 0.0
            for sid in batch:
                m = M(sid, delta, grad=True, swap=swap)
                (-m / len(batch)).backward()
                tot += float(m)
            opt.step()
            with torch.no_grad():
                delta.div_(delta.norm().clamp_min(1e-12))   # stay on the sphere: direction only
            curve.append(tot / len(batch))
            print(f"[S]   step {step + 1}/{args.steps} train M {curve[-1]:+.4f}", flush=True)
        return delta.detach().clone(), curve

    def gen(sid: str, delta: torch.Tensor | None) -> tuple[bool, bool]:
        ids = traces[sid]["l1b_prompt_ids"]
        steerer.set_spec(None if delta is None else SteerSpec(delta=delta, alpha=args.alpha, positions="last_prompt"),
                         prompt_len=len(ids), max_total=len(ids) + MAX_NEW_GEN)
        with torch.no_grad():
            o = model.generate(torch.tensor([ids], device=model.device), max_new_tokens=MAX_NEW_GEN,
                               do_sample=False, pad_token_id=tokz.eos_token_id)
        steerer.reset()
        got, ok = graded(tokz.decode(o[0][len(ids):], skip_special_tokens=True), traces[sid]["truth"])
        return ok, got is not None

    # unsteered references, once
    m0 = {s: float(M(s, None)) for s in sids}
    lp0_l0 = {s: float(logp(traces[s]["l0_prompt_ids"], traces[s]["l0_reply_ids"], None, False)) for s in sids}
    print(f"[S] M(noop) mean {np.mean(list(m0.values())):+.4f}", flush=True)

    boot = np.random.default_rng(SEED)

    def ci(x: np.ndarray) -> list[float]:
        if len(x) < 2:
            return [float("nan")] * 2
        mm = x[boot.integers(0, len(x), size=(5000, len(x)))].mean(1)
        return [float(np.percentile(mm, 2.5)), float(np.percentile(mm, 97.5))]

    splits_out = []
    for k in range(args.splits):
        if wall.expired():
            print("[S] wall budget reached", flush=True); break
        rs = random.Random(SEED + k)
        order = sids[:]; rs.shuffle(order)
        half = len(order) // 2
        train, test = order[:half], order[half:]
        rec = {"split": k, "seed": SEED + k, "train": train, "test": test}
        for name, swap in (("delta", False), ("delta_shuf", True)):
            t0 = time.time()
            delta, curve = optimise(train, SEED + k, swap)
            tr_dm = np.array([float(M(s, delta)) - m0[s] for s in train])
            te_dm = np.array([float(M(s, delta)) - m0[s] for s in test])
            spec = np.array([float(logp(traces[s]["l0_prompt_ids"], traces[s]["l0_reply_ids"], delta, False)) - lp0_l0[s]
                             for s in test]) / np.array([max(len(traces[s]["l0_reply_ids"]), 1) for s in test])
            r = {"curve": curve, "train_dM": float(tr_dm.mean()), "test_dM": float(te_dm.mean()),
                 "test_ci95": ci(te_dm), "test_dM_flippable": float(np.mean([float(M(s, delta)) - m0[s] for s in test if s in flip]))
                 if any(s in flip for s in test) else None,
                 "l0_specificity_dlogp_per_tok": float(spec.mean()), "l0_specificity_ci95": ci(spec),
                 "cos_V3_train_mean": cosine(delta.cpu(), torch.stack([acts[s] for s in train]).mean(0)),
                 "cos_V4_test_mean": float(np.mean([cosine(delta.cpu(), acts[s]) for s in test])),
                 "cos_V1_test_mean": float(np.mean([cosine(delta.cpu(), v1[s]) for s in test])) if v1 else None,
                 "minutes": round((time.time() - t0) / 60, 1)}
            if args.with_greedy and not swap:
                acc = [gen(s, delta) for s in test]
                r["greedy_test_acc"] = float(np.mean([a for a, _ in acc]))
                r["greedy_test_base_acc"] = float(np.mean([traces[s]["l1b_correct"] for s in test]))
                r["greedy_rescued_flippable"] = int(sum(1 for s, (a, _) in zip(test, acc) if s in flip and a))
                r["greedy_flippable_in_test"] = int(sum(1 for s in test if s in flip))
            torch.save(delta.cpu(), out / f"{name}_split{k}.pt")
            rec[name] = r
            print(f"[S] split {k} {name}: train dM {r['train_dM']:+.4f} · test dM {r['test_dM']:+.4f} "
                  f"CI {r['test_ci95']} · spec {r['l0_specificity_dlogp_per_tok']:+.4f}", flush=True)
        splits_out.append(rec)
        (out / "splits.json").write_text(json.dumps(splits_out, indent=2))

    # ── frozen rule ──
    n = len(splits_out)
    live = sum(1 for r in splits_out if r["delta"]["test_dM"] >= SUPPORT and r["delta"]["test_ci95"][0] > 0)
    shuf_live = sum(1 for r in splits_out if r["delta_shuf"]["test_dM"] >= SUPPORT and r["delta_shuf"]["test_ci95"][0] > 0)
    train_rises = sum(1 for r in splits_out if r["delta"]["train_dM"] > floor)
    dead = sum(1 for r in splits_out if r["delta"]["train_dM"] > floor and abs(r["delta"]["test_dM"]) <= floor)
    if train_rises == 0:
        verdict = "S-UNINF"
    elif live >= 4 and shuf_live < 4:
        verdict = "S-LIVE"
    elif dead >= 3:
        verdict = "S-DEAD"
    else:
        verdict = "S-MIXED"      # not in the table: report the counts, claim nothing
    res = {"experiment": "S_site_optimise", "seed": SEED, "argv": sys.argv, "model": model_name,
           "host": args.model, "layer": layer, "alpha": args.alpha, "floor": floor, "n_splits": n,
           "splits_live": live, "splits_shuf_live": shuf_live, "splits_train_rises": train_rises,
           "splits_dead": dead, "verdict": verdict, "smoke": args.smoke,
           "finished_utc": datetime.now(timezone.utc).isoformat(), "elapsed_hours": round(wall.elapsed_h(), 3)}
    (out / "site_stats.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2), flush=True)
    steerer.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
