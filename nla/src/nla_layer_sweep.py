"""H-S1: how many layers should we steer? Split-half-honest top-k sweep over the 33 live 4B pairs.

Pre-registered in log/nla-harness/2026-09-11_steering-sweep-prereg.md.

WHY A NEW RUNNER rather than more `sets:` in the gate config. Ranking layers by their single-layer
effect and then scoring the top-k on the SAME 60 items is selection bias: the ranking already used
those items' noise. This runner ranks on one half (`crc32(snippet_id) & 1`) and evaluates on the
other, both directions, then pools — so no layer set is ever chosen and scored on the same items.
A DEPTH-SPACED ladder (k layers evenly spaced over the live range) runs alongside as a
selection-free control; if the two ladders agree, selection was not doing the work.

Readout is `G_sum` (teacher-forced log-prob of the banked L0 reply under the L1b prompt), NOT
accuracy: the 4B host loses only 2 of 60 items to obfuscation and the flippable census found 6/60
flippable, so accuracy cannot support a claim here (see H-S4 in the prereg).

Everything is reused from the frozen gate: banked vectors (`gate/vectors/L{K}.npz`, 34 layers,
uniform 60 items / 471 spans), `PositionReplacer`, the arm definitions and the liveness rules.
No new GPU-side machinery and no re-extraction.
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

from nla_ml_gate import (liveness, load_cfg, load_host, load_vectors, root,  # noqa: E402
                         select_pairs)
from steer import PositionReplacer  # noqa: E402

TAG = "[LSWEEP]"
K_LADDER = (1, 2, 3, 4, 6, 8, 12, 16, 24, 33)   # frozen in the prereg
ARMS = ("c3", "swap", "edit", "random")          # transport, ceiling, edit, null
N_BOOT = 10_000


def half(sid: str) -> int:
    return zlib.crc32(sid.encode()) & 1


def boot_mean(x: np.ndarray, seed: int) -> dict:
    """Cluster bootstrap over items (the unit of analysis), as in every entry in this family."""
    if not len(x):
        return {"mean": float("nan"), "ci95": [float("nan")] * 2, "n": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(N_BOOT, len(x)))
    b = np.sort(x[idx].mean(1))
    return {"mean": float(x.mean()), "ci95": [float(b[int(.025 * N_BOOT)]), float(b[int(.975 * N_BOOT)])],
            "n": int(len(x))}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(_PROJ / "nla/configs/nla_ml_gate.yaml"))
    ap.add_argument("--root", default=None)
    ap.add_argument("--traces", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-hours", type=float, default=3.0)
    args = ap.parse_args()
    cfg = load_cfg(Path(args.config))
    if "qwen" in cfg["train"]["host"]["model_id"].lower():
        print(f"{TAG} REFUSED: Chinese models are not run in this project."); return 2
    # select_pairs() reads these off args (it is the gate's own item selector, reused verbatim so the
    # item set is identical to every other run in this family): _cfg for the seed, smoke for its
    # 3-item cap, plus repair/max_spans_per_item.
    args._cfg = cfg
    args.repair = bool(cfg["repair"]); args.max_spans_per_item = int(cfg["max_spans_per_item"])
    rt = root(cfg, args.root)
    args.traces = args.traces or str(_PROJ / "data/nla/p0/trace_llr/gemma4b/traces.jsonl")
    n_layers = int(cfg["train"]["host"]["n_layers"]); d_model = int(cfg["train"]["host"]["d_model"])
    vdir = rt / cfg["gate_subdir"] / "vectors"
    out = Path(args.out or (rt / cfg["gate_subdir"] / "layer_sweep"))
    out.mkdir(parents=True, exist_ok=True)

    # liveness first, exactly as the gate does it: a PAIR-DEAD layer is never steered
    live = sorted(K for K in range(n_layers)
                  if liveness(rt, K, cfg)["live"] and (vdir / f"L{K}.npz").exists())
    print(f"{TAG} live layers ({len(live)}): {live}", flush=True)
    if len(live) < max(K_LADDER):
        print(f"{TAG} note: ladder caps at {len(live)} live layers")

    traces = {t["snippet_id"]: t for t in map(json.loads, open(args.traces))}
    vec = {K: load_vectors(vdir, K) for K in live}
    ref = vec[live[0]][0]
    for K in live[1:]:                       # the gate's cross-layer anchoring assertion
        assert set(vec[K][0]) == set(ref), f"L{K}: item set differs from L{live[0]}"

    model, tok = load_host(cfg, args.device)
    reps = {K: PositionReplacer(model, K) for K in live}
    seed = int(cfg["seed"])

    @torch.no_grad()
    def logp(pids, rids, expect: int) -> float:
        seq = torch.tensor([list(pids) + list(rids)], device=model.device)
        for r in reps.values():
            r.reset()
        lg = model(seq, use_cache=False, logits_to_keep=len(rids) + 1).logits[0, :-1]
        tgt = seq[0, len(pids):]
        tot = 0.0
        for s in range(0, lg.shape[0], 256):
            tot += float(torch.log_softmax(lg[s:s + 256].float(), -1)
                         .gather(1, tgt[s:s + 256, None])[:, 0].sum())
        n_w = sum(r.n_positions_written for r in reps.values())
        if n_w != expect:
            raise RuntimeError(f"replacers wrote {n_w} positions, expected {expect}")
        return tot

    def targets(K: int, sid: str, kind: str) -> dict:
        by, arr = vec[K]
        tg = {}
        for r in by[sid]:
            i = int(r["vec"])
            if kind == "random":
                rng = np.random.default_rng(seed + zlib.crc32(f"{sid}#{r['span_i']}#L{K}#rnd".encode()))
                v = rng.standard_normal(d_model).astype(np.float32)
                v = torch.from_numpy(v / (np.linalg.norm(v) + 1e-12))
            else:
                v = torch.from_numpy(arr[{"c3": "c3", "swap": "h0", "edit": "edit"}[kind]][i])
            for q in r["positions"]:
                tg[q] = v
        return tg

    def run(Ks, sid, kind, pids, rids) -> float:
        n = 0
        for K, r in reps.items():
            t = targets(K, sid, kind) if K in Ks else None
            r.set_targets(t); n += len(t or {})
        return logp(pids, rids, n)

    sids = [p["snippet_id"] for p in select_pairs(args, traces) if p["snippet_id"] in ref]
    if args.limit:
        sids = sids[:args.limit]
    halves = {0: [s for s in sids if half(s) == 0], 1: [s for s in sids if half(s) == 1]}
    print(f"{TAG} {len(sids)} items · split halves {len(halves[0])}/{len(halves[1])}", flush=True)

    # ── stage 1: per-layer single-layer dG, per half (this is the RANKING signal) ──
    t0 = time.time(); n_fwd = 0
    base = {}
    single = {h: {K: [] for K in live} for h in (0, 1)}
    for sid in sids:
        tr = traces[sid]; pids, rids = tr["l1b_prompt_ids"], tr["l0_reply_ids"]
        for r in reps.values():
            r.set_targets(None)
        base[sid] = logp(pids, rids, 0); n_fwd += 1
        for K in live:
            single[half(sid)][K].append(run({K}, sid, "c3", pids, rids) - base[sid]); n_fwd += 1
        if time.time() - t0 > args.max_hours * 3600:
            print(f"{TAG} wall-clock stop during ranking"); break
    rank = {h: sorted(live, key=lambda K: -float(np.mean(single[h][K]))) for h in (0, 1)}
    print(f"{TAG} ranking on half 0: {rank[0][:8]}...\n{TAG} ranking on half 1: {rank[1][:8]}...", flush=True)

    # ── stage 2: evaluate top-k (ranked on the OTHER half) + a selection-free depth-spaced ladder ──
    rows = []
    for sid in sids:
        tr = traces[sid]; pids, rids = tr["l1b_prompt_ids"], tr["l0_reply_ids"]
        h = half(sid); other = 1 - h
        row = {"snippet_id": sid, "half": h, "logp_U": base[sid]}
        for k in K_LADDER:
            if k > len(live):
                continue
            topk = rank[other][:k]                                   # selected on the other half
            spaced = [live[round(i * (len(live) - 1) / max(1, k - 1))] for i in range(k)] if k > 1 else [live[len(live) // 2]]
            spaced = sorted(set(spaced))
            for name, Ks in (("topk", topk), ("spaced", spaced)):
                for arm in ARMS:
                    row[f"dG_{arm}_{name}{k}"] = run(set(Ks), sid, arm, pids, rids) - base[sid]; n_fwd += 1
            row[f"layers_topk{k}"] = sorted(topk); row[f"layers_spaced{k}"] = spaced
        rows.append(row)
        print(f"{TAG} {len(rows)}/{len(sids)} {sid} · {n_fwd} fwd · {(time.time()-t0)/60:.1f} min", flush=True)
        if time.time() - t0 > args.max_hours * 3600:
            print(f"{TAG} wall-clock stop after {len(rows)} items"); break
    for r in reps.values():
        r.close()

    (out / "sweep_rows.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    # ── score: the frozen H-S1 rules ──
    A = lambda key: np.array([r[key] for r in rows if key in r])  # noqa: E731
    stats = {"live_layers": live, "k_ladder": list(K_LADDER), "n_items": len(rows), "seed": seed,
             "n_forwards": n_fwd, "ms_per_forward": 1000 * (time.time() - t0) / max(n_fwd, 1),
             "ranking_half0": rank[0], "ranking_half1": rank[1], "ladders": {}}
    for name in ("topk", "spaced"):
        stats["ladders"][name] = {str(k): {arm: boot_mean(A(f"dG_{arm}_{name}{k}"), seed + k)
                                          for arm in ARMS}
                                  for k in K_LADDER if k <= len(live)}
    c3 = {k: stats["ladders"]["topk"][str(k)]["c3"]["mean"] for k in K_LADDER if k <= len(live)}
    sw1 = stats["ladders"]["topk"]["1"]["swap"]["mean"]
    best_k = max(c3, key=c3.get)
    gain = c3[best_k] - c3[1]
    drop = c3[1] - c3[max(c3)]
    thr = 0.10 * abs(sw1)
    d_best = boot_mean(A(f"dG_c3_topk{best_k}") - A("dG_c3_topk1"), seed + 1)
    d_max = boot_mean(A(f"dG_c3_topk{max(c3)}") - A("dG_c3_topk1"), seed + 2)
    if gain >= thr and d_best["ci95"][0] > 0:
        verdict = "MORE-IS-BETTER"
    elif drop >= thr and d_max["ci95"][1] < 0:
        verdict = "MORE-IS-WORSE"
    else:
        verdict = "SATURATES"
    sat = next((k for k in sorted(c3) if c3[k] >= 0.95 * c3[best_k]), best_k)
    stats.update({"verdict": verdict, "best_k": int(best_k), "gain_vs_k1": float(gain),
                  "threshold_0.10_S_swap": float(thr), "delta_best_vs_k1": d_best,
                  "delta_kmax_vs_k1": d_max, "k_at_95pct_of_max": int(sat)})
    (out / "layer_sweep_stats.json").write_text(json.dumps(stats, indent=1))
    print(f"\n{TAG} VERDICT {verdict} · best_k={best_k} (gain {gain:+.2f} vs thr {thr:.2f}) · "
          f"95% of max first at k={sat}", flush=True)
    for k in sorted(c3):
        t = stats['ladders']['topk'][str(k)]['c3']; s = stats['ladders']['spaced'][str(k)]['c3']
        print(f"{TAG}   k={k:<3} topk {t['mean']:+8.2f} [{t['ci95'][0]:+.2f},{t['ci95'][1]:+.2f}] · "
              f"spaced {s['mean']:+8.2f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
