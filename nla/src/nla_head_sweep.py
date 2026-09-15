"""H-S2/H-S3: which attention heads carry the NLA-steered state to the answer tokens? (4B host)

Pre-registered in log/nla-harness/2026-09-11_steering-sweep-prereg.md.

H-S1 (job 390856) settled the write configuration: steering ONE layer beats many
(k=33 minus k=1 = -11.70 nats, CI [-17.22,-6.51]), and the best single layer is **L7**
(dG_c3 +65.75, vs +50.26 for a depth-matched control). So this sweep writes at L7 only.

Structural fact that makes the question well-posed: the write lands on PROMPT span positions and
G_sum scores REPLY tokens, so every nat must cross positions through at least one attention layer
ABOVE the write. Writing at L7 leaves 26 downstream layers x 8 heads + 26 MLPs = 234 components.

Per component c, two patched forwards against two reference runs (U = unsteered L1b, S = L1b with
the c3 vector written at L7):
    suf_c = dG(U, c<-S)          what c carries forward on its own
    nec_c = dG(S) - dG(S, c<-U)  what is lost when c alone is blind to the write
Then a joint stage: greedy top-k by sufficiency, SELECTED ON THE OPPOSITE HALF of the items
(crc32 parity), against a layer-matched random-k null -- so "how many heads suffice" is never
chosen and scored on the same items.

Reuses `head_patch.py` (ComponentPatcher, tested on CPU) and the gate's banked vectors; it needs
NO AV/AR checkpoint, which is why it works on the 4B even though AR_CHECKPOINTS has no 4B entry.
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

from head_patch import MLP, Component, ComponentPatcher, components, is_global_layer  # noqa: E402
from nla_ml_gate import load_cfg, load_host, load_vectors, root, select_pairs  # noqa: E402
from steer import PositionReplacer  # noqa: E402

TAG = "[HEADS]"
WRITE_LAYER = 7          # H-S1 winner
K_LIST = (1, 2, 4, 8, 16, 32)
N_RANDOM = 3
IDENT_TOL = 0.05         # SELF_c determinism tolerance (nats), prereg
N_BOOT = 10_000


def half(sid: str) -> int:
    return zlib.crc32(sid.encode()) & 1


def boot(x: np.ndarray, seed: int) -> dict:
    if not len(x):
        return {"mean": float("nan"), "ci95": [float("nan")] * 2, "n": 0}
    rng = np.random.default_rng(seed)
    b = np.sort(x[rng.integers(0, len(x), size=(N_BOOT, len(x)))].mean(1))
    return {"mean": float(x.mean()), "ci95": [float(b[250]), float(b[9750])], "n": int(len(x))}


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = a.argsort().argsort().astype(float), b.argsort().argsort().astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    d = float(np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))
    return float((ra * rb).sum() / d) if d else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(_PROJ / "nla/configs/nla_ml_gate.yaml"))
    ap.add_argument("--root", default=None)
    ap.add_argument("--traces", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--write-layer", type=int, default=WRITE_LAYER)
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-hours", type=float, default=3.0)
    args = ap.parse_args()
    cfg = load_cfg(Path(args.config))
    if "qwen" in cfg["train"]["host"]["model_id"].lower():
        print(f"{TAG} REFUSED: Chinese models are not run in this project."); return 2
    args._cfg = cfg; args.repair = bool(cfg["repair"])
    args.max_spans_per_item = int(cfg["max_spans_per_item"])
    rt = root(cfg, args.root)
    args.traces = args.traces or str(_PROJ / "data/nla/p0/trace_llr/gemma4b/traces.jsonl")
    n_layers = int(cfg["train"]["host"]["n_layers"])
    W = int(args.write_layer)
    out = Path(args.out or (rt / cfg["gate_subdir"] / "head_sweep")); out.mkdir(parents=True, exist_ok=True)

    hcfg = json.loads((_PROJ / cfg["train"]["host"]["text_ckpt"] / "config.json").read_text())
    n_heads, head_dim = int(hcfg["num_attention_heads"]), int(hcfg["head_dim"])
    sweep = tuple(range(W + 1, n_layers))
    comps = components(sweep, n_heads=n_heads)
    print(f"{TAG} write L{W} · downstream layers {sweep[0]}..{sweep[-1]} · {n_heads} heads/layer "
          f"· {len(comps)} components", flush=True)

    traces = {t["snippet_id"]: t for t in map(json.loads, open(args.traces))}
    by, arr = load_vectors(rt / cfg["gate_subdir"] / "vectors", W)
    model, tok = load_host(cfg, args.device)
    rep = PositionReplacer(model, W)
    patcher = ComponentPatcher(model, layers=sweep, n_heads=n_heads, head_dim=head_dim)

    @torch.no_grad()
    def logp(pids, rids) -> float:
        seq = torch.tensor([list(pids) + list(rids)], device=model.device)
        lg = model(seq, use_cache=False, logits_to_keep=len(rids) + 1).logits[0, :-1]
        tgt = seq[0, len(pids):]
        return sum(float(torch.log_softmax(lg[s:s + 256].float(), -1)
                         .gather(1, tgt[s:s + 256, None])[:, 0].sum())
                   for s in range(0, lg.shape[0], 256))

    def span_targets(sid: str) -> dict:
        tg = {}
        for r in by[sid]:
            v = torch.from_numpy(arr["c3"][int(r["vec"])])
            for q in r["positions"]:
                tg[q] = v
        return tg

    sids = [p["snippet_id"] for p in select_pairs(args, traces) if p["snippet_id"] in by]
    if args.limit:
        sids = sids[:args.limit]
    rows, t0, n_fwd = [], time.time(), 0
    for pi, sid in enumerate(sids):
        tr = traces[sid]; pids, rids = tr["l1b_prompt_ids"], tr["l0_reply_ids"]
        # U reference + cache
        rep.set_targets(None); patcher.set_patches(None); cU = patcher.record()
        logp_U = logp(pids, rids); patcher.stop_recording(); n_fwd += 1
        # S reference + cache
        rep.set_targets(span_targets(sid)); patcher.set_patches(None); cS = patcher.record()
        logp_S = logp(pids, rids); patcher.stop_recording(); n_fwd += 1
        dG_S = logp_S - logp_U
        row = {"snippet_id": sid, "half": half(sid), "logp_U": logp_U, "logp_S": logp_S, "dG_S": dG_S,
               "n_tok": len(pids) + len(rids), "suf": {}, "nec": {}, "self": {}}
        # single-component sufficiency (patch S's component into the U run) and necessity
        for c in comps:
            rep.set_targets(None); patcher.set_patches({c: cS.get(c)})
            row["suf"][c.name] = logp(pids, rids) - logp_U; patcher.assert_written(); n_fwd += 1
            rep.set_targets(span_targets(sid)); patcher.set_patches({c: cU.get(c)})
            row["nec"][c.name] = dG_S - (logp(pids, rids) - logp_U); patcher.assert_written(); n_fwd += 1
        # identity checks: S with c<-S must equal dG_S; ALL components <- S must reproduce dG_S
        seeded = [comps[i] for i in np.random.default_rng(int(cfg["seed"]) + pi).choice(
            len(comps), size=min(4, len(comps)), replace=False)]
        for c in seeded:
            rep.set_targets(span_targets(sid)); patcher.set_patches({c: cS.get(c)})
            row["self"][c.name] = (logp(pids, rids) - logp_U) - dG_S; n_fwd += 1
        rep.set_targets(None); patcher.set_patches({c: cS.get(c) for c in comps})
        row["ALL_gap"] = (logp(pids, rids) - logp_U) - dG_S; patcher.assert_written(); n_fwd += 1
        patcher.set_patches(None); rep.set_targets(None)
        rows.append(row)
        print(f"{TAG} {pi+1}/{len(sids)} {sid} dG_S={dG_S:+.2f} ALL_gap={row['ALL_gap']:+.3f} "
              f"max|SELF|={max(abs(v) for v in row['self'].values()):.3f} · {n_fwd} fwd "
              f"· {(time.time()-t0)/60:.1f} min", flush=True)
        if time.time() - t0 > args.max_hours * 3600:
            print(f"{TAG} wall-clock stop after {len(rows)} items"); break

    # ── joint stage: greedy top-k ranked on the OPPOSITE half, vs a layer-matched random null ──
    names = [c.name for c in comps]
    rank = {}
    for h in (0, 1):
        m = {n: float(np.mean([r["suf"][n] for r in rows if r["half"] == h])) for n in names}
        rank[h] = sorted(names, key=lambda n: -m[n])
    rng = np.random.default_rng(int(cfg["seed"]))
    for r in rows:
        sid = r["snippet_id"]; tr = traces[sid]; pids, rids = tr["l1b_prompt_ids"], tr["l0_reply_ids"]
        # rebuild this item's caches (3 forwards) -- cheaper than holding 60 items of caches
        rep.set_targets(None); patcher.set_patches(None); cU = patcher.record(); lU = logp(pids, rids); patcher.stop_recording()
        rep.set_targets(span_targets(sid)); patcher.set_patches(None); cS = patcher.record(); logp(pids, rids); patcher.stop_recording()
        n_fwd += 2
        top = rank[1 - r["half"]]
        for k in K_LIST:
            if k > len(comps):
                continue
            sel = [Component.from_name(n) for n in top[:k]]
            rep.set_targets(None); patcher.set_patches({c: cS.get(c) for c in sel})
            r[f"dG_TOP{k}"] = logp(pids, rids) - lU; n_fwd += 1
            # layer-matched random: same layer multiset, random heads within those layers
            for j in range(N_RANDOM):
                pick = [Component(c.layer, int(rng.integers(0, n_heads)) if c.head != MLP else MLP) for c in sel]
                rep.set_targets(None); patcher.set_patches({c: cS.get(c) for c in pick})
                r[f"dG_LRAND{k}_{j}"] = logp(pids, rids) - lU; n_fwd += 1
        patcher.set_patches(None); rep.set_targets(None)
    patcher.close(); rep.close()

    (out / "head_rows.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    # ── score ──
    seed = int(cfg["seed"])
    dG = np.array([r["dG_S"] for r in rows])
    ident_ok = all(abs(v) <= IDENT_TOL for r in rows for v in r["self"].values())
    all_ok = bool(np.mean(np.abs([r["ALL_gap"] for r in rows])) <= 0.05 * abs(dG.mean()))
    suf_mean = {n: float(np.mean([r["suf"][n] for r in rows])) for n in names}
    nec_mean = {n: float(np.mean([r["nec"][n] for r in rows])) for n in names}
    st = {"write_layer": W, "sweep_layers": list(sweep), "n_components": len(comps),
          "n_items": len(rows), "n_forwards": n_fwd,
          "ms_per_forward": 1000 * (time.time() - t0) / max(n_fwd, 1),
          "dG_S": boot(dG, seed), "identity_passes": bool(ident_ok and all_ok),
          "max_abs_self": max(abs(v) for r in rows for v in r["self"].values()),
          "mean_abs_ALL_gap": float(np.mean(np.abs([r["ALL_gap"] for r in rows]))),
          "top10_sufficiency": sorted(suf_mean.items(), key=lambda kv: -kv[1])[:10],
          "top10_necessity": sorted(nec_mean.items(), key=lambda kv: -kv[1])[:10],
          "joint": {}, "rank_half0_top10": rank[0][:10], "rank_half1_top10": rank[1][:10]}
    for k in K_LIST:
        if f"dG_TOP{k}" not in rows[0]:
            continue
        top = np.array([r[f"dG_TOP{k}"] for r in rows])
        lr = np.array([np.mean([r[f"dG_LRAND{k}_{j}"] for j in range(N_RANDOM)]) for r in rows])
        st["joint"][str(k)] = {"top": boot(top, seed + k), "layer_matched_random": boot(lr, seed + 100 + k),
                              "recovered_frac": float(top.mean() / dG.mean()) if dG.mean() else float("nan"),
                              "top_minus_random": boot(top - lr, seed + 200 + k)}
    # H-S2 verdict (frozen): CONCENTRATED / DISTRIBUTED / INTERMEDIATE
    j16 = st["joint"].get("16")
    first50 = next((k for k in K_LIST if str(k) in st["joint"]
                    and st["joint"][str(k)]["recovered_frac"] >= 0.50), None)
    if j16 and j16["recovered_frac"] >= 0.50 and j16["top_minus_random"]["mean"] >= 0.30 * dG.mean() \
            and j16["top_minus_random"]["ci95"][0] > 0:
        st["verdict"] = "CONCENTRATED"
    elif first50 is None:
        st["verdict"] = "DISTRIBUTED"
    else:
        st["verdict"] = "INTERMEDIATE"
    st["first_k_reaching_50pct"] = first50
    # H-S3 descriptive: global vs local necessity, and the per-layer depth profile
    g = np.array([nec_mean[n] for n in names if is_global_layer(Component.from_name(n).layer)])
    l = np.array([nec_mean[n] for n in names if not is_global_layer(Component.from_name(n).layer)])
    st["H_S3"] = {"global_mean_nec": float(g.mean()) if len(g) else None,
                  "local_mean_nec": float(l.mean()) if len(l) else None,
                  "n_global": int(len(g)), "n_local": int(len(l)),
                  "spearman_suf_nec": spearman(np.array([suf_mean[n] for n in names]),
                                               np.array([nec_mean[n] for n in names])),
                  "per_layer_summed_nec": {str(L): float(sum(nec_mean[c.name] for c in comps if c.layer == L))
                                           for L in sweep}}
    (out / "head_sweep_stats.json").write_text(json.dumps(st, indent=1))
    print(f"\n{TAG} VERDICT {st['verdict']} · dG_S {st['dG_S']['mean']:+.2f} · identity "
          f"{'PASS' if st['identity_passes'] else 'FAIL'} (max|SELF| {st['max_abs_self']:.3f}, "
          f"ALL gap {st['mean_abs_ALL_gap']:.3f})", flush=True)
    for k, v in st["joint"].items():
        print(f"{TAG}   top-{k:<3} {v['top']['mean']:+8.2f} ({100*v['recovered_frac']:5.1f}% of dG_S) "
              f"· layer-matched random {v['layer_matched_random']['mean']:+8.2f} "
              f"· diff {v['top_minus_random']['mean']:+7.2f} "
              f"[{v['top_minus_random']['ci95'][0]:+.2f},{v['top_minus_random']['ci95'][1]:+.2f}]", flush=True)
    print(f"{TAG} top-5 by necessity: {[n for n,_ in st['top10_necessity'][:5]]}", flush=True)
    print(f"{TAG} H-S3 global nec {st['H_S3']['global_mean_nec']:+.3f} (n={st['H_S3']['n_global']}) vs "
          f"local {st['H_S3']['local_mean_nec']:+.3f} (n={st['H_S3']['n_local']}) · "
          f"rho(suf,nec)={st['H_S3']['spearman_suf_nec']:+.3f}", flush=True)
    return 0 if st["identity_passes"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
