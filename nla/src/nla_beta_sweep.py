"""H-S5..H-S9: can multi-layer NLA steering be made to work on the 33 live Gemma-3-4B-it pairs?

Pre-registered in log/nla-harness/2026-09-12_beta-layerset-prereg.md; rules and thresholds live in
nla/configs/nla_beta_sweep.yaml and are never passed ad-hoc on the CLI (CLAUDE.md §5).

WHY A NEW RUNNER rather than more k in `nla_layer_sweep.py`. Two things that runner could not do:

1. It scored `ARMS = ("c3","swap","edit","random")` -- no `foreign`, no `rt` -- so the SPECIFIC
   component of the NLA edit (`edit - foreign`, content held to the same shape/positions/norm) has
   never been measured at more than one layer. That is the primary quantity here. It matters because
   raw `edit` rewards non-specific decoy destruction: at L2/L3 a norm-matched RANDOM vector already
   scores +16.8/+18.9, and at the raw-`edit`-optimal layer (L1) only ~+2 of ~+36 is specific.

2. `PositionReplacer` had no coefficient, by deliberate design, so "many layers" could only mean
   "overwrite the span position's trajectory k times". Measured effect of that (layer_sweep_stats):
   from k=1 to k=33 the CEILING arm `swap` loses MORE (-14.50) than the NLA arm `c3` does (-11.70),
   i.e. the multi-layer loss was host damage, not channel corruption. `steer.PositionReplacer`'s new
   `beta` interpolates on the unit sphere at fixed norm, so every layer can nudge the position while
   the position keeps computing. beta=1.0 is the banked behaviour, bit-identical.

Readout is `G_sum` (teacher-forced log-prob of the banked L0 reply under the L1b prompt), NOT
accuracy: this host loses 2 of 60 items to obfuscation with 7 flippable, so the accuracy stage is
GATED on the nats result and is pre-declared unable to support a conclusion either way (H-S9).

Reuses the frozen gate wholesale -- banked vectors (`gate/vectors/L{K}.npz`, 34 layers, uniform
60 items / 471 spans), `select_pairs`, the liveness rules, the arm definitions. No AV/AR is loaded,
no re-extraction, host only. Layer RANKINGS come from the banked `gate/gate_rows.jsonl` (they are
beta=1 single-layer numbers already measured) and 3 layers per half are re-verified by a fresh
forward as identity gate 1.
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

TAG = "[BSWEEP]"
CORE_ARMS = ("edit", "foreign", "rt", "c3", "swap")   # every cell
VEC_KEY = {"c3": "c3", "swap": "h0", "edit": "edit", "foreign": "foreign", "rt": "rt"}


def half(sid: str) -> int:
    return zlib.crc32(sid.encode()) & 1


def boot_mean(x: np.ndarray, seed: int, n_boot: int) -> dict:
    """Cluster bootstrap over items (the unit of analysis), as in every entry in this family."""
    x = np.asarray(x, dtype=float)
    if not len(x):
        return {"mean": float("nan"), "ci95": [float("nan")] * 2, "n": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    b = np.sort(x[idx].mean(1))
    return {"mean": float(x.mean()),
            "ci95": [float(b[int(.025 * n_boot)]), float(b[int(.975 * n_boot)])],
            "n": int(len(x))}


# ── layer sets ────────────────────────────────────────────────────────────────────────────────
def spaced_set(live: list[int], k: int) -> list[int]:
    """k live layers evenly spaced over the live range -- a SELECTION-FREE control."""
    if k <= 1:
        return [live[len(live) // 2]]
    return sorted({live[round(i * (len(live) - 1) / (k - 1))] for i in range(k)})


def coverage_set(ed: dict[int, set], live: list[int], k: int) -> list[int]:
    """Greedy maximisation of the union of EDITABLE spans.

    Selected on span METADATA only (`editable` in the banked L{K}_spans.jsonl), never on any dG, so
    this is not outcome selection. It is the set H-S5's mechanism argues for: `v_edit == v_rt` on a
    non-editable span, and which spans are editable differs by layer (pairwise Jaccard 0.63-0.81),
    so k layers can carry a real edit on more spans than any one layer can.
    """
    chosen: list[int] = []
    have: set = set()
    for _ in range(min(k, len(live))):
        best = max((K for K in live if K not in chosen),
                   key=lambda K: (len(have | ed.get(K, set())), -K))
        chosen.append(best)
        have |= ed.get(best, set())
    return sorted(chosen)


def resolve_sets(cfg: dict, live: list[int], ed: dict[int, set]) -> dict[str, dict]:
    """Named sets from the config. A `selected` set is resolved per item from the OTHER half's rank."""
    out: dict[str, dict] = {}
    for name, spec in cfg["sets"].items():
        if spec == "all":
            out[name] = {"layers": list(live), "selected": False}
        elif spec == "spaced12":
            out[name] = {"layers": spaced_set(live, 12), "selected": False}
        elif spec == "coverage12":
            out[name] = {"layers": coverage_set(ed, live, 12), "selected": False}
        elif spec == "spec_rank12":
            # split-half: the layers depend on which half the item is in, so carried per half
            out[name] = {"layers": None, "selected": True, "k": 12, "by": "spec"}
        else:
            out[name] = {"layers": [K for K in spec if K in live], "selected": False}
    return out


def main() -> int:  # noqa: C901
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(_PROJ / "nla/configs/nla_beta_sweep.yaml"))
    ap.add_argument("--root", default=None)
    ap.add_argument("--traces", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--stage", default="grid", choices=("grid", "score"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-hours", type=float, default=3.0)
    args = ap.parse_args()

    cfg = load_cfg(Path(args.config))
    if "qwen" in cfg["train"]["host"]["model_id"].lower():
        print(f"{TAG} REFUSED: Chinese models are not run in this project.")
        return 2
    # select_pairs() reads these off args -- the gate's own item selector, reused verbatim so the
    # item set is identical to every other run in this family.
    args._cfg = cfg
    args.repair = bool(cfg["repair"])
    args.max_spans_per_item = int(cfg["max_spans_per_item"])

    rt_dir = root(cfg, args.root)
    args.traces = args.traces or str(_PROJ / "data/nla/p0/trace_llr/gemma4b/traces.jsonl")
    n_layers = int(cfg["train"]["host"]["n_layers"])
    d_model = int(cfg["train"]["host"]["d_model"])
    vdir = rt_dir / cfg["gate_subdir"] / "vectors"
    out = Path(args.out or (rt_dir / cfg["out_subdir"]))
    out.mkdir(parents=True, exist_ok=True)
    seed = int(cfg["seed"])
    n_boot = int(cfg["n_boot"])
    betas = [float(b) for b in cfg["beta_ladder"]]
    k_ladder = [int(k) for k in cfg["k_ladder"]]
    ident = cfg["identity"]

    live = sorted(K for K in range(n_layers)
                  if liveness(rt_dir, K, cfg)["live"] and (vdir / f"L{K}.npz").exists())
    print(f"{TAG} live layers ({len(live)}): {live}", flush=True)

    traces = {t["snippet_id"]: t for t in map(json.loads, open(args.traces))}
    vec = {K: load_vectors(vdir, K) for K in live}
    ref = vec[live[0]][0]
    for K in live[1:]:                      # the gate's cross-layer anchoring assertion
        assert set(vec[K][0]) == set(ref), f"L{K}: item set differs from L{live[0]}"

    # editable-span sets, from the banked span sidecars (metadata, not outcomes)
    ed = {K: {(sid, r["span_i"]) for sid, rs in vec[K][0].items() for r in rs if r.get("editable")}
          for K in live}
    n_span_total = sum(len(rs) for rs in vec[live[0]][0].values())

    # ── layer rankings from the BANKED beta=1 single-layer gate rows (no forwards) ──────────────
    grows = [json.loads(l) for l in open(rt_dir / cfg["gate_subdir"] / "gate_rows.jsonl")]
    gby = {r["snippet_id"]: r for r in grows}
    rank: dict[str, dict[int, list[int]]] = {"spec": {}, "c3": {}}
    for hh in (0, 1):
        rs = [r for r in grows if half(r["snippet_id"]) == hh]
        spec_mean = {K: float(np.mean([r[f"dG_S_edit_L{K}"] - r[f"dG_S_foreign_L{K}"] for r in rs]))
                     for K in live}
        c3_mean = {K: float(np.mean([r[f"dG_S_c3_L{K}"] for r in rs])) for K in live}
        rank["spec"][hh] = sorted(live, key=lambda K: -spec_mean[K])
        rank["c3"][hh] = sorted(live, key=lambda K: -c3_mean[K])
    print(f"{TAG} SPEC rank half0 {rank['spec'][0][:8]}... half1 {rank['spec'][1][:8]}...", flush=True)
    print(f"{TAG} c3   rank half0 {rank['c3'][0][:8]}... half1 {rank['c3'][1][:8]}...", flush=True)

    named = resolve_sets(cfg, live, ed)
    for name, d in named.items():
        if d["layers"] is not None:
            u = set()
            for K in d["layers"]:
                u |= ed.get(K, set())
            print(f"{TAG} set {name}: k={len(d['layers'])} {d['layers']} · editable-union {len(u)}",
                  flush=True)

    model, _tok = load_host(cfg, args.device)
    reps = {K: PositionReplacer(model, K) for K in live}

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
        """The gate's arm definitions verbatim (nla_ml_gate.targets_for)."""
        by, arr = vec[K]
        tg: dict[int, torch.Tensor] = {}
        for r in by[sid]:
            i = int(r["vec"])
            if kind == "self":
                a, b = r["self_range"]
                for q, j in zip(r["positions"], range(a, b)):
                    tg[q] = torch.from_numpy(arr["self_all"][j])
                continue
            if kind == "random":
                rng = np.random.default_rng(
                    seed + zlib.crc32(f"{sid}#{r['span_i']}#L{K}#rnd".encode()))
                v = rng.standard_normal(d_model).astype(np.float32)
                v = torch.from_numpy(v / (np.linalg.norm(v) + 1e-12))
            else:
                v = torch.from_numpy(arr[VEC_KEY[kind]][i])
            for q in r["positions"]:
                tg[q] = v
        return tg

    def run(Ks, sid, kind: str, beta: float, pids, rids) -> float:
        n = 0
        for K, r in reps.items():
            t = targets(K, sid, kind) if K in Ks else None
            r.set_beta(beta)
            r.set_targets(t)
            n += len(t or {})
        return logp(pids, rids, n)

    # ── the cell plan (frozen shape; see the prereg's Cost section) ─────────────────────────────
    # Kept explicit rather than a full cross product: `self` is an identity check, not a measurement,
    # and `random` only earns its forwards where host damage is the question (the k-ladder and the
    # two all-layer sets). That is what keeps the grid at ~27 k forwards / ~17 min.
    bl = betas if not args.smoke else betas[:1] + [betas[-1]]
    kl = k_ladder if not args.smoke else k_ladder[:2]
    # live_no_tail is kept in the smoke set so H-S6 (the ceiling rule) is actually exercised
    # rather than returning NOT-REPAIRED for want of its cells.
    nm = list(named) if not args.smoke else ["faithful", "live_no_tail", "all_live"]
    RANDOM_SETS = {"all_live", "live_no_tail"}
    SELF_SETS = {"all_live", "live_no_tail"}
    verify_layers = sorted({K for hh in (0, 1)
                            for K in rank["spec"][hh][:int(ident["verify_layers_per_half"])]})

    sids = [p["snippet_id"] for p in select_pairs(args, traces) if p["snippet_id"] in ref]
    if args.limit:
        sids = sids[:args.limit]
    print(f"{TAG} {len(sids)} items · betas {bl} · k {kl} · sets {nm}", flush=True)

    t0 = time.time()
    n_fwd = 0
    rows: list[dict] = []
    for sid in sids:
        tr = traces[sid]
        pids, rids = tr["l1b_prompt_ids"], tr["l0_reply_ids"]
        hh = half(sid)
        other = 1 - hh
        row: dict = {"snippet_id": sid, "half": hh, "n_tok": len(pids) + len(rids)}
        for r in reps.values():
            r.set_targets(None)
        base = logp(pids, rids, 0)
        n_fwd += 1
        row["logp_U"] = base

        def cell(tag: str, Ks, kind: str, beta: float) -> None:
            nonlocal n_fwd
            row[f"dG_{kind}_{tag}"] = run(set(Ks), sid, kind, beta, pids, rids) - base
            n_fwd += 1

        # H-S5: SPEC-ranked top-k ladder, selected on the OTHER half, across the beta ladder
        for k in kl:
            Ks = rank["spec"][other][:k]
            row[f"layers_topk{k}"] = sorted(Ks)
            for b in bl:
                tag = f"topk{k}_b{b}"
                for arm in CORE_ARMS:
                    cell(tag, Ks, arm, b)
                if k == max(kl):
                    cell(tag, Ks, "random", b)

        # H-S6/H-S7/H-S8: the named sets across the beta ladder
        for name in nm:
            d = named[name]
            Ks = rank["spec"][other][:d["k"]] if d["layers"] is None else d["layers"]
            row[f"layers_{name}"] = sorted(Ks)
            for b in bl:
                tag = f"{name}_b{b}"
                for arm in CORE_ARMS:
                    cell(tag, Ks, arm, b)
                if name in RANDOM_SETS:
                    cell(tag, Ks, "random", b)
                if name in SELF_SETS and b in (bl[0], bl[-1]):
                    cell(tag, Ks, "self", b)

        # H-S7: the c3-ranked top-1, the layer H-S1 would have picked
        for nm2, which in (("c3rank1", "c3"), ("specrank1", "spec")):
            Ks = rank[which][other][:1]
            row[f"layers_{nm2}"] = sorted(Ks)
            for arm in ("edit", "foreign"):
                cell(f"{nm2}_b1.0", Ks, arm, 1.0)

        # identity gate 1: beta=1 single-layer cells must reproduce the banked gate rows
        for K in verify_layers:
            for arm in ("edit", "foreign"):
                cell(f"verifyL{K}_b1.0", [K], arm, 1.0)
        # identity gate 2: beta=0 is a VALUE no-op that still writes every position
        for arm in CORE_ARMS:
            cell("beta0_faithful", named["faithful"]["layers"], arm, 0.0)

        rows.append(row)
        el = (time.time() - t0) / 60
        print(f"{TAG} {len(rows)}/{len(sids)} {sid} T={row['n_tok']} · {n_fwd} fwd · {el:.1f} min "
              f"· {1000 * (time.time() - t0) / max(1, n_fwd):.0f} ms/fwd", flush=True)
        if time.time() - t0 > args.max_hours * 3600:
            print(f"{TAG} wall-clock stop after {len(rows)} items", flush=True)
            break
    for r in reps.values():
        r.close()

    (out / "beta_rows.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    print(f"{TAG} wrote {out / 'beta_rows.jsonl'} · {n_fwd} forwards · "
          f"{(time.time() - t0) / 60:.1f} min", flush=True)

    # ── score ──────────────────────────────────────────────────────────────────────────────────
    stats = score(rows, cfg, live, named, rank, ed, gby, verify_layers, bl, kl, nm, n_span_total)
    stats["n_forwards"] = n_fwd
    stats["ms_per_forward"] = 1000 * (time.time() - t0) / max(1, n_fwd)
    (out / "beta_sweep_stats.json").write_text(json.dumps(stats, indent=1))
    print(f"{TAG} wrote {out / 'beta_sweep_stats.json'}", flush=True)
    for line in stats["summary"]:
        print(f"{TAG} {line}", flush=True)
    if not stats["identity_passes"]:
        print(f"{TAG} IDENTITY FAILED -> S-HARNESS-FAULT; results are not reportable")
        return 3
    return 0


def score(rows, cfg, live, named, rank, ed, gby, verify_layers, bl, kl, nm, n_span_total) -> dict:
    """All frozen rules from nla/configs/nla_beta_sweep.yaml, applied once, in one place."""
    seed = int(cfg["seed"])
    n_boot = int(cfg["n_boot"])
    ident = cfg["identity"]
    rules = cfg["rules"]
    B = lambda v, s=0: boot_mean(np.asarray(v, float), seed + s, n_boot)   # noqa: E731

    def spec(tag: str) -> list[float]:
        """The primary contrast: content-specific effect, paired per item."""
        return [r[f"dG_edit_{tag}"] - r[f"dG_foreign_{tag}"] for r in rows
                if f"dG_edit_{tag}" in r and f"dG_foreign_{tag}" in r]

    def arm(tag: str, a: str) -> list[float]:
        return [r[f"dG_{a}_{tag}"] for r in rows if f"dG_{a}_{tag}" in r]

    def paired(tag_a: str, tag_b: str, fn=spec) -> dict:
        A = fn(tag_a)
        Bv = fn(tag_b)
        if len(A) != len(Bv) or not A:
            return {"mean": float("nan"), "ci95": [float("nan")] * 2, "n": 0}
        return B([x - y for x, y in zip(A, Bv)], 7)

    st: dict = {"experiment": cfg["experiment"], "seed": seed, "n_boot": n_boot,
                "n_items": len(rows), "config_sha256": cfg.get("_sha256"),
                "live_layers": live, "beta_ladder": bl, "k_ladder": kl,
                "sets": {k: (rows[0].get(f"layers_{k}") if rows else None) for k in nm},
                "n_span_total": n_span_total, "summary": []}

    # ── identity gates ─────────────────────────────────────────────────────────────────────────
    idb: dict = {"beta1_vs_banked": {}, "beta0": {}, "self": {}}
    ok = True
    for K in verify_layers:
        for a in ("edit", "foreign"):
            got = [r[f"dG_{a}_verifyL{K}_b1.0"] for r in rows if f"dG_{a}_verifyL{K}_b1.0" in r]
            want = [gby[r["snippet_id"]][f"dG_S_{a}_L{K}"] for r in rows
                    if f"dG_{a}_verifyL{K}_b1.0" in r]
            d = float(np.max(np.abs(np.array(got) - np.array(want)))) if got else float("nan")
            p = bool(d <= float(ident["beta1_tol_nats"]))
            idb["beta1_vs_banked"][f"L{K}_{a}"] = {"max_abs_diff": d, "passes": p}
            ok &= p
    for a in CORE_ARMS:
        v = arm("beta0_faithful", a)
        d = float(np.max(np.abs(v))) if v else float("nan")
        p = bool(d <= float(ident["beta0_tol_nats"]))
        idb["beta0"][a] = {"max_abs_dG": d, "passes": p}
        ok &= p
    for name in ("all_live", "live_no_tail"):
        for b in (bl[0], bl[-1]):
            v = arm(f"{name}_b{b}", "self")
            if v:
                d = float(np.max(np.abs(v)))
                p = bool(d <= float(ident["self_tol_nats"]))
                idb["self"][f"{name}_b{b}"] = {"max_abs_dG": d, "passes": p}
                ok &= p
    st["identity"] = idb
    st["identity_passes"] = bool(ok)
    st["reportable"] = bool(ok)

    # ── the beta x k surface, and per-cell tables ──────────────────────────────────────────────
    surf: dict = {}
    for k in kl:
        surf[str(k)] = {}
        for b in bl:
            tag = f"topk{k}_b{b}"
            surf[str(k)][str(b)] = {"spec": B(spec(tag), 1),
                                    **{a: B(arm(tag, a), 2) for a in CORE_ARMS}}
    st["surface_topk"] = surf
    st["sets_table"] = {name: {str(b): {"spec": B(spec(f"{name}_b{b}"), 3),
                                        **{a: B(arm(f"{name}_b{b}", a), 4) for a in CORE_ARMS}}
                               for b in bl} for name in nm}

    # ── H-S5: does the SPECIFIC effect add across layers? ──────────────────────────────────────
    # beta is a new degree of freedom, so the per-k best beta is chosen on the OTHER half and the
    # value is read on this one -- beta is never selected and scored on the same items.
    def best_beta_other_half(tag_fn) -> tuple[dict, dict]:
        vals: list[float] = []
        picks: dict[str, float] = {}
        for r in rows:
            hh = r["half"]
            cand = {}
            for b in bl:
                t = tag_fn(b)
                o = [x[f"dG_edit_{t}"] - x[f"dG_foreign_{t}"] for x in rows
                     if x["half"] != hh and f"dG_edit_{t}" in x]
                if o:
                    cand[b] = float(np.mean(o))
            if not cand:
                continue
            bb = max(cand, key=cand.get)
            picks[r["snippet_id"]] = bb
            t = tag_fn(bb)
            vals.append(r[f"dG_edit_{t}"] - r[f"dG_foreign_{t}"])
        return B(vals, 5), picks

    hs5: dict = {"per_k": {}, "rule": {"additive": float(rules["hs5_additive_ratio"]),
                                       "saturating": float(rules["hs5_saturating_ratio"])}}
    per_k_vals: dict[int, list[float]] = {}
    for k in kl:
        res, picks = best_beta_other_half(lambda b, k=k: f"topk{k}_b{b}")
        u = set()
        for K in (rows[0].get(f"layers_topk{k}") or []):
            u |= ed.get(K, set())
        hs5["per_k"][str(k)] = {"spec_at_best_beta": res, "beta_picks": sorted(set(picks.values())),
                                "editable_union": len(u)}
        # re-derive the per-item series at each item's other-half beta, for the paired contrast
        series = []
        for r in rows:
            bb = picks.get(r["snippet_id"])
            if bb is None:
                continue
            t = f"topk{k}_b{bb}"
            series.append(r[f"dG_edit_{t}"] - r[f"dG_foreign_{t}"])
        per_k_vals[k] = series
    k1 = per_k_vals.get(kl[0], [])
    best_k = max((k for k in kl if per_k_vals.get(k)),
                 key=lambda k: float(np.mean(per_k_vals[k])), default=kl[0])
    ratio = (float(np.mean(per_k_vals[best_k])) / float(np.mean(k1))) if k1 and np.mean(k1) else float("nan")
    dpair = B([a - b for a, b in zip(per_k_vals[best_k], k1)], 6) if k1 else B([], 6)
    clears = bool(dpair["ci95"][0] > 0 or dpair["ci95"][1] < 0) and dpair["mean"] > 0
    hs5.update({"best_k": int(best_k), "ratio_best_over_k1": ratio,
                "paired_best_minus_k1": dpair,
                "union_ratio_prediction": (hs5["per_k"][str(best_k)]["editable_union"] /
                                           max(1, hs5["per_k"][str(kl[0])]["editable_union"]))})
    if clears and ratio >= float(rules["hs5_additive_ratio"]):
        hs5["verdict"] = "SPEC-ADDITIVE"
    elif clears and ratio >= float(rules["hs5_saturating_ratio"]):
        hs5["verdict"] = "SPEC-SATURATING"
    else:
        hs5["verdict"] = "SPEC-NON-ADDITIVE"
    st["H_S5"] = hs5
    st["summary"].append(f"H-S5 {hs5['verdict']}: best k={best_k} ratio {ratio:.3f} "
                         f"(union predicts {hs5['union_ratio_prediction']:.2f}), paired "
                         f"{dpair['mean']:+.2f} {dpair['ci95']}")

    # ── H-S6: does beta repair the ceiling that multi-layer replacement breaks? ─────────────────
    ceil1 = float(rules["hs6_ceiling_k1_nats"])
    want = float(rules["hs6_repaired_frac"]) * ceil1
    sw = {str(b): B(arm(f"live_no_tail_b{b}", "swap"), 8) for b in bl}
    cand = {b: sw[str(b)]["mean"] for b in bl if not np.isnan(sw[str(b)]["mean"])}
    bb = max(cand, key=cand.get) if cand else None
    b1 = sw[str(bl[-1])]["mean"] if str(bl[-1]) in sw else float("nan")
    rep_ok = bool(bb is not None and cand[bb] >= want and sw[str(bb)]["ci95"][0] > b1)
    st["H_S6"] = {"swap_by_beta": sw, "best_beta": bb, "ceiling_k1": ceil1,
                  "required": want, "beta1_value": b1,
                  "verdict": "CEILING-REPAIRED" if rep_ok else "NOT-REPAIRED",
                  "spec_by_beta": {str(b): B(spec(f"live_no_tail_b{b}"), 9) for b in bl}}
    st["summary"].append(f"H-S6 {st['H_S6']['verdict']}: best beta={bb} swap "
                         f"{cand.get(bb, float('nan')):+.2f} vs required {want:+.2f} "
                         f"(beta=1 gives {b1:+.2f})")

    # ── H-S7: was the SELECTION OBJECTIVE worth anything? ──────────────────────────────────────
    d7 = paired("specrank1_b1.0", "c3rank1_b1.0")
    ok7 = bool(d7["mean"] >= float(rules["hs7_min_nats"]) and d7["ci95"][0] > 0)
    st["H_S7"] = {"spec_minus_c3_selection": d7,
                  "layers": {"spec": rows[0].get("layers_specrank1") if rows else None,
                             "c3": rows[0].get("layers_c3rank1") if rows else None},
                  "min_nats": float(rules["hs7_min_nats"]),
                  "verdict": "SELECTION-MATTERS" if ok7 else "SELECTION-IMMATERIAL"}
    st["summary"].append(f"H-S7 {st['H_S7']['verdict']}: {d7['mean']:+.2f} {d7['ci95']}")

    # ── H-S8: layer-set shape at matched k (descriptive, no verdict word) ──────────────────────
    shape = {}
    for name in nm:
        cells = {b: B(spec(f"{name}_b{b}"), 10) for b in bl}
        bbn = max(cells, key=lambda b: cells[b]["mean"]) if cells else None
        u = set()
        for K in (rows[0].get(f"layers_{name}") or []):
            u |= ed.get(K, set())
        shape[name] = {"k": len(rows[0].get(f"layers_{name}") or []) if rows else 0,
                       "editable_union": len(u), "best_beta": bbn,
                       "spec_at_best_beta": cells.get(bbn)}
    st["H_S8"] = shape
    ranked = sorted((v["spec_at_best_beta"]["mean"], k) for k, v in shape.items()
                    if v["spec_at_best_beta"])
    if ranked:
        st["summary"].append("H-S8 sets by SPEC: " +
                             " · ".join(f"{k} {m:+.2f}" for m, k in reversed(ranked)))

    # ── H-S9 gate (the accuracy stage runs only if this fires) ─────────────────────────────────
    singles = [B(spec(f"{nm2}_b1.0"), 11)["mean"] for nm2 in ("specrank1", "c3rank1")]
    best_single = max([s for s in singles if not np.isnan(s)], default=float("nan"))
    allc = [(v["spec_at_best_beta"]["mean"], f"set:{k}") for k, v in shape.items()
            if v["spec_at_best_beta"]]
    allc += [(hs5["per_k"][str(k)]["spec_at_best_beta"]["mean"], f"topk{k}") for k in kl
             if hs5["per_k"][str(k)]["spec_at_best_beta"]]
    bestcfg = max(allc, default=(float("nan"), None))
    gap = bestcfg[0] - best_single
    fires = bool(gap >= float(rules["hs9_gate_min_nats"]))
    st["H_S9_gate"] = {"best_config": bestcfg[1], "best_config_spec": bestcfg[0],
                       "best_single_layer_spec": best_single, "gap": gap,
                       "min_nats": float(rules["hs9_gate_min_nats"]), "fires": fires}
    st["summary"].append(f"H-S9 gate {'FIRES' if fires else 'does NOT fire'}: best "
                         f"{bestcfg[1]} {bestcfg[0]:+.2f} vs best single {best_single:+.2f} "
                         f"(gap {gap:+.2f}, need {rules['hs9_gate_min_nats']:+.2f})")
    st["summary"].append(f"identity {'PASS' if ok else 'FAIL'}")
    return st


if __name__ == "__main__":
    raise SystemExit(main())
