"""H-W31 — which downstream heads and MLPs carry the NLA-transported clean state?

Pre-registered: `log/nla-harness/2026-09-07_head-mediation-prereg.md`; frozen thresholds in
`nla/configs/nla_heads.yaml`.

The W family showed that writing AR(AV(h_L0)) at the L1b identifier spans of layer 32 repairs the
teacher-forced L0 reply by +44.94 G_sum nats (98.3 % of the raw clean state's +45.71). The write
lands on PROMPT positions and G_sum is scored on REPLY tokens, so every nat crosses positions
through at least one attention head in layers 33..47. This script maps which ones, by activation
patching between two runs of the same sequence:

    U   unsteered L1b run            S   same run with the arm's vector written at the spans
    suf_c  = dG(U, c <- S)           what one component carries forward on its own
    nec_c  = dG(S) - dG(S, c <- U)   what is lost when one component is blind to the write
    read_h = logp_mref[L] - logp(S, head h at L cannot attend to the span keys)

followed by a joint pass (top-k by `suf`, selected on the OTHER split half) against uniform and
layer-matched random-k nulls, and two identities that gate everything: SELF_c (S patched with its
own cache) and ALL (every component of 33..47 from S into U must reproduce S).

Three stages, so the sweep never holds the AV/AR in memory:

    --stage vectors   host + AV + AR: per-span h_L0 and AR(AV(h_L0)) on the repaired anchoring,
                      persisted to vectors.npz / spans.jsonl (H-W25's fidelity falls out of them)
    --stage sweep     host only: references, 255 x (suf, nec), SELF, ALL, optional read knockout
    --stage joint     host only: split-half top-k, RAND_k, LRAND_k, ALL

Every row carries `n_pos_*` stamps (arm_guard) and every patched forward asserts the number of
components it actually overwrote, so an arm that silently did nothing is refused, not averaged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
import zlib
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import yaml

_HERE = Path(__file__).resolve().parent
_NLA_ROOT = _HERE.parent
_PROJ = _NLA_ROOT.parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_NLA_ROOT / "vendor" / "nla-repo"))

from extract import ActivationExtractor  # noqa: E402
from span_positions import span_token_positions  # noqa: E402
from steer import PositionReplacer  # noqa: E402
from steer_run import HOSTS, AR_CHECKPOINTS, build_user, load_pairs, SEED  # noqa: E402
from nla_cycle import AV_CHECKPOINTS, MAX_NEW_READ  # noqa: E402
from nla_writeback import term_spans  # noqa: E402
from arm_guard import ArmNotWritten, paired, record_positions  # noqa: E402
from repair_pairs import recover  # noqa: E402
from head_patch import (AttentionKnockout, Component, ComponentPatcher, MLP,  # noqa: E402
                        SWEEP_LAYERS, components, draw_layer_matched, draw_uniform, half_of,
                        is_global_layer, rank_on_other_half)

EXPERIMENT = "W31_head_localisation"
ARMS = ("C3pure", "P_patch")
NULL_ARMS = ("N_sibling", "N_foreign", "N_random")   # H-W35, built from banked vectors
CFG_PATH = _NLA_ROOT / "configs" / "nla_heads.yaml"
TAG = "[W31]"


def load_cfg(path: Path = CFG_PATH) -> dict:
    cfg = yaml.safe_load(open(path))
    cfg["_sha256"] = hashlib.sha256(open(path, "rb").read()).hexdigest()
    return cfg


def _layer_range(spec: str | list) -> tuple[int, ...]:
    if isinstance(spec, str):
        a, b = spec.split("-")
        return tuple(range(int(a), int(b) + 1))
    a, b = spec
    return tuple(range(int(a), int(b) + 1))


# ── stage 0: vectors ──────────────────────────────────────────────────────────

def stage_vectors(args: argparse.Namespace, cfg: dict, out: Path) -> int:
    """Per-span h_L0 and AR(AV(h_L0)) on the repaired anchoring, persisted for the later stages.

    Replicates nla_fidelity.py's C3pure / P_patch construction with nla_tiers.py's `--repair`
    anchoring (recover() consulted only where the recorded rename_map wrote a '?unpaired'
    sentinel). Runs the host + AV + AR together (~72 GB), the reason this is its own stage.
    """
    from nla_inference import NLACritic  # noqa: E402
    from local_av import LocalAV  # noqa: E402

    t0 = time.time()
    model_name, layer = HOSTS[args.model]
    trc = Path(args.traces or _PROJ / f"data/nla/p0/trace_llr/{args.model}/traces.jsonl")
    if not trc.exists():
        print(f"{TAG} REFUSED: missing prerequisite {trc}"); return 2
    traces = {t["snippet_id"]: t for t in map(json.loads, open(trc))}

    ex = ActivationExtractor(model_name, layer, device=args.device)
    tokz = ex.tokenizer
    av = LocalAV(AV_CHECKPOINTS[args.model], device=args.device)
    ar = NLACritic(AR_CHECKPOINTS[args.model], device=args.device)

    pairs = load_pairs(args.limit or (3 if args.smoke else None), random.Random(SEED))
    if args.smoke:
        pairs = pairs[:3]
    pairs = [p for p in pairs if p["snippet_id"] in traces]

    spans_out, h0s, c3s = [], [], []
    diag = {"spans": 0, "no_true": 0, "no_occ": 0, "no_pos": 0, "items": 0}
    for pi, p in enumerate(pairs):
        if time.time() - t0 > args.max_hours * 3600:
            diag["wall_clock_stop"] = diag.get("wall_clock_stop", 0) + 1; break
        sid = p["snippet_id"]
        user1 = build_user(p["code_l1b"], p["call_l1b"])
        user0 = build_user(p["code_l0"], p["call_l0"])
        inv = {v: k for k, v in (p["rename_map"] or {}).items()}
        lang = p.get("language") or "python"
        rec_inv = ({v: k for k, v in recover(p["code_l0"], p["code_l1b"], lang)[0].items()}
                   if args.repair else {})
        spans = (p.get("id_spans_l1b") or [])[: args.max_spans_per_item]
        if not spans:
            continue
        full0 = ex.extract_chat(user0, positions=None, text_id=sid)
        l0_cache: dict[str, np.ndarray] = {}
        n_item = 0
        for si, sp in enumerate(spans):
            diag["spans"] += 1
            decoy = p["code_l1b"][int(sp[0]):int(sp[1])].strip()
            true = inv.get(decoy)
            if (not true or str(true).startswith("?unpaired")) and args.repair:
                true = rec_inv.get(decoy)
            if not true or str(true).startswith("?unpaired"):
                diag["no_true"] += 1; continue
            occ = term_spans(p["code_l0"], true)
            if not occ:
                diag["no_occ"] += 1; continue
            pos, _ = span_token_positions(tokz, user1, p["code_l1b"], [sp])
            if not pos or max(pos) >= len(traces[sid]["l1b_prompt_ids"]):
                diag["no_pos"] += 1; continue
            if true not in l0_cache:
                pos0, _ = span_token_positions(tokz, user0, p["code_l0"], [occ[0]])
                if not pos0 or max(pos0) >= len(full0.activations):
                    diag["no_pos"] += 1; continue
                l0_cache[true] = full0.activations[max(pos0)].astype(np.float32)
            h0 = l0_cache[true]
            read = av.generate(h0, temperature=0.0, max_new_tokens=MAX_NEW_READ)
            c3 = ar.reconstruct(read).numpy().astype(np.float32)
            spans_out.append({"snippet_id": sid, "span_i": si, "decoy": decoy, "true": true,
                              "positions": [int(q) for q in pos], "read": read,
                              "vec": len(h0s)})
            h0s.append(h0); c3s.append(c3); n_item += 1
        diag["items"] += int(n_item > 0)
        print(f"{TAG} vectors {pi+1}/{len(pairs)} {sid} spans={n_item}", flush=True)

    if not spans_out:
        print(f"{TAG} REFUSED: no span produced a vector"); return 2
    np.savez(out / "vectors.npz", h0=np.stack(h0s), c3=np.stack(c3s))
    with open(out / "spans.jsonl", "w") as f:
        for r in spans_out:
            f.write(json.dumps(r) + "\n")
    cos = [float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
           for a, b in zip(h0s, c3s)]
    manifest = {"experiment": EXPERIMENT, "stage": "vectors", "seed": SEED, "host": model_name,
                "layer": layer, "av": str(AV_CHECKPOINTS[args.model]),
                "ar": str(AR_CHECKPOINTS[args.model]), "repair": bool(args.repair),
                "max_spans_per_item": args.max_spans_per_item, "smoke": bool(args.smoke),
                "n_spans": len(spans_out), "n_items": diag["items"], "diag": diag,
                "cos_h0_c3_mean": float(np.mean(cos)), "config_sha256": cfg["_sha256"],
                "elapsed_s": time.time() - t0}
    json.dump(manifest, open(out / "vectors_manifest.json", "w"), indent=2)
    print(f"{TAG} vectors done: {len(spans_out)} spans / {diag['items']} items, "
          f"mean cos(h0,c3)={np.mean(cos):.3f}, diag={diag}", flush=True)
    return 0


def load_spans(out: Path) -> tuple[dict[str, list[dict]], np.ndarray, np.ndarray]:
    sp, vp = out / "spans.jsonl", out / "vectors.npz"
    if not sp.exists() or not vp.exists():
        raise FileNotFoundError(f"run --stage vectors first ({sp}, {vp})")
    by_item: dict[str, list[dict]] = {}
    for r in map(json.loads, open(sp)):
        by_item.setdefault(r["snippet_id"], []).append(r)
    z = np.load(vp)
    return by_item, z["h0"], z["c3"]


def item_targets(spans: list[dict], h0: np.ndarray, c3: np.ndarray,
                 arms: Sequence[str] = ARMS, pool: list[dict] | None = None,
                 ) -> dict[str, dict[int, torch.Tensor]]:
    """Per-arm {position: vector} for one item.

    H-W35's null arms write DIFFERENT content at the SAME positions, so the route can be compared
    against the correct-content arms. All three come from the banked `vectors.npz` — no AV/AR:

      N_sibling  another span OF THE SAME ITEM      (right item, wrong span; banked whole-effect
                                                     87.4 % of P_patch)
      N_foreign  a span from a DIFFERENT item       (content, wrong item; 33.2 %)
      N_random   a random unit direction            (no content; the operator alone, −365.59)

    Draws are seeded per (snippet, span, arm) so the arm is reproducible and independent of corpus
    order, and are refused rather than silently falling back when the draw is impossible — a
    single-span item has no sibling, and `arm_guard` must see that as a missing position, not as a
    position written with the wrong thing.
    """
    tg: dict[str, dict[int, torch.Tensor]] = {a: {} for a in arms}
    n_spans = len(spans)
    for si, r in enumerate(spans):
        vec_self = int(r["vec"])
        for q in r["positions"]:
            if "C3pure" in tg:
                tg["C3pure"][q] = torch.from_numpy(c3[vec_self])
            if "P_patch" in tg:
                tg["P_patch"][q] = torch.from_numpy(h0[vec_self])
            if "N_sibling" in tg and n_spans > 1:
                rng = np.random.default_rng(SEED + zlib.crc32(f"{r['snippet_id']}#{si}#sib".encode()))
                j = int(rng.integers(0, n_spans - 1))
                j = j + 1 if j >= si else j                      # never the span's own vector
                tg["N_sibling"][q] = torch.from_numpy(h0[int(spans[j]["vec"])])
            if "N_foreign" in tg and pool:
                rng = np.random.default_rng(SEED + zlib.crc32(f"{r['snippet_id']}#{si}#for".encode()))
                cand = [p for p in pool if p["snippet_id"] != r["snippet_id"]]
                if cand:
                    tg["N_foreign"][q] = torch.from_numpy(h0[int(cand[int(rng.integers(0, len(cand)))]["vec"])])
            if "N_random" in tg:
                rng = np.random.default_rng(SEED + zlib.crc32(f"{r['snippet_id']}#{si}#rnd".encode()))
                v = rng.standard_normal(h0.shape[1]).astype(np.float32)
                tg["N_random"][q] = torch.from_numpy(v / (np.linalg.norm(v) + 1e-12))
    return {a: t for a, t in tg.items() if t}


class Host:
    """The subject model plus the three hook families, and the one scoring closure."""

    def __init__(self, args: argparse.Namespace, cfg: dict, knockout: bool):
        model_name, layer = HOSTS[args.model]
        self.ex = ActivationExtractor(model_name, layer, device=args.device)
        self.ex.close()                     # drop its per-forward cloning hook on layer 32
        self.model = self.ex.model
        self.sweep_layers = _layer_range(cfg["sweep_layers"])
        self.rep = PositionReplacer(self.model, layer)
        self.patcher = ComponentPatcher(self.model, self.sweep_layers,
                                        n_heads=cfg["n_heads"], head_dim=cfg["head_dim"])
        self.ko = (AttentionKnockout(self.model, self.sweep_layers, n_heads=cfg["n_heads"],
                                     window=cfg["sliding_window"],
                                     global_fn=lambda L: is_global_layer(L, cfg["global_layers_every"]))
                   if knockout else None)
        self.n_forwards = 0
        self.forward_s = 0.0

    @torch.no_grad()
    def logp(self, pids: list[int], rids: list[int], expect_positions: int = 0) -> float:
        t = time.time()
        seq = torch.tensor([list(pids) + list(rids)], device=self.model.device)
        plen = len(pids)
        self.rep.reset(); self.patcher.reset()
        # use_cache=False: the default allocates a ~650 MB DynamicCache per forward for nothing.
        # logits_to_keep: skip the 262k-vocab lm_head over the prompt; identical numbers.
        lg = self.model(seq, use_cache=False, logits_to_keep=len(rids) + 1).logits[0, :-1]
        assert lg.shape[0] == len(rids)
        tgt = seq[0, plen:]; tot = 0.0
        for st in range(0, lg.shape[0], 256):
            tot += float(torch.log_softmax(lg[st:st + 256].float(), -1)
                         .gather(1, tgt[st:st + 256, None])[:, 0].sum())
        self.patcher.assert_written()
        if self.rep.n_positions_written != expect_positions:
            raise RuntimeError(f"PositionReplacer wrote {self.rep.n_positions_written} positions, "
                               f"expected {expect_positions}")
        self.n_forwards += 1; self.forward_s += time.time() - t
        return tot

    def ms_per_forward(self) -> float:
        return 1000.0 * self.forward_s / max(self.n_forwards, 1)


def _select_pairs(args: argparse.Namespace, by_item: dict) -> list[str]:
    pairs = load_pairs(args.limit or (3 if args.smoke else None), random.Random(SEED))
    if args.smoke:
        pairs = pairs[:3]
    return [p["snippet_id"] for p in pairs if p["snippet_id"] in by_item]


# ── stage 1: single-component sweep ───────────────────────────────────────────

def stage_sweep(args: argparse.Namespace, cfg: dict, out: Path) -> int:
    t0 = time.time()
    by_item, h0, c3 = load_spans(out)
    trc = Path(args.traces or _PROJ / f"data/nla/p0/trace_llr/{args.model}/traces.jsonl")
    traces = {t["snippet_id"]: t for t in map(json.loads, open(trc))}
    sids = _select_pairs(args, by_item)
    pool = [r for rs in by_item.values() for r in rs]      # H-W35 foreign draws
    host = Host(args, cfg, knockout=args.read_knockout)
    single_layers = _layer_range(args.layers) if args.layers else (
        _layer_range(cfg["smoke_layers"]) if args.smoke else host.sweep_layers)
    assert set(single_layers) <= set(host.sweep_layers)
    singles = components(single_layers, cfg["n_heads"])
    all_comps = components(host.sweep_layers, cfg["n_heads"])
    tol = float(cfg["identity"]["self_tol_nats"])

    sink = open(out / "heads_rows.jsonl", "w")
    n_rows = 0
    for ii, sid in enumerate(sids):
        if time.time() - t0 > args.max_hours * 3600:
            print(f"{TAG} wall-clock stop after {ii} items", flush=True); break
        tr = traces[sid]
        pids, rids = tr["l1b_prompt_ids"], tr["l0_reply_ids"]
        T = len(pids) + len(rids)
        targets = item_targets(by_item[sid], h0, c3, args.arm_list, pool)
        span_keys = sorted(next(iter(targets.values())))

        host.rep.set_targets(None); host.patcher.set_patches(None)
        cU = host.patcher.record()
        logp_U = host.logp(pids, rids)
        host.patcher.stop_recording()

        for arm in args.arm_list:
            if arm not in targets:
                continue          # arm_guard: a draw that could not be made is absent, not zero
            tg = targets[arm]; n_pos = len(tg)
            host.rep.set_targets(tg); host.patcher.set_patches(None)
            cS = host.patcher.record()
            logp_S = host.logp(pids, rids, n_pos)
            host.patcher.stop_recording()
            dG_S = logp_S - logp_U
            row = {"snippet_id": sid, "arm": arm, "T": T, "n_tok": len(rids),
                   "n_spans": len(by_item[sid]), "half": half_of(sid),
                   "last_span_pos": max(span_keys), "layers": list(single_layers),
                   "logp_U": logp_U, "logp_S": logp_S, "dG_S": dG_S,
                   "suf": {}, "nec": {}, "self": {}, "read": None, "ko_gap": None}
            for c in singles:
                host.rep.set_targets(None)
                host.patcher.set_patches({c: cS.get(c)})
                row["suf"][c.name] = host.logp(pids, rids) - logp_U
                host.rep.set_targets(tg)
                host.patcher.set_patches({c: cU.get(c)})
                row["nec"][c.name] = logp_S - host.logp(pids, rids, n_pos)
            # SELF identities: all components in smoke, a seeded rotating subset in the full run
            if args.smoke or args.self_per_item >= len(singles):
                self_set = list(singles)
            else:
                rng = np.random.default_rng(SEED + zlib.crc32(f"{sid}|{arm}".encode()))
                self_set = draw_uniform(singles, args.self_per_item, rng)
            host.rep.set_targets(tg)
            for c in self_set:
                host.patcher.set_patches({c: cS.get(c)})
                row["self"][c.name] = host.logp(pids, rids, n_pos) - logp_S
            worst = max((abs(v) for v in row["self"].values()), default=0.0)
            if worst > 0.0:
                host.patcher.set_patches(None)
                row["repeat_gap"] = host.logp(pids, rids, n_pos) - logp_S
            # ALL: every component of 33..47 from S into U must reproduce S
            host.rep.set_targets(None)
            all_patches = {c: cS.get(c) for c in all_comps}
            host.patcher.set_patches(all_patches)
            row["ALL_gap"] = host.logp(pids, rids) - logp_S
            host.patcher.set_patches(None)
            # read knockout: head h at L blind to the span keys, against a mask-only reference
            if host.ko is not None:
                row["read"], row["ko_gap"] = {}, {}
                host.rep.set_targets(tg)
                for L in single_layers:
                    host.ko.set(L, None, None, materialize_only=True)
                    mref = host.logp(pids, rids, n_pos)
                    assert host.ko.n_masks_applied == 1
                    row["ko_gap"][str(L)] = mref - logp_S
                    for h in range(cfg["n_heads"]):
                        host.ko.set(L, h, span_keys)
                        row["read"][f"L{L}H{h}"] = mref - host.logp(pids, rids, n_pos)
                        assert host.ko.n_masks_applied == 1
                host.ko.clear()
            host.rep.set_targets(None)
            record_positions(row, {"S": tg, "ALL": all_patches})
            row["ms_per_forward"] = host.ms_per_forward()
            sink.write(json.dumps(row) + "\n"); sink.flush(); n_rows += 1
            best = max(row["suf"].items(), key=lambda kv: kv[1])
            print(f"{TAG} {ii+1}/{len(sids)} {sid} {arm:7s} T={T} pos={n_pos} dG_S={dG_S:+.2f} "
                  f"ALL_gap={row['ALL_gap']:+.3f} SELF_max={worst:.4f} "
                  f"best_suf={best[0]}:{best[1]:+.2f} {host.ms_per_forward():.0f} ms/fwd "
                  f"({host.n_forwards} fwd, {(time.time()-t0)/60:.1f} min)", flush=True)
            if worst > tol or abs(row["ALL_gap"]) > max(tol, cfg["identity"]["all_tol_frac"] * abs(dG_S)):
                print(f"{TAG} identity FAILED on {sid}/{arm}: SELF_max={worst:.4f} "
                      f"ALL_gap={row['ALL_gap']:+.4f} (dG_S={dG_S:+.2f})", flush=True)
                if args.smoke:
                    sink.close(); return 3
    sink.close()
    print(f"{TAG} sweep done: {n_rows} rows, {host.n_forwards} forwards, "
          f"{host.ms_per_forward():.0f} ms/forward, {(time.time()-t0)/3600:.2f} h", flush=True)
    return 0


# ── stage 2: joint pass ───────────────────────────────────────────────────────

def stage_joint(args: argparse.Namespace, cfg: dict, out: Path) -> int:
    t0 = time.time()
    hp_ = out / "heads_rows.jsonl"
    if not hp_.exists():
        print(f"{TAG} REFUSED: run --stage sweep first ({hp_})"); return 2
    hrows = [json.loads(l) for l in open(hp_)]
    by_item, h0, c3 = load_spans(out)
    trc = Path(args.traces or _PROJ / f"data/nla/p0/trace_llr/{args.model}/traces.jsonl")
    traces = {t["snippet_id"]: t for t in map(json.loads, open(trc))}
    pool = [r for rs in by_item.values() for r in rs]
    host = Host(args, cfg, knockout=False)
    all_comps = components(host.sweep_layers, cfg["n_heads"])
    k_list = [int(k) for k in cfg["k_list"]]
    n_random = int(cfg["n_random"])

    # Selection profiles per arm: item -> {component: suf}; the pool is the single list the sweep ran.
    per_item = {a: {r["snippet_id"]: r["suf"] for r in hrows if r["arm"] == a} for a in ARMS}
    names = list(hrows[0]["suf"].keys())
    pool = [Component.from_name(n) for n in names]
    sids = [r["snippet_id"] for r in hrows if r["arm"] == ARMS[0]]

    sink = open(out / "joint_rows.jsonl", "w")
    for ii, sid in enumerate(sids):
        if time.time() - t0 > args.max_hours * 3600:
            print(f"{TAG} wall-clock stop after {ii} items", flush=True); break
        tr = traces[sid]
        pids, rids = tr["l1b_prompt_ids"], tr["l0_reply_ids"]
        targets = item_targets(by_item[sid], h0, c3, args.arm_list, pool)
        host.rep.set_targets(None); host.patcher.set_patches(None)
        cU = host.patcher.record(); logp_U = host.logp(pids, rids); host.patcher.stop_recording()
        for arm in args.arm_list:
            if arm not in targets:
                continue
            tg = targets[arm]; n_pos = len(tg)
            host.rep.set_targets(tg); host.patcher.set_patches(None)
            cS = host.patcher.record(); logp_S = host.logp(pids, rids, n_pos); host.patcher.stop_recording()
            host.rep.set_targets(None)
            row = {"snippet_id": sid, "arm": arm, "half": half_of(sid), "logp_U": logp_U,
                   "logp_S": logp_S, "dG_S": logp_S - logp_U, "n_pos_S": n_pos, "comps": {}}
            try:
                ranked = rank_on_other_half(per_item[arm], half_of(sid), names)
                row["selection"] = "other_half"
            except ValueError:
                # Only possible when every item fell in one crc32 half (3-item smoke). In-sample
                # selection is fine for a harness check and is stamped so it can never be scored
                # as a localisation result.
                if not args.smoke:
                    raise
                ranked = sorted(names, key=lambda n: (-float(np.mean(
                    [per_item[arm][s][n] for s in per_item[arm]])), n))
                row["selection"] = "in_sample_smoke"

            def run(label: str, comps: list[Component]) -> None:
                host.patcher.set_patches({c: cS.get(c) for c in comps})
                row[f"dG_{label}"] = host.logp(pids, rids) - logp_U
                row[f"n_pos_{label}"] = len(comps)
                row["comps"][label] = [c.name for c in comps]

            for k in k_list:
                if k > len(pool):
                    continue
                run(f"TOP_{k}", [Component.from_name(n) for n in ranked[:k]])
                top = [Component.from_name(n) for n in ranked[:k]]
                rs, ls = [], []
                for r in range(n_random):
                    rng = np.random.default_rng(SEED + zlib.crc32(f"{sid}|{arm}|{k}|{r}".encode()))
                    run(f"RAND_{k}_{r}", draw_uniform(pool, k, rng))
                    run(f"LRAND_{k}_{r}", draw_layer_matched(top, pool, rng))
                    rs.append(row[f"dG_RAND_{k}_{r}"]); ls.append(row[f"dG_LRAND_{k}_{r}"])
                row[f"dG_RAND_{k}"], row[f"n_pos_RAND_{k}"] = float(np.mean(rs)), k
                row[f"dG_LRAND_{k}"], row[f"n_pos_LRAND_{k}"] = float(np.mean(ls)), k
            run("ALL", all_comps)
            host.patcher.set_patches(None)
            sink.write(json.dumps(row) + "\n"); sink.flush()
            ks = [k for k in k_list if f"dG_TOP_{k}" in row]
            print(f"{TAG} joint {ii+1}/{len(sids)} {sid} {arm:7s} dG_S={row['dG_S']:+.2f} " +
                  " ".join(f"top{k}={row[f'dG_TOP_{k}']:+.1f}/r{row[f'dG_RAND_{k}']:+.1f}" for k in ks) +
                  f" ALL={row['dG_ALL']:+.2f} ({host.ms_per_forward():.0f} ms/fwd)", flush=True)
    sink.close()
    print(f"{TAG} joint done: {host.n_forwards} forwards, {(time.time()-t0)/60:.1f} min", flush=True)
    return 0


# ── scoring ───────────────────────────────────────────────────────────────────

def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr
    if len(a) < 3 or np.all(a == a[0]) or np.all(b == b[0]):
        return float("nan")
    return float(spearmanr(a, b).correlation)


def score(hrows: list[dict], jrows: list[dict], cfg: dict) -> dict:
    n_boot = int(cfg["n_boot"]); rng = np.random.default_rng(SEED)
    ident = cfg["identity"]; loc = cfg["localisation"]; same = cfg["same_circuit"]
    tol = float(ident["self_tol_nats"])

    def boot_mean(v: np.ndarray) -> tuple[float, float, float]:
        v = np.asarray(v, float)
        if len(v) == 0:
            return float("nan"), float("nan"), float("nan")
        d = v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(1)
        return float(v.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))

    def boot_matrix(M: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Per-column mean and CI with one shared set of item draws (cluster bootstrap)."""
        n = M.shape[0]
        idx = rng.integers(0, n, size=(n_boot, n))
        means = np.empty((n_boot, M.shape[1]), dtype=np.float32)
        for st in range(0, n_boot, 500):
            means[st:st + 500] = M[idx[st:st + 500]].mean(1)
        return M.mean(0), np.percentile(means, 2.5, axis=0), np.percentile(means, 97.5, axis=0)

    stats: dict = {"experiment": EXPERIMENT, "seed": SEED, "n_boot": n_boot,
                   "config_sha256": cfg["_sha256"], "arms": {}, "identity": {}}
    names = list(hrows[0]["suf"].keys())
    comps = [Component.from_name(n) for n in names]
    head_idx = [i for i, c in enumerate(comps) if c.head != MLP]
    mlp_idx = [i for i, c in enumerate(comps) if c.head == MLP]
    glob_idx = [i for i in head_idx if is_global_layer(comps[i].layer, cfg["global_layers_every"])]
    loc_idx = [i for i in head_idx if i not in set(glob_idx)]

    # ── identities (the gate) ──
    self_max = max((abs(v) for r in hrows for v in r["self"].values()), default=0.0)
    all_fail = [r["snippet_id"] for r in hrows
                if abs(r["ALL_gap"]) > max(tol, ident["all_tol_frac"] * abs(r["dG_S"]))]
    ko_max = max((abs(v) for r in hrows if r["ko_gap"] for v in r["ko_gap"].values()), default=0.0)
    n_self = sum(len(r["self"]) for r in hrows)
    stats["identity"] = {
        "self_max_abs": self_max, "self_n": n_self, "self_tol_nats": tol,
        "self_nonzero_frac": (sum(1 for r in hrows for v in r["self"].values() if v != 0.0)
                              / max(n_self, 1)),
        "repeat_gap_max_abs": max((abs(r.get("repeat_gap", 0.0)) for r in hrows), default=0.0),
        "all_gap_max_abs": max(abs(r["ALL_gap"]) for r in hrows), "all_tol_frac": ident["all_tol_frac"],
        "all_fail_items": all_fail, "ko_gap_max_abs": ko_max,
        "ko_gap_tol_nats": ident["ko_gap_tol_nats"],
        "passes": (self_max <= tol and not all_fail and ko_max <= ident["ko_gap_tol_nats"])}

    profiles, top16 = {}, {}
    for arm in sorted({r["arm"] for r in hrows}):
        R = [r for r in hrows if r["arm"] == arm]
        if not R:
            continue
        S = np.array([[r["suf"][n] for n in names] for r in R])
        N = np.array([[r["nec"][n] for n in names] for r in R])
        dG = np.array([r["dG_S"] for r in R])
        sm, slo, shi = boot_matrix(S); nm, nlo, nhi = boot_matrix(N)
        order = np.argsort(-sm)
        k16 = min(16, len(names))
        top16[arm] = [names[i] for i in order[:k16]]
        profiles[arm] = sm
        a: dict = {"n_items": len(R), "dG_S": boot_mean(dG),
                   "sum_single_suf_over_dG_S": float(sm.sum() / dG.mean()) if dG.mean() else float("nan"),
                   "top10_suf": [(names[i], float(sm[i]), float(slo[i]), float(shi[i])) for i in order[:10]],
                   "top10_nec": [(names[i], float(nm[i]), float(nlo[i]), float(nhi[i]))
                                 for i in np.argsort(-nm)[:10]],
                   "per_component": {n: {"suf": [float(sm[i]), float(slo[i]), float(shi[i])],
                                         "nec": [float(nm[i]), float(nlo[i]), float(nhi[i])]}
                                     for i, n in enumerate(names)}}
        # H-W31b: heads vs MLPs
        bh = max(head_idx, key=lambda i: sm[i]) if head_idx else None
        bm = max(mlp_idx, key=lambda i: sm[i]) if mlp_idx else None
        a["H_W31b"] = {"best_head": (names[bh], float(sm[bh])) if bh is not None else None,
                       "best_mlp": (names[bm], float(sm[bm])) if bm is not None else None,
                       "mlps_in_top16": sum(1 for n in top16[arm] if n.endswith("M")),
                       "heads_in_top16": sum(1 for n in top16[arm] if not n.endswith("M"))}
        # depth profile: summed necessity per layer
        layers = sorted({c.layer for c in comps})
        a["depth_nec"] = {str(L): float(sum(nm[i] for i, c in enumerate(comps) if c.layer == L))
                          for L in layers}
        a["depth_suf"] = {str(L): float(sum(sm[i] for i, c in enumerate(comps) if c.layer == L))
                          for L in layers}
        # H-W31d: global vs local heads, on nec (primary) and read (secondary)
        if glob_idx and loc_idx:
            gl = N[:, glob_idx].mean(1) - N[:, loc_idx].mean(1)
            long = np.array([(r["T"] - r["last_span_pos"]) > cfg["global_heads"]["long_item_tokens"]
                             for r in R])
            d: dict = {"nec_global_minus_local": boot_mean(gl), "n_long": int(long.sum()),
                       "nec_gap_long": boot_mean(gl[long]) if long.any() else None,
                       "nec_gap_short": boot_mean(gl[~long]) if (~long).any() else None}
            m, lo, hi = d["nec_global_minus_local"]
            d["clears"] = bool(lo > 0)
            if all(r["read"] for r in R):
                Rd = np.array([[r["read"][names[i]] for i in head_idx] for r in R])
                hpos = {i: j for j, i in enumerate(head_idx)}
                gr = Rd[:, [hpos[i] for i in glob_idx]].mean(1) - Rd[:, [hpos[i] for i in loc_idx]].mean(1)
                d["read_global_minus_local"] = boot_mean(gr)
            a["H_W31d"] = d
        # H-W31e: direct readers
        if all(r["read"] for r in R) and head_idx:
            Rd = np.array([[r["read"][names[i]] for i in head_idx] for r in R])
            rm, rlo, rhi = boot_matrix(Rd)
            nec_heads = nm[head_idx]
            top_nec = np.argsort(-nec_heads)[:min(16, len(head_idx))]
            a["H_W31e"] = {"spearman_read_vs_nec": _spearman(rm, nec_heads),
                           "top16_nec_with_read_ci_excl0": int(sum(1 for j in top_nec
                                                                   if rlo[j] > 0 or rhi[j] < 0)),
                           "top10_read": [(names[head_idx[j]], float(rm[j]), float(rlo[j]), float(rhi[j]))
                                          for j in np.argsort(-rm)[:10]]}
            for j, i in enumerate(head_idx):
                a["per_component"][names[i]]["read"] = [float(rm[j]), float(rlo[j]), float(rhi[j])]
        # H-W31a: joint pass
        J = [r for r in jrows if r["arm"] == arm]
        if J:
            jd: dict = {"n_items": len(J), "dG_S": boot_mean([r["dG_S"] for r in J]),
                        "ALL_over_dG_S": float(np.mean([r["dG_ALL"] for r in J]) /
                                               np.mean([r["dG_S"] for r in J])), "k": {}}
            mean_dGS = float(np.mean([r["dG_S"] for r in J]))
            first_k = None
            for k in [int(k) for k in cfg["k_list"]]:
                if f"dG_TOP_{k}" not in J[0]:
                    continue
                e: dict = {}
                for lab in ("TOP", "RAND", "LRAND"):
                    e[lab] = boot_mean([r[f"dG_{lab}_{k}"] for r in J])
                e["recovered"] = e["TOP"][0] / mean_dGS if mean_dGS else float("nan")
                for lab in ("RAND", "LRAND"):
                    try:
                        diffs, rep = paired(J, f"TOP_{k}", f"{lab}_{k}", allow_exact_zero=True)
                        m, lo, hi = boot_mean(np.array(diffs))
                        e[f"TOP_minus_{lab}"] = {"mean": m, "ci95": [lo, hi], "n": len(diffs),
                                                 "frac_of_dG_S": m / mean_dGS if mean_dGS else float("nan"),
                                                 "clears": bool(lo > 0), "guard": rep}
                    except ArmNotWritten as ex_:
                        e[f"TOP_minus_{lab}"] = {"absent": str(ex_)}
                if first_k is None and e["recovered"] >= loc["recover_frac"]:
                    first_k = k
                jd["k"][str(k)] = e
            kp = str(loc["k_primary"])
            verdict = "W31-NOT-EVALUATED"
            if kp in jd["k"]:
                e = jd["k"][kp]; tr_ = e.get("TOP_minus_RAND", {})
                if (e["recovered"] >= loc["recover_frac"] and not tr_.get("absent")
                        and tr_.get("frac_of_dG_S", 0) >= loc["over_random_frac"] and tr_.get("clears")):
                    verdict = "W31-CONCENTRATED"
                elif first_k is None or first_k > loc["k_max"]:
                    verdict = "W31-DISTRIBUTED"
                else:
                    verdict = "W31-INTERMEDIATE"
            jd["first_k_at_recover_frac"] = first_k
            jd["H_W31a"] = verdict
            a["joint"] = jd
        stats["arms"][arm] = a

    # H-W31c: does the NLA state route like the raw state?
    if all(a in profiles for a in ARMS):
        rho = _spearman(profiles["C3pure"], profiles["P_patch"])
        s1, s2 = set(top16["C3pure"]), set(top16["P_patch"])
        jac = len(s1 & s2) / max(len(s1 | s2), 1)
        if rho >= same["spearman_same"] and jac >= same["jaccard_same"]:
            v = "W31-SAME-CIRCUIT"
        elif rho < same["spearman_different"]:
            v = "W31-DIFFERENT-CIRCUIT"
        else:
            v = "W31-INDETERMINATE"
        # residue: components where C3pure suf < P_patch suf by a CI-clearing margin
        Rc = {r["snippet_id"]: r for r in hrows if r["arm"] == "C3pure"}
        Rp = {r["snippet_id"]: r for r in hrows if r["arm"] == "P_patch"}
        common = sorted(set(Rc) & set(Rp))
        resid = []
        if common:
            Dm = np.array([[Rc[s]["suf"][n] - Rp[s]["suf"][n] for n in names] for s in common])
            dm, dlo, dhi = boot_matrix(Dm)
            resid = [(names[i], float(dm[i]), float(dlo[i]), float(dhi[i]))
                     for i in np.argsort(dm) if dhi[i] < 0][:10]
        stats["H_W31c"] = {"spearman_suf": rho, "jaccard_top16": jac, "top16": top16,
                           "verdict": v, "c3_below_p_patch_ci_clearing": resid}

    stats["verdict"] = ("W31-HARNESS-FAULT" if not stats["identity"]["passes"] else
                        {a: stats["arms"][a].get("joint", {}).get("H_W31a", "W31-NOT-EVALUATED")
                         for a in stats["arms"]})
    return stats


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="gemma12b", choices=sorted(HOSTS))
    ap.add_argument("--allow-banked-host", action="store_true")
    ap.add_argument("--stage", required=True, choices=("vectors", "sweep", "joint"))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--config", default=str(CFG_PATH))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-spans-per-item", type=int, default=None)
    ap.add_argument("--traces", default=None)
    ap.add_argument("--repair", action="store_true", help="H-W23 repaired anchoring (config default: on)")
    ap.add_argument("--no-repair", action="store_true")
    ap.add_argument("--layers", default=None, help="single-component layer range, e.g. 45-47")
    ap.add_argument("--read-knockout", action="store_true")
    ap.add_argument("--arms", default=None,
                    help="comma-separated subset; default C3pure,P_patch. H-W35 nulls: N_sibling,N_foreign,N_random")
    ap.add_argument("--self-per-item", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--score-only", action="store_true")
    ap.add_argument("--max-hours", type=float, default=9.0)
    args = ap.parse_args()

    if args.model == "qwen7b" and not args.allow_banked_host:
        print(f"{TAG} REFUSED: qwen7b is a Chinese model; pass --allow-banked-host.")
        return 2
    cfg = load_cfg(Path(args.config))
    args.repair = bool(cfg.get("repair", True)) if not (args.repair or args.no_repair) else (
        args.repair and not args.no_repair)
    if args.max_spans_per_item is None:
        args.max_spans_per_item = int(cfg["max_spans_per_item"])
    if args.self_per_item is None:
        args.self_per_item = int(cfg["self_per_item"])
    args.arm_list = tuple(a.strip() for a in args.arms.split(",")) if args.arms else ARMS
    bad = [a for a in args.arm_list if a not in ARMS + NULL_ARMS]
    if bad:
        print(f"{TAG} REFUSED: unknown arms {bad}"); return 2
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    if args.score_only:
        hp_, jp_ = out / "heads_rows.jsonl", out / "joint_rows.jsonl"
        if not hp_.exists():
            print(f"{TAG} REFUSED: no banked rows at {hp_}"); return 2
        hrows = [json.loads(l) for l in open(hp_)]
        jrows = [json.loads(l) for l in open(jp_)] if jp_.exists() else []
        st = score(hrows, jrows, cfg)
        json.dump(st, open(out / "heads_stats.json", "w"), indent=2)
        _print_stats(st)
        return 0 if st["identity"]["passes"] else 3

    rc = {"vectors": stage_vectors, "sweep": stage_sweep, "joint": stage_joint}[args.stage](args, cfg, out)
    if rc != 0:
        return rc
    if args.stage in ("sweep", "joint"):
        hrows = [json.loads(l) for l in open(out / "heads_rows.jsonl")]
        jp_ = out / "joint_rows.jsonl"
        jrows = [json.loads(l) for l in open(jp_)] if jp_.exists() else []
        if hrows:
            st = score(hrows, jrows, cfg)
            st["stage"] = args.stage
            json.dump(st, open(out / "heads_stats.json", "w"), indent=2)
            _print_stats(st)
            if not st["identity"]["passes"]:
                print(f"{TAG} REFUSED: identity gate failed — results are NOT reportable.", flush=True)
                return 3
    return 0


def _print_stats(st: dict) -> None:
    i = st["identity"]
    print(f"\n  identity: SELF max {i['self_max_abs']:.4f} (tol {i['self_tol_nats']}, "
          f"{i['self_nonzero_frac']*100:.0f}% nonzero, repeat gap {i['repeat_gap_max_abs']:.4f}) "
          f"ALL max {i['all_gap_max_abs']:.4f} fails={len(i['all_fail_items'])} "
          f"ko_gap max {i['ko_gap_max_abs']:.4f} -> {'PASS' if i['passes'] else 'FAIL'}")
    for arm, a in st["arms"].items():
        m, lo, hi = a["dG_S"]
        print(f"  {arm:7s} n={a['n_items']} dG_S={m:+.2f} [{lo:+.2f},{hi:+.2f}] "
              f"sum_suf/dG_S={a['sum_single_suf_over_dG_S']:.2f}")
        print("    top suf: " + ", ".join(f"{n}:{v:+.1f}" for n, v, *_ in a["top10_suf"][:6]))
        print("    top nec: " + ", ".join(f"{n}:{v:+.1f}" for n, v, *_ in a["top10_nec"][:6]))
        if "H_W31d" in a:
            d = a["H_W31d"]; m, lo, hi = d["nec_global_minus_local"]
            print(f"    H-W31d global-local nec {m:+.2f} [{lo:+.2f},{hi:+.2f}] clears={d['clears']} "
                  f"n_long={d['n_long']}")
        if "H_W31e" in a:
            e = a["H_W31e"]
            print(f"    H-W31e spearman(read,nec)={e['spearman_read_vs_nec']:.2f} "
                  f"top16-nec with read CI>0: {e['top16_nec_with_read_ci_excl0']}")
        if "joint" in a:
            j = a["joint"]
            for k, e in j["k"].items():
                tr_ = e.get("TOP_minus_RAND", {})
                print(f"    k={k:>2s} TOP {e['TOP'][0]:+.2f} RAND {e['RAND'][0]:+.2f} "
                      f"LRAND {e['LRAND'][0]:+.2f} recovered={e['recovered']:.2f} "
                      f"TOP-RAND={tr_.get('mean', float('nan')):+.2f} clears={tr_.get('clears')}")
            print(f"    H-W31a {j['H_W31a']} (first k >= 50%: {j['first_k_at_recover_frac']}, "
                  f"ALL/dG_S={j['ALL_over_dG_S']:.3f})")
    if "H_W31c" in st:
        c = st["H_W31c"]
        print(f"  H-W31c spearman={c['spearman_suf']:.3f} jaccard16={c['jaccard_top16']:.2f} "
              f"-> {c['verdict']}")
    print(f"{TAG} VERDICT {st['verdict']}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
