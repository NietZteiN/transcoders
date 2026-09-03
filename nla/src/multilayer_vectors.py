"""Per-layer V3/V4 directions, and an EMPIRICAL energy match for multi-layer steering.

Two jobs, both prerequisites for the KV-bypass run, neither involving the NLA.

1. PER-LAYER DIRECTIONS. V3 (held-out mean contrastive direction) and V4 (item oracle) are plain
   activation differences, so they exist at every layer. Extraction reuses
   `layer_rotation.all_layer_acts` verbatim — the same read position (final prompt token, the site
   B4 steers) through the same code path that carries the layer-index gate — so the per-layer
   vectors cannot drift from the banked layer-20 construction. V3 is built LEAVE-ONE-OUT at every
   layer: it is fit on pairs and applied to items, so including the applied item would leak
   (steer_vectors.py flags this for the L20 case; the hazard is identical here).

   V1/V2 deliberately do NOT ride along. The AR is trained at layer 20 only, so there is no
   licensed way to produce a layer-7 NLA direction. This run therefore tests the CHANNEL, not the
   NLA — which is exactly what P0 is for.

2. ENERGY MATCHING, EMPIRICALLY. The gate measured what analysis predicted: alpha=1.0 written at
   21 layers compounds to a 682-unit displacement, because each edit changes the input to the next
   layer, which is then edited again. Comparing that against single-layer alpha=1.0 would confound
   "the correction now reaches the KV cache" with "the perturbation is orders of magnitude bigger"
   — and P0.2 already showed where oversized perturbation leads (V1's parse rate collapsed to
   0.100 when coverage widened at unchanged alpha).

   The analytic 1/sqrt(n) in MultiLayerSpec.energy_matched holds the sum of squared COEFFICIENTS
   constant, which is not the same as holding the delivered perturbation constant once edits
   compound. So this script measures the real thing: relative displacement of the final-layer
   residual at the steered position,
        r(alpha) = || h_27^steered - h_27^unsteered || / || h_27^unsteered ||,
   for single-layer steering at the banked grid and for multi-layer steering across a wide alpha
   sweep, then reports the multi-layer alpha whose MEDIAN r matches each single-layer alpha's.

   This is a calibration on displacement, not on any outcome — no accuracy, parse rate or reply is
   consulted. Stated plainly so it is auditable: the matched alphas are picked from these curves
   BEFORE the steering run, and frozen into the pre-registration.
"""
from __future__ import annotations

import argparse, json, sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_NLA_ROOT = _HERE.parent
_PROJ = _NLA_ROOT.parent
sys.path.insert(0, str(_NLA_ROOT / "vendor" / "nla-repo"))
sys.path.insert(0, str(_HERE))

from steer_run import HOSTS, DEFAULT_HOST, LAYER_INDEX, TARGET_MODEL, build_user, load_pairs  # noqa: E402
from layer_rotation import all_layer_acts  # noqa: E402
from steer import ActivationSteerer, SteerSpec  # noqa: E402
from steer_multilayer import MultiLayerSpec, MultiLayerSteerer, model_dims  # noqa: E402

SEED = 20260724
SINGLE_ALPHAS = [0.25, 0.5, 1.0, 2.0, 4.0]           # the banked B4 grid
MULTI_ALPHAS = [0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]


def v3_loo(D: np.ndarray, i: int) -> np.ndarray:
    """[n_layers, d] held-out mean direction for item i. D is [n_pairs, n_layers, d]."""
    return (D.sum(axis=0) - D[i]) / (D.shape[0] - 1)


def displacement(model, tokz, user: str, steerer, spec, target_layer: int) -> float:
    """|| h_final^steered - h_final^unsteered || / || h_final^unsteered || at the edited position."""
    ids = tokz.apply_chat_template([{"role": "user", "content": user}], tokenize=True,
                                   add_generation_prompt=True, return_dict=False)
    x = torch.tensor([ids], device=model.device)
    steerer.set_spec(None) if isinstance(steerer, MultiLayerSteerer) else steerer.set_spec(None)
    with torch.no_grad():
        base = model(x, output_hidden_states=True).hidden_states[-1][0, -1].float()
    if isinstance(steerer, MultiLayerSteerer):
        steerer.set_spec(spec)
    else:
        steerer.set_spec(spec, prompt_len=len(ids), max_total=len(ids) + 1)
    with torch.no_grad():
        st = model(x, output_hidden_states=True).hidden_states[-1][0, -1].float()
    steerer.set_spec(None)
    return float((st - base).norm() / base.norm().clamp_min(1e-12))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=DEFAULT_HOST, choices=sorted(HOSTS))
    ap.add_argument("--model", default=None)
    ap.add_argument("--target-layer", type=int, default=None)
    ap.add_argument("--limit", type=int, default=0, help="0 = all pairs")
    ap.add_argument("--calib-items", type=int, default=12,
                    help="items used for the displacement curves (a median, not a fit)")
    ap.add_argument("--out-dir", default=str(_PROJ / "data/nla/p0/steerv2"))
    args = ap.parse_args()
    hf_id, host_layer = HOSTS[args.host]
    args.model = args.model or hf_id
    if args.target_layer is None:
        args.target_layer = host_layer
    # Artifacts are namespaced by host: the bank is [n_pairs, n_layers, d] and its d and n_layers
    # are host-specific, so one shared filename would let a gemma run load a qwen bank and steer
    # with vectors of the wrong width at layers that do not correspond.
    out = Path(args.out_dir) / args.host; out.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(SEED); np.random.seed(SEED)

    tokz = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
                                                 device_map="cuda").eval()
    import random
    pairs = load_pairs(args.limit or None, random.Random(SEED))
    L = args.target_layer
    d_model, n_layers = model_dims(model)
    if not 0 <= L < n_layers:
        raise SystemExit(f"--target-layer {L} outside this host's {n_layers} layers")
    print(f"[mlv] {len(pairs)} pairs · target layer {L} · {n_layers} layers · d_model {d_model}",
          flush=True)

    # ── 1. per-layer contrastive differences, one forward pass per condition per item ──
    # The CLEAN absolute activations are saved alongside the differences, because state
    # replacement (V5) must ASSIGN h_l := h_clean,l rather than add h_clean,l - h_obf,l. Adding
    # the difference at every layer does not reproduce the clean state: layer l+1 receives the
    # state layer l already corrected and adds its own difference on top, so error accumulates
    # linearly with depth. Only an absolute target is idempotent w.r.t. what propagated up.
    deltas, cleans = [], []
    for i, p in enumerate(pairs):
        c = all_layer_acts(model, tokz, build_user(p["code_l0"], p["call_l0"]))
        o = all_layer_acts(model, tokz, build_user(p["code_l1b"], p["call_l1b"]))
        deltas.append(c - o)
        cleans.append(c)
        if (i + 1) % 10 == 0:
            print(f"[mlv]   extracted {i + 1}/{len(pairs)}", flush=True)
    D = np.stack(deltas).astype(np.float32)          # [n_pairs, n_layers, d]
    C = np.stack(cleans).astype(np.float32)          # [n_pairs, n_layers, d]
    ids_ = [p["snippet_id"] for p in pairs]

    bank = out / "multilayer_bank.npz"
    np.savez_compressed(bank, deltas=D, clean=C, item_ids=np.array(ids_),
                        target_layer=L, seed=SEED)
    print(f"[mlv] bank {tuple(D.shape)} -> {bank}", flush=True)

    # Consistency check against P0.1's published layer-20 vector geometry: the mean direction at
    # the instrument layer must reproduce, or the two scripts disagree about what they extracted.
    mean_L = D.mean(axis=0)[L]
    cohs = []
    for i in range(len(pairs)):
        a, b = D[i][L], v3_loo(D, i)[L]
        cohs.append(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)))
    print(f"[mlv] L{L} LOO coherence {np.mean(cohs):.4f}  ||mean_D|| {np.linalg.norm(mean_L):.2f}",
          flush=True)

    # ── 2. displacement curves ────────────────────────────────────────────────────────
    calib = list(range(min(args.calib_items, len(pairs))))
    single = ActivationSteerer(model, L)
    ml = MultiLayerSteerer(model, range(L + 1))
    curves: dict[str, dict[str, float]] = {"single": {}, "multi": {}}

    for a in SINGLE_ALPHAS:
        rs = []
        for i in calib:
            d = torch.from_numpy(v3_loo(D, i)[L])
            rs.append(displacement(model, tokz, build_user(pairs[i]["code_l1b"], pairs[i]["call_l1b"]),
                                   single, SteerSpec(delta=d, alpha=a, positions="last_prompt"), L))
        curves["single"][f"{a}"] = float(np.median(rs))
        print(f"[mlv] single alpha={a:<5} median rel-displacement {np.median(rs):.4f}", flush=True)

    for a in MULTI_ALPHAS:
        rs = []
        for i in calib:
            per = {l: torch.from_numpy(v3_loo(D, i)[l]) for l in range(L + 1)}
            rs.append(displacement(model, tokz, build_user(pairs[i]["code_l1b"], pairs[i]["call_l1b"]),
                                   ml, MultiLayerSpec(deltas=per, alpha=a, positions="last_prompt"), L))
        curves["multi"][f"{a}"] = float(np.median(rs))
        print(f"[mlv] multi  alpha={a:<5} median rel-displacement {np.median(rs):.4f}", flush=True)
    single.close(); ml.close()

    # ── 3. matched alphas: nearest multi alpha by log-displacement ─────────────────────
    ma = np.array([float(k) for k in curves["multi"]])
    mr = np.array([curves["multi"][k] for k in curves["multi"]])
    matched = {}
    for k, target in curves["single"].items():
        if target <= 0 or mr.max() <= 0:
            continue
        j = int(np.argmin(np.abs(np.log(np.clip(mr, 1e-9, None)) - np.log(max(target, 1e-9)))))
        matched[k] = {"multi_alpha": float(ma[j]), "single_rel_disp": target,
                      "multi_rel_disp": float(mr[j]),
                      "ratio": float(mr[j] / target) if target else None}

    rep = {"experiment": "multilayer_vectors_and_energy_match", "model": args.model,
           "host": args.host,
           "target_layer": L, "n_pairs": len(pairs), "n_calib_items": len(calib),
           "seed": SEED, "bank": str(bank), "loo_coherence_at_target_layer": float(np.mean(cohs)),
           "displacement_curves": curves, "matched_alphas": matched,
           "analytic_1_over_sqrt_n": 1.0 / ((L + 1) ** 0.5),
           "finished_utc": datetime.now(timezone.utc).isoformat()}
    (out / "energy_match.json").write_text(json.dumps(rep, indent=2))
    print("\n[mlv] matched alphas (single -> multi at equal delivered displacement):")
    for k, v in matched.items():
        print(f"[mlv]   single {k:<5} (r={v['single_rel_disp']:.4f})  ->  "
              f"multi {v['multi_alpha']:<6} (r={v['multi_rel_disp']:.4f}, ratio {v['ratio']:.2f})")
    print(f"[mlv] analytic 1/sqrt(n) would have said {rep['analytic_1_over_sqrt_n']:.4f}")
    print(f"[mlv] -> {out / 'energy_match.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
