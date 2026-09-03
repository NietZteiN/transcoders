"""P0.1 — how the clean-minus-obfuscated direction rotates across depth.

Pre-registration: log/nla-harness/2026-08-27_p0-triage-prereg.md

THE QUESTION. B4 steers at layer 20 because that is the only layer the AR was trained at. Whether
that is a real constraint or an incidental one has never been measured: if the task direction
`h_clean - h_obf` points essentially the same way at layers 21-27, the L20-derived vector can be
injected across that whole band with no retraining, and the single-layer limit is soft. If it
rotates away, the limit is hard and we know it as a fact rather than a worry.

WHY `output_hidden_states` HERE AND A HOOK EVERYWHERE ELSE. The project's ActivationExtractor
hooks `layers[K]` deliberately, and documents the correspondence as `hidden_states[K+1]` ("their
index 0 is the embedding output"). Twenty-eight hooks would mean twenty-eight forward passes;
`output_hidden_states=True` gives every layer in one. That is a different code path from the one
every banked number came through, so it is NOT trusted on assertion -- `--gate` reproduces the
extractor's layer-20 vector through this path and fails loudly on mismatch. An off-by-one here
would mislabel every point on the curve and the curve would still look perfectly plausible.

THREE METRICS, because rotation alone can mislead:
  rotation   cos(D20, D_l)              -- does the direction point the same way as at 20
  coherence  mean_i cos(d_i, D_l^(-i))  -- is there a consistent direction at that layer AT ALL
  rel. mag   ||D_l|| / mean_i ||h_i||   -- how large is the contrast relative to the stream

Coherence is leave-one-out on purpose. Scoring an item against a mean it helped build inflates
agreement, and with n=60 that bias is not small. A layer can post a large ||D_l|| that is pure
cancellation noise; coherence is what separates the two.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_NLA_ROOT = _HERE.parent
_PROJ = _NLA_ROOT.parent

import sys
sys.path.insert(0, str(_HERE))

from steer_run import LAYER_INDEX, TARGET_MODEL, build_user, load_pairs  # noqa: E402
from extract import ActivationExtractor  # noqa: E402


def all_layer_acts(model, tokz, user: str) -> np.ndarray:
    """[n_layers, d] — every layer's residual stream at the FINAL PROMPT TOKEN.

    The final prompt token is the site B4 steers, so the contrast is measured exactly where the
    intervention is applied rather than at some other position that happens to be convenient.
    """
    ids = tokz.apply_chat_template([{"role": "user", "content": user}], tokenize=True,
                                   add_generation_prompt=True, return_dict=False)
    with torch.no_grad():
        out = model(torch.tensor([ids], device=model.device), output_hidden_states=True)
    # hidden_states[0] is the embedding output, so layer K == hidden_states[K+1].
    hs = out.hidden_states
    return np.stack([hs[k + 1][0, -1].float().cpu().numpy() for k in range(len(hs) - 1)])


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    return float(a @ b / (na * nb))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=20260724)
    ap.add_argument("--out", default=str(_PROJ / "data" / "nla" / "p0" / "layer_rotation.json"))
    ap.add_argument("--gate-only", action="store_true", help="run the layer-index gate and stop")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    pairs = load_pairs(args.limit, rng)
    print(f"[p0.1] {len(pairs)} L0/L1b pairs", flush=True)

    ex = ActivationExtractor(TARGET_MODEL, LAYER_INDEX, device=args.device)
    model, tokz = ex.model, ex.tokenizer

    # ── the gate: this code path must reproduce the hooked extractor at layer 20 ──────────
    probe = build_user(pairs[0]["code_l0"], pairs[0]["call_l0"])
    mine = all_layer_acts(model, tokz, probe)[LAYER_INDEX]
    theirs = np.asarray(ex.extract_chat(probe, None, text_id="gate").activations[-1], dtype=float)
    rel = float(np.linalg.norm(mine - theirs) / max(np.linalg.norm(theirs), 1e-12))
    print(f"[p0.1] layer-index gate: relative error {rel:.3e} (must be < 1e-4)", flush=True)
    if not rel < 1e-4:
        print("[p0.1] GATE FAILED — hidden_states indexing does not match the hooked extractor. "
              "Per the pre-registration, P0.1 is not reported. Nothing written.", flush=True)
        return 1
    print("[p0.1] gate PASSED", flush=True)
    if args.gate_only:
        return 0

    # Gemma-3's config keeps these under text_config; see steer_multilayer.model_dims.
    from steer_multilayer import model_dims
    _d_model, n_layers = model_dims(model)
    deltas, norms = [], []
    for i, p in enumerate(pairs):
        c = all_layer_acts(model, tokz, build_user(p["code_l0"], p["call_l0"]))
        o = all_layer_acts(model, tokz, build_user(p["code_l1b"], p["call_l1b"]))
        deltas.append(c - o)
        norms.append(np.linalg.norm(o, axis=1))
        if (i + 1) % 10 == 0:
            print(f"[p0.1] {i + 1}/{len(pairs)}", flush=True)

    D = np.stack(deltas)                      # [n_pairs, n_layers, d]
    N = np.stack(norms)                       # [n_pairs, n_layers]
    mean_d = D.mean(axis=0)                   # [n_layers, d]

    rows = []
    for l in range(n_layers):
        # Leave-one-out coherence: an item is never scored against a mean it contributed to.
        loo = [_cos(D[i, l], (D[:, l].sum(axis=0) - D[i, l]) / (len(D) - 1)) for i in range(len(D))]
        rows.append({
            "layer": l,
            "rotation_vs_l20": _cos(mean_d[LAYER_INDEX], mean_d[l]),
            "coherence_loo": float(np.mean(loo)),
            "coherence_loo_sd": float(np.std(loo)),
            "rel_magnitude": float(np.linalg.norm(mean_d[l]) / max(float(N[:, l].mean()), 1e-12)),
            "mean_act_norm": float(N[:, l].mean()),
        })

    band = [r for r in rows if 21 <= r["layer"] <= 27]
    min_band = min(r["rotation_vs_l20"] for r in band)
    verdict = "SOFT" if min_band >= 0.50 else "HARD"
    best_coh = max(rows, key=lambda r: r["coherence_loo"])["layer"]

    out = {
        "experiment": "p0.1_layer_rotation",
        "prereg": "log/nla-harness/2026-08-27_p0-triage-prereg.md",
        "model": TARGET_MODEL, "seed": args.seed, "n_pairs": len(pairs),
        "n_layers": n_layers, "read_position": "final prompt token",
        "gate_relative_error": rel,
        "min_rotation_layers_21_27": min_band,
        "verdict": verdict,
        "threshold": 0.50,
        "argmax_coherence_layer": best_coh,
        "instrument_layer": LAYER_INDEX,
        "layers": rows,
    }
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1))

    print(f"\n[p0.1] {'layer':>5} {'rot vs L20':>11} {'coherence':>10} {'rel mag':>9}")
    for r in rows:
        mark = "  <- instrument" if r["layer"] == LAYER_INDEX else ""
        print(f"[p0.1] {r['layer']:>5} {r['rotation_vs_l20']:>11.3f} "
              f"{r['coherence_loo']:>10.3f} {r['rel_magnitude']:>9.4f}{mark}")
    print(f"\n[p0.1] min rotation over layers 21-27: {min_band:.3f} -> {verdict}")
    print(f"[p0.1] most coherent layer: {best_coh} (instrument sits at {LAYER_INDEX})")
    print(f"[p0.1] wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
