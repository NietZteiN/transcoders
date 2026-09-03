"""Does this host's AR say anything aligned with this host's residual stream?

WHY THIS EXISTS. V1 steering injects `AR(gloss_true) - AR(gloss_decoy)` into the residual stream.
That is only meaningful if the AR's output lives in the same space, at the same scale, and points
somewhere the actual clean-vs-decoy contrast points. On Qwen that was established by the original
replication. On Gemma NOTHING has established it: the port's G0 stage A checked extraction, and
stage B (the AV->AR round trip) needs an AV server and has never run.

Steering with an unvalidated AR would produce numbers either way and they would mean nothing --
the same class of error as loading the wrong host's dictionary. So four checks, none needing the
AV server, because `reconstruct()` is AR-only:

  1. DIMENSION. AR output width == the subject's d_model. A mismatch raises at write time on
     Gemma but would be SILENT on a same-width host, so it is checked explicitly.
  2. SCALE. ||AR output|| against the real activation norms at the AR's own layer, taken from the
     multilayer bank's absolute clean activations. The write hook unit-normalises, so scale does
     not break steering -- but an AR output many orders of magnitude off its training
     distribution is evidence the checkpoint is being used wrong.
  3. SEPARATION. AR(gloss_true) != AR(gloss_decoy). If the AR maps both glosses to the same
     vector, V1 is identically zero and every V1 result is a null by construction.
  4. ALIGNMENT -- the one that matters. cos(V1, V3) per item, where V3 is the real
     clean-minus-decoy activation difference at that layer. This does NOT have to be high for V1
     to be worth running (V1 could be a different, better direction). But if it is ~0 AND V1
     steering then does nothing, the two facts together say the AR is not addressing this space,
     which is a different conclusion from "there is no belief to edit" -- and the difference is
     exactly what B4 was unable to distinguish.

Reports, does not gate on 4: a low alignment is a finding, not a failure.
"""
from __future__ import annotations

import argparse, json, os, sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_NLA_ROOT = _HERE.parent
_PROJ = _NLA_ROOT.parent
sys.path.insert(0, str(_NLA_ROOT / "vendor" / "nla-repo"))
sys.path.insert(0, str(_HERE))

from steer_run import AR_CHECKPOINTS, HOSTS, build_user, load_pairs  # noqa: E402
from steer_vectors import cosine  # noqa: E402

SEED = 20260724


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="gemma12b", choices=sorted(HOSTS))
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import random, yaml
    from nla_inference import NLACritic

    ckpt = AR_CHECKPOINTS.get(args.host)
    if ckpt is None:
        sys.exit(f"no AR registered for host {args.host}")
    meta = yaml.safe_load((Path(ckpt) / "nla_meta.yaml").read_text())
    layer = int(meta["extraction_layer_index"])
    d_meta = int(meta["d_model"])
    hf_id, host_layer = HOSTS[args.host]
    out = Path(args.out or _PROJ / f"data/nla/p0/steerv2/{args.host}/ar_gate.json")
    out.parent.mkdir(parents=True, exist_ok=True)

    rep = {"experiment": "ar_gate", "host": args.host, "subject": hf_id,
           "ar_checkpoint": str(ckpt), "ar_layer": layer, "ar_d_model": d_meta,
           "mse_scale": meta["extraction"]["mse_scale"], "seed": SEED, "checks": {}}

    pairs = load_pairs(args.limit, random.Random(SEED))
    print(f"[ar] {args.host} · AR {ckpt}", flush=True)
    print(f"[ar] AR layer {layer} · d_model {d_meta} · {len(pairs)} pairs", flush=True)

    ar = NLACritic(ckpt, device=args.device)

    # ── 1. dimension ────────────────────────────────────────────────────────
    probe = ar.reconstruct("a function that adds two numbers")
    d_out = int(probe.shape[-1])
    rep["checks"]["dimension"] = {"ar_output_dim": d_out, "sidecar_d_model": d_meta,
                                  "passed": d_out == d_meta}

    # ── 2. scale, against the real activations at the AR's own layer ────────
    bank_p = _PROJ / f"data/nla/p0/steerv2/{args.host}/multilayer_bank.npz"
    scale = {"ar_norm_median": None, "activation_norm_median": None, "ratio": None,
             "passed": None, "note": "no bank on disk; run multilayer_vectors.py"}
    if bank_p.exists():
        bank = np.load(bank_p, allow_pickle=True)
        if "clean" in bank.files and layer < bank["clean"].shape[1]:
            act = np.linalg.norm(bank["clean"][:, layer, :], axis=-1)
            ar_norms = []
            for p in pairs[:8]:
                ar_norms.append(float(ar.reconstruct(p["gloss_true"]).norm()))
            a_med, r_med = float(np.median(act)), float(np.median(ar_norms))
            ratio = r_med / a_med if a_med else None
            # Three orders of magnitude is the band where "same space, different scale" stops
            # being a plausible reading. The hook normalises, so this informs rather than blocks.
            scale = {"ar_norm_median": round(r_med, 3),
                     "activation_norm_median": round(a_med, 3),
                     "ratio": round(ratio, 6) if ratio else None,
                     "passed": bool(ratio and 1e-3 < ratio < 1e3)}
    rep["checks"]["scale"] = scale

    # ── 3 & 4. separation and alignment ─────────────────────────────────────
    V3_mean = None
    if bank_p.exists():
        bank = np.load(bank_p, allow_pickle=True)
        ids = {str(k): i for i, k in enumerate(bank["item_ids"])}
        D = bank["deltas"]
    cos_v1v3, zero_v1, norms_v1 = [], 0, []
    for p in pairs:
        v_true = ar.reconstruct(p["gloss_true"]).float()
        v_dec = ar.reconstruct(p["gloss_decoy"]).float()
        v1 = (v_true - v_dec).numpy()
        n = float(np.linalg.norm(v1))
        norms_v1.append(n)
        if n < 1e-6:
            zero_v1 += 1
            continue
        if bank_p.exists() and p["snippet_id"] in ids and layer < D.shape[1]:
            # leave-one-out V3 at the AR's layer, the same construction steer_run uses
            i = ids[p["snippet_id"]]
            v3 = (D.sum(axis=0)[layer] - D[i][layer]) / (D.shape[0] - 1)
            cos_v1v3.append(cosine(torch.from_numpy(v1), torch.from_numpy(v3.copy())))

    rep["checks"]["separation"] = {
        "n_items": len(pairs), "n_identical_glosses": zero_v1,
        "v1_norm_median": round(float(np.median(norms_v1)), 4) if norms_v1 else None,
        "passed": zero_v1 == 0,
        "note": "AR(gloss_true) == AR(gloss_decoy) would make V1 identically zero",
    }
    rep["checks"]["alignment"] = {
        "n": len(cos_v1v3),
        "cos_v1_v3_mean": round(float(np.mean(cos_v1v3)), 4) if cos_v1v3 else None,
        "cos_v1_v3_median": round(float(np.median(cos_v1v3)), 4) if cos_v1v3 else None,
        "cos_v1_v3_min": round(float(np.min(cos_v1v3)), 4) if cos_v1v3 else None,
        "cos_v1_v3_max": round(float(np.max(cos_v1v3)), 4) if cos_v1v3 else None,
        "passed": True,   # reported, never gating — see module docstring
        "note": ("Low alignment does NOT invalidate V1; it means V1 points somewhere the "
                 "contrastive direction does not. Read together with the V1 steering result: "
                 "low alignment AND a null is 'the AR is not addressing this space', which is a "
                 "different conclusion from 'there is no belief to edit'."),
    }

    gating = [k for k in ("dimension", "scale", "separation") if rep["checks"][k]["passed"] is False]
    rep["verdict"] = "FAIL" if gating else "PASS"
    rep["failed_checks"] = gating
    rep["finished_utc"] = datetime.now(timezone.utc).isoformat()
    out.write_text(json.dumps(rep, indent=2))

    for k, v in rep["checks"].items():
        st = {True: "PASS", False: "FAIL", None: "SKIP"}[v["passed"]]
        body = {kk: vv for kk, vv in v.items() if kk not in ("passed", "note")}
        print(f"[ar] {k:<12}{st}  {json.dumps(body)}", flush=True)
    print(f"\n[ar] VERDICT {rep['verdict']} -> {out}")
    return 0 if rep["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
