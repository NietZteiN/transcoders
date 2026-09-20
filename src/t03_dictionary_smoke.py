"""T0.3 — dictionary smoke: is a Llama Scope SAE healthy on the checkpoint we will actually read?

Backlog: docs/EXPERIMENT_BACKLOG.md. This is the gate in front of E1/E2/E7, not a formality.

WHY IT IS A GATE. The Llama Scope dictionaries are trained on Llama-3.1-8B **base** (`lm_config.json`),
while every E1/E2 claim would be read on **Instruct**. CLAUDE.md §4 names base->instruct dictionary
transfer as a silent-failure mode, alongside degenerate reconstruction and dead features. If the SAE
does not reconstruct Instruct activations, E1's design has to change BEFORE it produces a
Semantic-Capture Score, not after.

LOADING IS DELEGATED, ON PURPOSE. Llama Scope normalises activations dataset-wise (this checkpoint:
`dataset_average_activation_norm` 6.3125, d_model 4096) and uses JumpReLU at a stored threshold. Getting
that convention wrong produces exactly the signature of a broken dictionary -- near-zero FVE, L0 at 0 or
at d_sae -- so a hand-rolled encode/decode could manufacture a "degenerate on Instruct" verdict out of
my own arithmetic. SAELens ships the official `llama_scope` loader; it is used here so that a bad number
means a bad dictionary rather than a bad reimplementation.

PROVENANCE CHECK. The loader fetches `fnlp/...`; configs/dictionaries.yaml pins
`OpenMOSS-Team/...@8dbc1d85` and that is what is cached locally. The two are compared tensor-wise, so
either the smoke transfers to the pinned copy E1 will use, or we learn now that it does not.

Reported per CLAUDE.md §4 telemetry: FVE, L0, dead-feature share, on real stimuli from this project.

SETTLED CONVENTIONS (established 2026-09-20 by jobs 413871 / 413880 / 413882 / 413884, each of which
refuted a hypothesis rather than confirming one):
  * INPUT IS RAW x. `normalize_activations` is `none` on this cfg and the dataset-wise scaling is folded
    into the weights -- confirmed by the provenance scalars, 10.1386 = sqrt(4096)/6.3125 on the encoder
    and its reciprocal on the decoder, residual ~1e-14. Feeding x*k or unit-normed x drives L0 to ~350.
  * BOS IS EXCLUDED (see above).
  * NO RESCALING. Our mean activation norm at layer 8 is 5.8724 against the stored 6.3125 -- a 7 %
    difference. Applying that correction moves FVE 0.4934 -> 0.5070. Scaling by ~2x would reach FVE
    0.571 and L0 66, but nothing in the data justifies it and it overshoots the trained top_k of 50.
  * The residual low FVE is NOT base->instruct transfer (gap 0.016-0.049), NOT domain shift (Java beats
    English by 0.057), and NOT a scale bug. It is uniform across checkpoint, domain and layer, which is
    what makes an L0-vs-L1b CONTRAST still interpretable even at FVE ~0.5.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

TAG = "[T0.3]"
FVE_HEALTHY, FVE_DEGENERATE = 0.70, 0.50
L0_LO, L0_HI = 10, 200          # trained with top_k 50; an order of magnitude either side is fine


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-id", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--release", default="llama_scope_lxr_8x")
    ap.add_argument("--sae-id", default="l8r_8x")
    ap.add_argument("--layer", type=int, default=8)
    ap.add_argument("--pinned-dir", default="/work/jvl210002/migration/hf_home/hub/"
                    "models--OpenMOSS-Team--Llama3_1-8B-Base-LXR-8x/snapshots/"
                    "8dbc1d85edfced43081c03c38b05514dbab1368b/Llama3_1-8B-Base-L8R-8x")
    ap.add_argument("--packs", default="/scratch/juno/jvl210002/ase2026/full/packs_orig_swapsubset.jsonl")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from sae_lens import SAE
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{TAG} loading SAE {args.release}/{args.sae_id}", flush=True)
    sae = SAE.from_pretrained(release=args.release, sae_id=args.sae_id, device=dev, dtype="float32")
    cfg = sae.cfg
    print(f"{TAG} SAE d_in={getattr(cfg,'d_in',None)} d_sae={getattr(cfg,'d_sae',None)} "
          f"hook={getattr(cfg,'hook_name',None)}", flush=True)

    # --- provenance: does the fetched dictionary equal the revision configs/dictionaries.yaml pins? ---
    prov = {"pinned_dir": args.pinned_dir, "checked": False}
    pin = Path(args.pinned_dir, "checkpoints", "final.safetensors")
    if pin.exists():
        from safetensors.torch import load_file
        w = load_file(str(pin))
        sd = sae.state_dict()
        pairs = [("encoder.weight", "W_enc"), ("decoder.weight", "W_dec")]
        res = {}
        for a, b in pairs:
            if a in w and b in sd:
                A = w[a].float()
                B = sd[b].float().cpu()
                if A.shape != B.shape:
                    A = A.T
                if A.shape != B.shape:
                    res[a] = {"shape_match": False}
                    continue
                # Raw equality is the WRONG test and the first run showed why: max|diff| came back at
                # 4.57 on the encoder. Llama Scope normalises activations dataset-wise, and the loader
                # folds that scaling into the weights, so the two copies are expected to differ by a
                # SCALAR. Test proportionality instead -- identical direction, one constant ratio --
                # which distinguishes "same dictionary, different convention" from "different weights".
                num = float((A * B).sum())
                den = float((A * A).sum()) + 1e-12
                k = num / den                       # least-squares scalar taking A to B
                resid = float(((B - k * A) ** 2).sum() / (float((B * B).sum()) + 1e-12))
                cos = num / ((float((A * A).sum()) ** 0.5) * (float((B * B).sum()) ** 0.5) + 1e-12)
                res[a] = {"shape_match": True,
                          "exact_allclose": bool(torch.allclose(A, B, atol=1e-3)),
                          "best_scalar_k": k, "relative_residual_after_scaling": resid,
                          "cosine": cos,
                          "same_dictionary_up_to_scale": bool(resid < 1e-3 and cos > 0.999)}
        prov = {"pinned_dir": args.pinned_dir, "checked": True, "tensors": res}
        print(f"{TAG} provenance vs pinned OpenMOSS revision: {res}", flush=True)

    print(f"{TAG} loading {args.model_id}", flush=True)
    cache = os.path.join(os.environ["HF_HOME"], "hub") if os.environ.get("HF_HOME") else None
    tok = AutoTokenizer.from_pretrained(args.model_id, cache_dir=cache)
    model = AutoModelForCausalLM.from_pretrained(args.model_id, cache_dir=cache,
                                                 dtype=torch.bfloat16, device_map={"": 0})
    model.eval()

    packs = [json.loads(l) for l in open(args.packs) if l.strip()][: args.n]
    texts = [Path(r["java_path"]).read_text()[:3000] for r in packs]
    print(f"{TAG} {len(texts)} code prompts from this project's own stimuli", flush=True)

    tot_num = tot_den = 0.0
    l0s, fired = [], torch.zeros(int(getattr(cfg, "d_sae")), dtype=torch.bool)
    with torch.no_grad():
        for t in texts:
            enc = tok(t, return_tensors="pt", truncation=True, max_length=1024).to(dev)
            out = model(**enc, output_hidden_states=True, use_cache=False)
            # EXCLUDE BOS. Llama's attention-sink token carries a residual norm an order of
            # magnitude above every other position and the SAE reconstructs it badly, so including it
            # dominates the sum of squares: the same activations scored FVE -2100.8 with it and +0.444
            # without (jobs 413862 vs 413871). This correction applies to E1/E2/E7 too, not just here.
            x = out.hidden_states[args.layer + 1][0].float()[1:]    # resid_post of `layer`, BOS dropped
            z = sae.encode(x)
            xh = sae.decode(z)
            tot_num += float(((x - xh) ** 2).sum())
            tot_den += float(((x - x.mean(0, keepdim=True)) ** 2).sum())
            l0s.append(float((z > 0).float().sum(-1).mean()))
            fired |= (z > 0).any(0).cpu()
    fve = 1.0 - tot_num / tot_den
    l0 = float(sum(l0s) / len(l0s))
    dead = 1.0 - float(fired.float().mean())
    verdict = ("DICT-HEALTHY" if fve >= FVE_HEALTHY and L0_LO <= l0 <= L0_HI else
               "DICT-DEGENERATE" if fve < FVE_DEGENERATE or l0 < 1 else "DICT-MARGINAL")
    rep = {"model": args.model_id, "release": args.release, "sae_id": args.sae_id, "layer": args.layer,
           "n_prompts": len(texts), "fve": fve, "l0": l0, "dead_feature_share_on_sample": dead,
           "thresholds": {"fve_healthy": FVE_HEALTHY, "fve_degenerate": FVE_DEGENERATE,
                          "l0_range": [L0_LO, L0_HI]},
           "verdict": verdict, "provenance": prov}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rep, indent=1, default=str))
    print(f"\n{TAG} FVE {fve:.4f} · L0 {l0:.1f} (trained top_k 50) · "
          f"dead-on-sample {dead:.3f} of {int(getattr(cfg,'d_sae'))}")
    print(f"{TAG} VERDICT {verdict}")
    print(f"{TAG} wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
