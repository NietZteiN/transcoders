"""E3 gate — do the base-trained Llama Scope dictionaries read the INSTRUCT model?

WHY THIS GATES EVERYTHING. The replicated effect lives on `Llama-3.1-8B-Instruct` at tier L2,
peaking at layers 5-9. Every Llama Scope dictionary was trained on `meta-llama/Llama-3.1-8B`
BASE. The registry's own note says "validate base->Instruct transfer" and the charter lists it
among the silent-failure checks. Not a formality: B1 showed a frozen encoder degrades
DIRECTIONALLY across an architecturally identical sibling, losing 0.170 median round-trip cosine
while every dimension matched.

THE COMPARISON DECIDES, NOT THE ABSOLUTE (the B1 principle). An FVE of 0.6 on obfuscated code
means nothing alone — code is far from the dictionary's training distribution. The same
dictionary runs over the same token positions on BOTH hosts and the gate is the paired drop.
Base and Instruct share a tokenizer and every dimension, so token sequences and position indices
are identical: the only variable is the weights.

BASE IS ALSO THE IMPLEMENTATION CONTROL. A wrong normalisation or activation convention would
produce a bad FVE that still looks like a number, and would read as "transfer failed" when it
actually means "the loader is wrong". So the base host must reconstruct WELL — it is the
dictionary's own training distribution. A poor base FVE means stop and fix the loader, not
report a transfer failure.

SITE-CORRECT HOOKS. The two sites are not interchangeable:
  res  hook_point_in = hook_point_out = blocks.<L>.hook_resid_post   (an SAE: reconstructs itself)
  tc   hook_point_in = blocks.<L>.ln2.hook_normalized                (a transcoder:
       hook_point_out = blocks.<L>.hook_mlp_out                       predicts the MLP's output)
Scoring the transcoder against the residual stream would yield a meaningless FVE that still
prints. In HF terms ln2.hook_normalized is the output of `post_attention_layernorm` and
hook_mlp_out is the output of `mlp`.

A NULL ATTACHED TO THE HEADLINE. Every reconstruction is also scored against a FOREIGN
activation — another item's vector at the matched position. A dictionary that reconstructs
foreign activations as well as its own is not reading anything item-specific, whatever its FVE.

Env `nla-mi`: the TC site is not in the SAELens registry, so weights are read straight from
safetensors — the loading path the registry itself documents.
"""
from __future__ import annotations

import argparse, json, random, statistics as st, sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_NLA_ROOT = _HERE.parent
_PROJ = _NLA_ROOT.parent
sys.path.insert(0, str(_HERE))

SEED = 20260724
BASE, INSTRUCT = "meta-llama/Llama-3.1-8B", "meta-llama/Llama-3.1-8B-Instruct"
MAX_FVE_DROP = 0.15      # frozen: the gate is the paired drop, not an absolute floor
MIN_BASE_FVE = 0.50      # frozen: below this the LOADER is suspect, not the transfer


def load_dictionary(site: str, layer: int):
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    tag = "TC" if site == "tc" else "R"
    repo = f"OpenMOSS-Team/Llama3_1-8B-Base-LX{tag}-8x"
    sub = f"Llama3_1-8B-Base-L{layer}{tag}-8x"
    sd = load_file(hf_hub_download(repo_id=repo, filename=f"{sub}/checkpoints/final.safetensors"))
    hp = json.loads(Path(hf_hub_download(repo_id=repo,
                                         filename=f"{sub}/hyperparams.json")).read_text())
    return sd, hp, f"{repo}/{sub}"


def reconstruct(x, sd, hp, torch):
    """Llama Scope forward pass, following hyperparams rather than assuming a convention.

    dataset-wise normalisation: inputs are scaled so the dataset-average norm maps to sqrt(d),
    and the decoder output is scaled back by the OUTPUT statistic — which differs from the input
    one for a transcoder (64.0 vs 3.39 here), so using one for both would silently mis-scale.
    """
    d = hp["d_model"]
    navg = hp.get("dataset_average_activation_norm") or {}
    s_in = (d ** 0.5) / navg.get("in", d ** 0.5)
    s_out = navg.get("out", d ** 0.5) / (d ** 0.5)

    W_e = sd["encoder.weight"].float()
    b_e = sd["encoder.bias"].float()
    W_d = sd["decoder.weight"].float()
    b_d = sd["decoder.bias"].float()

    xn = x * s_in
    pre = xn @ W_e.T + b_e
    thr = hp.get("jump_relu_threshold", 0.0)
    acts = torch.where(pre > thr, pre, torch.zeros_like(pre))      # JumpReLU
    out = acts @ W_d.T
    if hp.get("use_decoder_bias", True):
        out = out + b_d
    return out * s_out, acts


def fve(x, xhat) -> float:
    num = float(((x - xhat) ** 2).sum())
    den = float(((x - x.mean(0, keepdim=True)) ** 2).sum())
    return 1.0 - num / max(den, 1e-12)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=8)
    ap.add_argument("--site", choices=["tc", "res"], default="tc")
    ap.add_argument("--tier", default="L2")
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = Path(args.out or _PROJ / f"data/nla/p0/e3/transfer_{args.site}_L{args.layer}.json")
    out.parent.mkdir(parents=True, exist_ok=True)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from p1b_ladder import load_tier
    from steer_run import build_user

    sd, hp, dict_id = load_dictionary(args.site, args.layer)
    print(f"[e3] {dict_id} · act_fn {hp.get('act_fn')} thr {hp.get('jump_relu_threshold')} "
          f"· norm {hp.get('dataset_average_activation_norm')}", flush=True)
    print(f"[e3] in={hp.get('hook_point_in')}  out={hp.get('hook_point_out')}", flush=True)

    items = load_tier(args.tier, random.Random(SEED))[: args.limit]
    prompts = [build_user(i["code"], i["call"]) for i in items]

    rep = {"experiment": "e3_base_to_instruct_transfer_gate", "seed": SEED,
           "layer": args.layer, "site": args.site, "tier": args.tier, "dictionary": dict_id,
           "n_prompts": len(prompts), "max_fve_drop": MAX_FVE_DROP,
           "min_base_fve": MIN_BASE_FVE,
           "hook_in": hp.get("hook_point_in"), "hook_out": hp.get("hook_point_out"),
           "hosts": {}}

    for label, model_id in (("base", BASE), ("instruct", INSTRUCT)):
        tok = AutoTokenizer.from_pretrained(model_id)
        m = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16,
                                                 device_map=args.device).eval()
        blk = m.model.layers[args.layer]
        cap = {}
        h1 = blk.post_attention_layernorm.register_forward_hook(
            lambda _m, _i, o: cap.__setitem__("in", o.detach()))
        h2 = blk.mlp.register_forward_hook(
            lambda _m, _i, o: cap.__setitem__("mlp_out", o.detach()))

        X, Y = [], []
        for p in prompts:
            ids = tok(p, return_tensors="pt", truncation=True, max_length=4096).to(m.device)
            with torch.no_grad():
                o = m(**ids, output_hidden_states=True)
            resid = o.hidden_states[args.layer + 1][0, -1]
            if args.site == "tc":
                X.append(cap["in"][0, -1].float().cpu())
                Y.append(cap["mlp_out"][0, -1].float().cpu())
            else:
                X.append(resid.float().cpu()); Y.append(resid.float().cpu())
        h1.remove(); h2.remove()
        del m; torch.cuda.empty_cache()

        X = torch.stack(X); Y = torch.stack(Y)
        Yhat, acts = reconstruct(X, sd, hp, torch)
        # foreign null: reconstruct item i's input, score against item i+1's target
        roll = torch.roll(Y, 1, dims=0)
        cs = torch.nn.functional.cosine_similarity(Y, Yhat, dim=1)
        cs_f = torch.nn.functional.cosine_similarity(roll, Yhat, dim=1)
        rep["hosts"][label] = {
            "fve": round(fve(Y, Yhat), 4),
            "fve_foreign_null": round(fve(roll, Yhat), 4),
            "cos_median": round(float(cs.median()), 4),
            "cos_foreign_null": round(float(cs_f.median()), 4),
            "l0_mean": round(float((acts > 0).float().sum(1).mean()), 1),
            "act_norm_mean": round(float(X.norm(dim=1).mean()), 2),
        }
        b = rep["hosts"][label]
        print(f"[e3] {label:<9} FVE {b['fve']:+.4f} (foreign {b['fve_foreign_null']:+.4f}) · "
              f"cos {b['cos_median']:.4f} (foreign {b['cos_foreign_null']:.4f}) · "
              f"L0 {b['l0_mean']} · ||x|| {b['act_norm_mean']}", flush=True)

    bs, ins = rep["hosts"]["base"], rep["hosts"]["instruct"]
    drop = bs["fve"] - ins["fve"]
    rep["fve_drop_base_minus_instruct"] = round(drop, 4)
    if bs["fve"] < MIN_BASE_FVE:
        rep["verdict"] = (f"LOADER SUSPECT — base FVE {bs['fve']:.4f} < {MIN_BASE_FVE} on the "
                          "dictionary's own training distribution; fix the loader before reading "
                          "anything into the transfer number")
    elif drop <= MAX_FVE_DROP:
        rep["verdict"] = "TRANSFER OK — Instruct reconstructs within tolerance of base"
    else:
        rep["verdict"] = (f"TRANSFER DEGRADED — Instruct falls {drop:.4f} below base "
                          f"(tolerance {MAX_FVE_DROP})")
    rep["finished_utc"] = datetime.now(timezone.utc).isoformat()
    out.write_text(json.dumps(rep, indent=2))
    print(f"\n[e3] {rep['verdict']}\n[e3] wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
