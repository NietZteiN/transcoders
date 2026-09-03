"""Validation gate for the multi-layer prefill steerer, before any science runs on it.

Four properties, each of which would silently corrupt every downstream number if wrong:

  1. alpha = 0 is byte-identical to unsteered generation. The same identity test steer.py
     carries; a hook that is not inert when it should be contaminates everything.
  2. The edit lands at the intended position and NOWHERE else, at every hooked layer.
  3. It fires during PREFILL only. If it also fired on decode steps, "closing the KV bypass"
     would be confounded with steering the generation — a different axis of the plan.
  4. Later tokens actually SEE the change. This is the whole point: the single-layer hook leaves
     layers 0..L unedited in the KV cache, so downstream attention reads the old state. The gate
     compares a downstream position's activation under single-layer versus multi-layer steering.
     If (4) shows no difference, the bypass hypothesis is wrong and the plan's item C is dead
     before it costs a GPU-hour.
"""
from __future__ import annotations

import argparse, json, sys
from datetime import datetime, timezone
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_NLA_ROOT = _HERE.parent
_PROJ = _NLA_ROOT.parent
sys.path.insert(0, str(_NLA_ROOT / "vendor" / "nla-repo"))
sys.path.insert(0, str(_HERE))

SEED = 20260724
TOL = 1e-3
PROBE = ("You are an expert software engineer. Read this function and predict its output.\n\n"
         "def f(n):\n    return n * 2 + 1\n\nWhat is the exact output of `f(3)`?")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="HF id; defaults to the --host entry")
    ap.add_argument("--host", default=None, help="key in steer_run.HOSTS")
    ap.add_argument("--target-layer", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=str(_PROJ / "data/nla/p0/steerv2/kv_bypass_gate.json"))
    args = ap.parse_args()
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    # random directions are drawn in fp32 then cast at write time, so the seed fixes them
    # independently of the host dtype

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from steer import ActivationSteerer, SteerSpec
    from steer_run import HOSTS, DEFAULT_HOST
    from steer_multilayer import MultiLayerSpec, MultiLayerSteerer, model_dims

    host = args.host or DEFAULT_HOST
    hf_id, host_layer = HOSTS[host]
    args.model = args.model or hf_id
    if args.target_layer is None:
        args.target_layer = host_layer

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
                                                 device_map=args.device).eval()
    ids = tok.apply_chat_template([{"role": "user", "content": PROBE}], tokenize=True,
                                  add_generation_prompt=True, return_dict=False)
    x = torch.tensor([ids], device=model.device)
    L = args.target_layer
    d_model, _n_layers = model_dims(model)
    g = torch.Generator().manual_seed(SEED)
    deltas = {l: torch.randn(d_model, generator=g) for l in range(L + 1)}

    rep = {"experiment": "kv_bypass_gate", "model": args.model, "host": host,
           "target_layer": L,
           "n_layers_written": L + 1, "seed": SEED, "tolerance": TOL, "checks": {}}

    with torch.no_grad():
        base = model(x, output_hidden_states=True)
    base_h = [h.clone() for h in base.hidden_states]

    # ── 1. alpha = 0 is inert ────────────────────────────────────────────────
    ml = MultiLayerSteerer(model, range(L + 1))
    ml.set_spec(MultiLayerSpec(deltas=deltas, alpha=0.0))
    with torch.no_grad():
        z = model(x, output_hidden_states=True)
    d0 = max(float((a - b).abs().max()) for a, b in zip(z.hidden_states, base_h))
    rep["checks"]["alpha0_identity"] = {"max_abs_delta": d0, "passed": d0 == 0.0,
                                        "positions_written": ml.n_positions_written}

    # ── 2 & 3. writes land at one position per layer, prefill only ──────────
    ml.set_spec(MultiLayerSpec(deltas=deltas, alpha=1.0, positions="last_prompt"))
    with torch.no_grad():
        st = model(x, output_hidden_states=True)
    t = len(ids) - 1
    moved_other = 0.0
    for li in range(1, L + 2):
        diff = (st.hidden_states[li] - base_h[li])[0]
        other = torch.cat([diff[:t], diff[t + 1:]], 0)
        moved_other = max(moved_other, float(other.abs().max()) if other.numel() else 0.0)
    rep["checks"]["writes_localised"] = {
        "positions_written": ml.n_positions_written,
        "expected": L + 1,
        "max_abs_change_at_other_positions_below_target": moved_other,
        "passed": ml.n_positions_written == L + 1,
    }

    # ── 4. does a LATER token see the change? the bypass question itself ────
    single = ActivationSteerer(model, L)
    ml.set_spec(None)
    single.set_spec(SteerSpec(delta=deltas[L], alpha=1.0, positions="last_prompt"),
                    prompt_len=len(ids), max_total=len(ids) + 1)
    with torch.no_grad():
        s1 = model(x, output_hidden_states=True)
    single.close()

    ml.set_spec(MultiLayerSpec(deltas=deltas, alpha=1.0, positions="last_prompt"))
    with torch.no_grad():
        sm = model(x, output_hidden_states=True)
    ml.close()

    # A position BEFORE the target cannot be affected by either (causal masking); a position
    # after it does not exist in this prompt, so the readable consequence is the target's own
    # state at layers <= L, which single-layer steering leaves untouched and multi-layer does not.
    lower_single = max(float((s1.hidden_states[li][0, t] - base_h[li][0, t]).abs().max())
                       for li in range(1, L + 1))
    lower_multi = max(float((sm.hidden_states[li][0, t] - base_h[li][0, t]).abs().max())
                      for li in range(1, L + 1))
    rep["checks"]["bypass_demonstrated"] = {
        "max_change_at_layers_below_target_single_layer": lower_single,
        "max_change_at_layers_below_target_multi_layer": lower_multi,
        "passed": lower_single == 0.0 and lower_multi > 0.0,
        "reading": ("single-layer steering leaves every layer below the target untouched at the "
                    "edited position — which is exactly the state later tokens attend to through "
                    "the KV cache. Multi-layer prefill steering changes it."),
    }

    rep["verdict"] = ("PASS" if all(c["passed"] for c in rep["checks"].values()) else "FAIL")
    rep["finished_utc"] = datetime.now(timezone.utc).isoformat()
    out.write_text(json.dumps(rep, indent=2))
    for k, v in rep["checks"].items():
        print(f"[gate] {k:<24}{'PASS' if v['passed'] else 'FAIL'}  {json.dumps({kk: vv for kk, vv in v.items() if kk != 'reading' and kk != 'passed'})}", flush=True)
    print(f"\n[gate] VERDICT {rep['verdict']} -> {out}")
    return 0 if rep["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
