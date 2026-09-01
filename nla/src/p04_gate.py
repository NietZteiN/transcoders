"""P0.4 validation gate — does the steerer write where the extractor reads?

WHY THIS EXISTS. `extract.ActivationExtractor` hooks `layers[K]` and `steer.ActivationSteerer`
hooks `layers[K]`, through the same path-probing helper, and both files carry a comment saying
"keep in sync". Nothing has ever checked it. That was tolerable while K was the constant 20 in
both places; P0.4 makes K a flag, and an off-by-one between the read site and the write site
would produce a complete, plausible, entirely mislabelled depth curve. P0.1's gate precedent
applies: if this fails, P0.4 is not reported at all.

WHAT IS ACTUALLY CHECKED. Three forward hooks are registered on `layers[L]` in the same order
`steer_run.main` registers them — extractor first, steerer second — plus a third that captures
the block's final output. Because the extractor's hook returns None it does not modify the
output, so with that ordering the extractor sees the PRE-steer tensor and the third hook sees
the POST-steer tensor. The gate then requires, at the steered position,

    post - pre  ==  alpha * ||pre|| * delta/||delta||

to within 1e-3 relative, and requires every other position to be untouched. A wrong layer index
gives a residual of 1.0 (nothing written where we look); a wrong position gives an untouched
target and a touched neighbour; a broken norm convention gives the right direction at the wrong
scale. All three are distinguishable in the output.

DTYPE. The gate runs in fp32, where 1e-3 is a meaningful tolerance. The production run is bf16,
whose 8-bit mantissa puts the same residual around 2e-3 on arithmetic grounds alone — so the
bf16 pass is reported as an observation, not as the gate. Layer indexing is dtype-independent;
this is a test of *which tensor*, not of precision.

Env `nla-mi`, one GPU, ~2 min.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_NLA_ROOT = _HERE.parent
_PROJ = _NLA_ROOT.parent
sys.path.insert(0, str(_NLA_ROOT / "vendor" / "nla-repo"))
sys.path.insert(0, str(_HERE))

# The gate is host-agnostic — it checks that the extractor and the steerer resolve to the SAME
# decoder block, which is a structural question, not a Qwen one. Made a flag for the Gemma port,
# where `_layers()` had to learn `model.language_model.layers` and the two files could silently
# drift apart. Default preserves the original behaviour.
HOSTS = {
    "qwen7b":   ("Qwen/Qwen2.5-7B-Instruct", "6,13,20"),
    "gemma12b": ("google/gemma-3-12b-it", "8,20,32"),
    "gemma27b": ("google/gemma-3-27b-it", "10,25,41"),
}
TARGET_MODEL = "Qwen/Qwen2.5-7B-Instruct"
SEED = 20260724
TOL = 1e-3
ALPHA = 1.0
PROBE = ("You are an expert software engineer. Read this function and predict its output.\n\n"
         "def f(n):\n    return n * 2 + 1\n\nWhat is the exact output of `f(3)`?")


def check(ex, steerer_cls, spec_cls, layer: int, ids, torch) -> dict:
    """One layer. Returns the residual and the position bookkeeping."""
    layers = ex._layers()

    # Registration order must mirror steer_run.main: extractor's hook, then the steerer's.
    # Re-registering the extractor moves it to the END of the hook list, so it is torn down
    # and rebuilt first for every layer rather than reused in place.
    ex._handle.remove()
    ex.layer_index = layer
    ex._captured = None
    ex._handle = layers[layer].register_forward_hook(ex._hook)      # 1st — reads
    st = steerer_cls(ex.model, layer)                               # 2nd — writes

    post = {}

    def capture_post(_m, _i, output):                               # 3rd — observes the result
        h = output[0] if isinstance(output, tuple) else output
        post["h"] = h.detach().clone()

    h_post = layers[layer].register_forward_hook(capture_post)

    try:
        n = int(ids.shape[1])
        g = torch.Generator().manual_seed(SEED + layer)
        delta = torch.randn(ex.d_model, generator=g)                # deliberately NOT unit norm:
        st.set_spec(spec_cls(delta=delta, alpha=ALPHA,              # the hook must normalize it
                             positions="last_prompt"),
                    prompt_len=n, max_total=n + 1)
        with torch.no_grad():
            ex.model(input_ids=ids)

        pre = ex._captured.float()[0]                               # [T, d] pre-steer
        got = post["h"].float()[0]                                  # [T, d] post-steer
        t = n - 1                                                   # "last_prompt"

        d_hat = (delta / delta.norm()).to(got.device, torch.float32)
        want = ALPHA * pre[t].norm() * d_hat
        obs = got[t] - pre[t]
        resid = float((obs - want).norm() / want.norm().clamp_min(1e-12))

        # Nothing else may have moved. A position error shows up here and nowhere else.
        other = torch.cat([got[:t] - pre[:t], got[t + 1:] - pre[t + 1:]], dim=0)
        other_max = float(other.abs().max()) if other.numel() else 0.0

        return {
            "layer": layer,
            "residual_rel": resid,
            "passed": resid <= TOL and other_max == 0.0 and st.n_positions_written == 1,
            "positions_written": int(st.n_positions_written),
            "hook_calls": int(st.n_hook_calls),
            "target_position": t,
            "untouched_max_abs_delta": other_max,
            "applied_norm": float(obs.norm()),
            "expected_norm": float(want.norm()),
            "act_norm_at_target": float(pre[t].norm()),
        }
    finally:
        h_post.remove()
        st.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", choices=sorted(HOSTS), default="qwen7b")
    ap.add_argument("--layers", default=None,
                    help="comma-separated layer indices; defaults to the host's spread")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=str(_PROJ / "data/nla/p0/p04/p04_gate.json"))
    args = ap.parse_args()

    import torch
    from extract import ActivationExtractor
    from steer import ActivationSteerer, SteerSpec

    model_id, default_layers = HOSTS[args.host]
    want_layers = [int(x) for x in (args.layers or default_layers).split(",")]
    report = {"experiment": "p0.4_layer_indexing_gate",
              "prereg": "log/nla-harness/2026-08-28_p04-depth-prereg.md",
              "host": args.host,
              "model": model_id, "seed": SEED, "alpha": ALPHA, "tolerance_rel": TOL,
              "probe_chars": len(PROBE), "passes": {}}

    for dtype, label, is_gate in ((torch.float32, "fp32", True),
                                  (torch.bfloat16, "bf16", False)):
        ex = ActivationExtractor(model_id, want_layers[0], device=args.device, dtype=dtype)
        # `return_tensors="pt"` hands back a BatchEncoding on transformers 5.x, not a tensor.
        # Mirror steer_run.gen exactly instead: ask for the id list and wrap it ourselves, so
        # the gate tokenizes the way the production path does.
        tok_ids = ex.tokenizer.apply_chat_template(
            [{"role": "user", "content": PROBE}], tokenize=True,
            add_generation_prompt=True, return_dict=False)
        ids = torch.tensor([tok_ids], device=ex.model.device)
        rows = [check(ex, ActivationSteerer, SteerSpec, L, ids, torch) for L in want_layers]
        report["passes"][label] = {"is_gate": is_gate, "layers": rows,
                                   "all_passed": all(r["passed"] for r in rows)}
        for r in rows:
            print(f"[gate:{label}] layer {r['layer']:>2} residual {r['residual_rel']:.2e} "
                  f"· wrote {r['positions_written']} pos at t={r['target_position']} "
                  f"· others moved {r['untouched_max_abs_delta']:.1e} "
                  f"· {'PASS' if r['passed'] else 'FAIL'}", flush=True)
        ex.close()
        del ex
        torch.cuda.empty_cache()

    report["verdict"] = "PASS" if report["passes"]["fp32"]["all_passed"] else "FAIL"
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\n[gate] VERDICT {report['verdict']} (fp32 is the gate; bf16 is an observation)")
    print(f"[gate] wrote {out}")
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
