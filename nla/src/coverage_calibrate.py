"""Energy-match across TOKEN COVERAGE, so a coverage comparison is not a magnitude comparison.

THE PROBLEM THIS SOLVES. Every banked steering result wrote at ONE token. Widening to the
annotated identifier spans is a ~44x increase in edited positions (median 44 tokens, 18.4% of the
prompt, vs 1). P0.2 already ran the uncontrolled version of this experiment: widening to all reply
positions at unchanged alpha drove the parse rate to 0.100 — it destroyed generation rather than
under-delivering, and the result was uninterpretable as a coverage test.

So the same discipline as the depth axis: match on DELIVERED PERTURBATION, measured, not on the
coefficient. r(alpha, coverage) = || h_final^steered - h_final^unsteered || / || h_final^unsteered ||,
read at the LAST PROMPT TOKEN in every arm — the position the answer is generated from, and the
only position all coverage settings have in common. Median over items.

FROZEN RULE (before any generation, consulting no outcome): for each coverage setting, the matched
alpha is the grid point minimising |log r(alpha, coverage) - log r(1.0, last_prompt)|. Refuse and
widen the grid if the best ratio falls outside [0.5, 2.0].

Note the asymmetry this exposes and does not hide: matching delivered perturbation AT ONE POSITION
across settings that edit different numbers of positions is a choice. The alternative — matching
total injected energy across all edited positions — would make the wide arm far weaker at the
read-out site. Both are defensible; this one is chosen because the question is "does editing the
identifier sites change what the answer position ends up holding", so the answer position is where
delivery should be equalised. Stated here so it is auditable rather than implicit.
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

from steer_run import HOSTS, build_user, load_pairs  # noqa: E402
from steer import ActivationSteerer, SteerSpec  # noqa: E402
from span_positions import span_token_positions  # noqa: E402

SEED = 20260724
REF_ALPHA = 1.0
ALPHAS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="gemma12b", choices=sorted(HOSTS))
    ap.add_argument("--items", type=int, default=12)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import random
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(SEED); np.random.seed(SEED)

    hf_id, layer = HOSTS[args.host]
    out = _PROJ / f"data/nla/p0/steerv2/{args.host}"; out.mkdir(parents=True, exist_ok=True)
    bank = np.load(out / "multilayer_bank.npz", allow_pickle=True)
    D = bank["deltas"].astype(np.float32)
    idx = {str(k): i for i, k in enumerate(bank["item_ids"])}

    tokz = AutoTokenizer.from_pretrained(hf_id)
    model = AutoModelForCausalLM.from_pretrained(hf_id, dtype=torch.bfloat16,
                                                 device_map=args.device).eval()
    pairs = load_pairs(None, random.Random(SEED))[:args.items]
    st = ActivationSteerer(model, layer)
    print(f"[cov] {args.host} L{layer} · {len(pairs)} items · ref alpha {REF_ALPHA}", flush=True)

    def v3(sid: str) -> torch.Tensor:
        i = idx[sid]
        return torch.from_numpy(((D.sum(axis=0)[layer] - D[i][layer]) / (D.shape[0] - 1)).copy())

    def disp(p, positions, alpha) -> float | None:
        user = build_user(p["code_l1b"], p["call_l1b"])
        ids = tokz.apply_chat_template([{"role": "user", "content": user}], tokenize=True,
                                       add_generation_prompt=True, return_dict=False)
        x = torch.tensor([ids], device=model.device)
        st.set_spec(None)
        with torch.no_grad():
            base = model(x, output_hidden_states=True).hidden_states[-1][0, -1].float()
        st.set_spec(SteerSpec(delta=v3(p["snippet_id"]), alpha=alpha, positions=positions),
                    prompt_len=len(ids), max_total=len(ids) + 1)
        with torch.no_grad():
            got = model(x, output_hidden_states=True).hidden_states[-1][0, -1].float()
        st.set_spec(None)
        return float((got - base).norm() / base.norm().clamp_min(1e-12))

    span_pos, ntok = {}, {}
    for p in pairs:
        pos, dg = span_token_positions(tokz, build_user(p["code_l1b"], p["call_l1b"]),
                                       p["code_l1b"], p.get("id_spans_l1b") or [])
        if pos:
            span_pos[p["snippet_id"]] = pos; ntok[p["snippet_id"]] = len(pos)
    print(f"[cov] id_spans mapped {len(span_pos)}/{len(pairs)} · median "
          f"{int(np.median(list(ntok.values()))) if ntok else 0} tokens/item", flush=True)

    COVERAGES = {"last_prompt": lambda p: "last_prompt",
                 "id_spans": lambda p: span_pos.get(p["snippet_id"])}
    curves: dict[str, dict[str, float]] = {}
    for cname, resolve in COVERAGES.items():
        curves[cname] = {}
        for a in ALPHAS:
            rs = [disp(p, resolve(p), a) for p in pairs if resolve(p)]
            rs = [r for r in rs if r is not None]
            curves[cname][f"{a}"] = float(np.median(rs)) if rs else None
            print(f"[cov] {cname:<12} alpha={a:<5} median r {curves[cname][f'{a}']:.4f}", flush=True)
    st.close()

    ref = curves["last_prompt"][f"{REF_ALPHA}"]
    matched = {}
    for cname, c in curves.items():
        ks = [k for k in c if c[k]]
        j = int(np.argmin([abs(np.log(max(c[k], 1e-9)) - np.log(max(ref, 1e-9))) for k in ks]))
        matched[cname] = {"alpha": float(ks[j]), "r": c[ks[j]],
                          "ratio": c[ks[j]] / ref if ref else None}

    rep = {"experiment": "coverage_energy_match", "host": args.host, "layer": layer,
           "seed": SEED, "reference": {"coverage": "last_prompt", "alpha": REF_ALPHA, "r": ref},
           "n_items": len(pairs), "tokens_per_item_median":
               int(np.median(list(ntok.values()))) if ntok else None,
           "curves": curves, "matched": matched,
           "read_position": "last prompt token (common to every coverage setting)",
           "finished_utc": datetime.now(timezone.utc).isoformat()}
    (out / "coverage_match.json").write_text(json.dumps(rep, indent=2))
    print("\n[cov] matched alphas at equal delivered displacement (read at last prompt token):")
    for k, v in matched.items():
        print(f"[cov]   {k:<12} alpha {v['alpha']:<6} r {v['r']:.4f}  ratio {v['ratio']:.2f}")
    bad = [k for k, v in matched.items() if not (0.5 <= (v["ratio"] or 0) <= 2.0)]
    if bad:
        print(f"[cov] REFUSE: {bad} outside the [0.5, 2.0] acceptance band — widen ALPHAS")
        return 1
    print(f"[cov] -> {out / 'coverage_match.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
