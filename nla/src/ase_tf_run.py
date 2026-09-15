"""H-R1a stage 2 — run the ASE-2026 T/F verification protocol on a PERMITTED model.

Pre-registered in log/nla-harness/2026-09-14_ase-replication-prereg.md.

THE PROTOCOL IS THEIRS, THE RUNNER IS OURS. `build_counterfactual_instruction`,
`parse_predicted_labels` and `score_case_predictions` are imported from the artifact unmodified, and
the prompt is assembled exactly as `models.ModelRunner._build_prompt` does it:

    f"{instruction}\\n\\n```{language}\\n{code}\\n```{answer_prefix}"   with answer_prefix "\\n\\nJSON answer:\\n"

Decoding matches their defaults (`models.py`: temperature 0.7, top_p 1.0, sampled), and Pass@k is
their definition: per CASE, passed at k if any of the first k independent responses matches the
ground-truth label; the score is the proportion of cases passing. Three runs, so Pass@1/2/3.

WHY SAMPLING MATTERS HERE. Pass@k over "independent runs" is only meaningful under sampling; with
greedy decoding the runs would be near-copies (and this host's greedy is not even bit-reproducible
-- 37 % of generations diverge in tokens, measured in H-A13). Sampling at T=0.7 is therefore both
their regime and the only regime in which Pass@2/@3 mean anything.

MODEL. Default `llama8b` (Llama-3.1-8B-Instruct): permitted here, and the artifact's own
`steering/backends/llama_backend.py` shows CodeSteer supports this family, so the later three-way
comparison can run on one host. Qwen/DeepSeek -- their four evaluated models -- are barred.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_PROJ = _HERE.parents[1]
sys.path.insert(0, str(_HERE))
from steer_run import HOSTS  # noqa: E402

TAG = "[TF]"
SEED = 20260724
TEMP, TOP_P = 0.7, 1.0          # their models.py defaults
ANSWER_PREFIX = "\n\nJSON answer:\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifact", default="/scratch/juno/jvl210002/ase2026/LLM-Attention-Fixation_submission")
    ap.add_argument("--packs", default="/scratch/juno/jvl210002/ase2026/packs_humaneval_java.jsonl")
    ap.add_argument("--model", default="llama8b", choices=sorted(HOSTS))
    # --model-id runs a model that is not in HOSTS without editing HOSTS, which ~10 other scripts
    # import and whose contents define the banked item sets. Used for the permitted CODER models
    # (CodeLlama / StarCoder2 / CodeGemma): their paper's four models are all Qwen2.5 or DeepSeek
    # coder models and all barred here, so a permitted coder model is the nearest available analogue
    # and the test of whether the phenomenon is coder-model-specific.
    ap.add_argument("--model-id", default=None, help="HF id/path overriding --model")
    ap.add_argument("--allow-banked-host", action="store_true")
    ap.add_argument("--condition", default="original")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--language", default="java")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-hours", type=float, default=3.0)
    args = ap.parse_args()

    R = Path(args.artifact)
    sys.path.insert(0, str(R))
    import counterfactual_eval as ce            # noqa: E402  (their code, unmodified)

    packs = [json.loads(l) for l in open(args.packs) if l.strip()]
    packs = [p for p in packs if "pack" in p]
    if args.limit:
        packs = packs[:args.limit]
    n_cases = sum(len(p["pack"].get("cases", [])) for p in packs)
    print(f"{TAG} {len(packs)} snippets · {n_cases} cases · condition={args.condition} "
          f"· model={args.model} · runs={args.runs} · T={TEMP} top_p={TOP_P}", flush=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    name = args.model_id or HOSTS[args.model][0]
    low = name.lower()
    if any(v in low for v in ("qwen", "deepseek", "yi-", "glm", "internlm", "baichuan")):
        print(f"{TAG} REFUSED: {name} is a Chinese model; this project does not run them."); return 2
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16,
                                                 device_map=args.device).eval()
    torch.manual_seed(SEED)

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["snippet"] for l in open(out) if l.strip()} if out.exists() else set()
    if done:
        print(f"{TAG} resuming; {len(done)} snippets already scored", flush=True)
    sink = open(out, "a")
    t0 = time.time()

    for i, rec in enumerate(packs, 1):
        sid = rec["snippet"]
        if sid in done:
            continue
        pack = rec["pack"]
        cases = pack.get("cases", [])
        if not cases:
            continue
        # Code comes from the pack's own java_path. The renamed condition is a DIFFERENT packs file,
        # built by running ase_casepacks.py over the renamed corpus -- because their obfuscation runner
        # rebuilds the case pack from the obfuscated variant (obfuscation/main.py:428), so the case
        # expressions carry the renamed method name and the labels are re-validated on that variant.
        # See log/nla-harness/2026-09-14_ase-prereg-amendment.md. A --code-root override existed here
        # and was REMOVED: it would silently reproduce the wrong protocol.
        code = Path(rec["java_path"]).read_text()

        instruction = ce.build_counterfactual_instruction(pack)
        prompt = f"{instruction}\n\n```{args.language}\n{code}\n```{ANSWER_PREFIX}"
        ids = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=True,
                                      add_generation_prompt=True, return_dict=False)
        ids_t = torch.tensor([list(ids)], device=model.device)
        case_ids = [c["case_id"] for c in cases]
        truth = {c["case_id"]: bool(c["expected_bool"]) for c in cases}

        per_run = []
        for r in range(args.runs):
            with torch.no_grad():
                o = model.generate(ids_t, max_new_tokens=args.max_new_tokens, do_sample=True,
                                   temperature=TEMP, top_p=TOP_P,
                                   pad_token_id=tok.eos_token_id)
            text = tok.decode(o[0][len(ids):], skip_special_tokens=True)
            pred, meta = ce.parse_predicted_labels(text, case_ids, strict_json=True)
            scored = ce.score_case_predictions(pack, pred)
            per_run.append({"pred": pred, "parse_mode": meta.get("mode"),
                            "n_parsed": len(pred), "scored": scored,
                            "reply_chars": len(text)})

        # Pass@k, their definition: per case, passed at k if ANY of the first k runs matches truth
        passk = {}
        for k in range(1, args.runs + 1):
            hit = [any(per_run[j]["pred"].get(cid) == truth[cid] for j in range(k)) for cid in case_ids]
            passk[f"pass@{k}"] = float(np.mean(hit))
        row = {"snippet": sid, "condition": args.condition, "n_cases": len(cases),
               "parsed_frac": float(np.mean([r["n_parsed"] / len(cases) for r in per_run])),
               **passk, "runs": per_run}
        sink.write(json.dumps(row) + "\n"); sink.flush()
        print(f"{TAG} {i}/{len(packs)} {sid} cases={len(cases)} "
              + " ".join(f"P@{k}={passk[f'pass@{k}']:.3f}" for k in range(1, args.runs + 1))
              + f" parsed={row['parsed_frac']:.2f} · {(time.time()-t0)/60:.1f} min", flush=True)
        if time.time() - t0 > args.max_hours * 3600:
            print(f"{TAG} wall-clock stop after {i}; re-run to resume", flush=True); break
    sink.close()

    rows = [json.loads(l) for l in open(out) if l.strip()]
    # CASE-weighted, as their Pass@k is a proportion of cases (not of snippets)
    tc = sum(r["n_cases"] for r in rows)
    print(f"\n{TAG} === {args.condition} · {len(rows)} snippets · {tc} cases ===")
    for k in range(1, args.runs + 1):
        key = f"pass@{k}"
        cw = sum(r[key] * r["n_cases"] for r in rows) / tc if tc else float("nan")
        print(f"{TAG}   Pass@{k} (case-weighted) {cw:.4f} · snippet-mean {np.mean([r[key] for r in rows]):.4f}")
    print(f"{TAG}   parsed fraction {np.mean([r['parsed_frac'] for r in rows]):.4f}")
    print(f"{TAG}   reference (their Qwen2.5-7B, HumanEval-X original): P@1 0.7649 P@2 0.8615 P@3 0.8882")
    print(f"{TAG}   chance is 0.50 by construction (negation-paired cases)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
