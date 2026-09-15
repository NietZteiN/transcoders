"""Bank a host's own L0 / L1b greedy replies on the 60 items — the `traces.jsonl` every W-family
readout scores against.

`trace_llr.py` produced these for gemma12b as the first stage of a larger run. A new host (Phase B:
gemma4b) needs the same file and nothing else from that script, so this is the trace stage on its
own. Same prompt (`build_user`), same chat template, same greedy budget `MAX_NEW_GEN` (1100 — a
smaller budget silently zeroes the control, 2026-08-30), same row schema, so `nla_tiers.py`,
`nla_writeback.py`, `nla_heads.py` and the Phase-B gate consume it unchanged.

The readout `G_sum` is the teacher-forced log-prob of the host's OWN clean (L0) reply under the
L1b prompt: a host must score its own replies, never another host's, which is why this is
per-host data and not shared.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_PROJ = _HERE.parent.parent
sys.path.insert(0, str(_HERE))
from steer_run import (BANKED_DATASETS, HOSTS, MAX_NEW_GEN, SEED, build_user,  # noqa: E402
                       graded, load_pairs)
from capture_core import JsonlSink  # noqa: E402
import random  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma4b", choices=sorted(HOSTS))
    ap.add_argument("--allow-banked-host", action="store_true")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    # H-S14 (2026-09-13). Two opt-in flags; both defaults reproduce the banked traces exactly.
    #   --datasets: the banked 60-item pool is `dataset_a,dataset_b`; `dataset_c` is the 257-snippet
    #     HumanEval-X pool built by build_dataset_c.py, requested explicitly and never inherited.
    #   --max-new-gen: H-A8 proved 1100 truncates ~20 % of answers into scored-wrong non-responses
    #     (parsed 0.800 -> 0.967 at 2600, p = 0.0020), so a NEW screen must not reuse that budget --
    #     screening at 1100 would misclassify items as L0-incorrect purely for running out of tokens,
    #     and every one of those is an item silently removed from the flippable pool.
    ap.add_argument("--datasets", default=",".join(BANKED_DATASETS))
    ap.add_argument("--max-new-gen", type=int, default=None)
    ap.add_argument("--max-hours", type=float, default=None)
    ap.add_argument("--fast-host", action="store_true",
                    help="load the banked text-only checkpoint (same loader as nla_accuracy.py); the "
                         "HF multimodal path measured ~144 s per 2600-token generation vs ~17.5 s here")
    args = ap.parse_args()
    datasets = tuple(d.strip() for d in args.datasets.split(",") if d.strip())
    max_new = int(args.max_new_gen or MAX_NEW_GEN)
    if args.model == "qwen7b" and not args.allow_banked_host:
        print("[TR] REFUSED: qwen7b is a Chinese model; pass --allow-banked-host."); return 2
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_name, _ = HOSTS[args.model]
    out = Path(args.out_dir or _PROJ / "data/nla/p0/trace_llr" / args.model)
    out.mkdir(parents=True, exist_ok=True)
    if args.fast_host:
        # 2026-09-14: the HF path above drags in Gemma-3's multimodal wrapper and measured ~144 s per
        # 2600-token generation on an H100 NVL (job 394532), against ~17.5 s for the text-only path
        # nla_accuracy.py uses -- 20 h vs 2.5 h for a 257-item screen. It is also the MORE consistent
        # choice: every accuracy number this screen feeds (H-A8 at 2600) came through `load_host`, and
        # nla_ml_gate.py:251 asserts the two paths' prompt ids agree, so no token id can move.
        if args.model != "gemma4b":
            print("[TR] REFUSED: --fast-host is wired to the banked gemma4b text checkpoint only")
            return 2
        from gemma_text import load_gemma_text, load_tokenizer
        ckpt = _PROJ / "data/nla/ml/gemma4b/host_text"
        if not ckpt.exists():
            print(f"[TR] REFUSED: no text checkpoint at {ckpt}"); return 2
        tokz = load_tokenizer(ckpt)
        model = load_gemma_text(ckpt, dtype=torch.bfloat16, device=args.device).eval()
        print(f"[TR] fast host: {ckpt}", flush=True)
    else:
        tokz = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16,
                                                     device_map=args.device).eval()
    pairs = load_pairs(args.limit, random.Random(SEED), datasets=datasets)
    print(f"[TR] {len(pairs)} pairs · {model_name} · datasets {datasets} · max_new_gen {max_new}",
          flush=True)
    sink = JsonlSink(out / "traces.jsonl", key_field="snippet_id")
    done = sink.done_keys()
    t0 = time.time()
    for i, p in enumerate(pairs, 1):
        sid = p["snippet_id"]
        if sid in done:
            continue
        row = {"snippet_id": sid, "truth": p["truth"]}
        for tier, code, call in (("l0", p["code_l0"], p["call_l0"]),
                                 ("l1b", p["code_l1b"], p["call_l1b"])):
            ids = list(tokz.apply_chat_template(
                [{"role": "user", "content": build_user(code, call)}],
                tokenize=True, add_generation_prompt=True, return_dict=False))
            with torch.no_grad():
                o = model.generate(torch.tensor([ids], device=model.device),
                                   max_new_tokens=max_new, do_sample=False,
                                   pad_token_id=tokz.eos_token_id)
            rep_ids = o[0][len(ids):].tolist()
            rep = tokz.decode(rep_ids, skip_special_tokens=True)
            got, ok = graded(rep, p["truth"])
            row.update({f"{tier}_prompt_ids": ids, f"{tier}_reply_ids": rep_ids,
                        f"{tier}_reply": rep, f"{tier}_answer": got, f"{tier}_correct": ok,
                        f"{tier}_parsed": got is not None})
        sink.append(row)
        print(f"[TR] {i}/{len(pairs)} {sid} L0={row['l0_correct']} L1b={row['l1b_correct']} "
              f"n_tok={len(row['l0_reply_ids'])} · {(time.time()-t0)/60:.1f} min", flush=True)
        if args.max_hours and time.time() - t0 > args.max_hours * 3600:
            print(f"[TR] wall-clock stop after {i} items; JsonlSink resumes from here", flush=True)
            break
    sink.close()
    rows = [json.loads(l) for l in open(out / "traces.jsonl")]
    acc0 = float(np.mean([r["l0_correct"] for r in rows]))
    acc1 = float(np.mean([r["l1b_correct"] for r in rows]))
    flip = [r["snippet_id"] for r in rows if r["l0_correct"] and not r["l1b_correct"]]
    summ = {"experiment": "host_traces", "host": args.model, "model": model_name, "n": len(rows),
            "datasets": list(datasets), "max_new_gen": max_new,
            "acc_l0": acc0, "acc_l1b": acc1, "flippable": flip, "n_flippable": len(flip),
            "parsed_l0": float(np.mean([r["l0_parsed"] for r in rows])),
            "parsed_l1b": float(np.mean([r["l1b_parsed"] for r in rows])),
            "n_tok_l0_median": float(np.median([len(r["l0_reply_ids"]) for r in rows])),
            "finished_utc": datetime.now(timezone.utc).isoformat(), "argv": sys.argv}
    json.dump(summ, open(out / "traces_summary.json", "w"), indent=2)
    print(f"[TR] acc L0 {acc0:.3f} · L1b {acc1:.3f} · flippable {len(flip)}/{len(rows)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
