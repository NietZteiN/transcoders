"""Screen the whole C0/C1/C2 corpus for items where the model is BEHAVIOURALLY deceived.

WHY THIS EXISTS. N11's one live cell was behavioural coupling: every C2 item where the model
*stated* the injected wrong algorithm also carried a recurrent internal read (3/3), against 8.1%
where it did not. Right direction, right structure, useless n — because the model is almost
never behaviourally deceived (3 of 77 parsed C2 captures). Coupling has no denominator.

The fix is not a bigger read capture. Reads cost ~4 s each and 21 of them per item; a *stated
answer* costs one generation. So screen wide and cheap here, then spend read budget only on the
items that carry behavioural signal plus matched controls.

**No NLA, no AV server, no activation extraction** — just the subject model. That also means
generations can be **batched**, which the read capture cannot be (batching was measured to break
AV determinism: 6.3x faster but only 15% byte-identical, so the read path stays sequential).
Batching needs LEFT padding; `extract.py` deliberately uses RIGHT padding because activation
positions must line up with token indices. Nothing here touches activations, so left padding is
safe — but do not copy this setting into a capture script.

Selection on `stated_hit` is legitimate stratified sampling for the coupling question, since the
read rate is reported *within* stratum (stated vs not-stated). It would not license an unconditional
deception-rate claim, and the screen's own marginal rate is reported so that stays visible.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_NLA_ROOT = _HERE.parent
_PROJ = _NLA_ROOT.parent
sys.path.insert(0, str(_HERE))

from deception_capture import ANSWER_RE, CONDITIONS, PREAMBLE, load_triples  # noqa: E402

TARGET_MODEL = "Qwen/Qwen2.5-7B-Instruct"
SEED = 20260724
MAX_NEW_GEN = 400


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="humaneval,cruxeval")
    ap.add_argument("--limit", type=int, default=None, help="snippets per dataset")
    ap.add_argument("--conditions", default="C1,C2",
                    help="C0 is the untouched source and cannot be 'deceived'; screening it "
                         "costs a third of the budget for a cell we already know is 0/98")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out", default=str(_PROJ / "data/nla/n11/screen.jsonl"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    sys.path.insert(0, str(_HERE))
    from belief_grade import algorithm_claims
    from capture_core import JsonlSink

    torch.manual_seed(SEED)
    rng = random.Random(SEED)

    jobs = []
    for ds in args.datasets.split(","):
        for t in load_triples(ds, args.limit, random.Random(SEED)):
            for c in args.conditions.split(","):
                jobs.append((t, c))
    print(f"[scr] {len(jobs)} generations queued", flush=True)

    sink = JsonlSink(Path(args.out), key_field="task_key")
    done = sink.done_keys()
    jobs = [(t, c) for t, c in jobs if f"{t['dataset']}|{t['snippet']}|{c}" not in done]
    print(f"[scr] {len(jobs)} to go ({len(done)} already done)", flush=True)
    if not jobs:
        return 0

    tok = AutoTokenizer.from_pretrained(TARGET_MODEL)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"          # batched generation only — see module docstring
    model = AutoModelForCausalLM.from_pretrained(
        TARGET_MODEL, dtype=torch.bfloat16, device_map=args.device).eval()

    t0 = time.time()
    n_done = 0
    for i in range(0, len(jobs), args.batch_size):
        chunk = jobs[i:i + args.batch_size]
        prompts = [tok.apply_chat_template(
            [{"role": "user", "content": PREAMBLE + t["src"][c]}],
            tokenize=False, add_generation_prompt=True) for t, c in chunk]
        enc = tok(prompts, return_tensors="pt", padding=True,
                  add_special_tokens=False).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=MAX_NEW_GEN, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        replies = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)

        for (t, c), reply in zip(chunk, replies):
            m = ANSWER_RE.search(reply)
            stated = m.group(1) if m else None
            claims = algorithm_claims(stated or "")
            sink.append({
                "task_key": f"{t['dataset']}|{t['snippet']}|{c}",
                "dataset": t["dataset"], "snippet": t["snippet"], "condition": c,
                "injected_algorithm": t["injected_algorithm"],
                "stated_algorithm": stated, "parsed_answer": bool(m),
                "stated_hit": bool(t["injected_algorithm"] in claims),
                "stated_claims": sorted(claims),
                "reply_chars": len(reply),
            })
        n_done += len(chunk)
        rate = n_done / (time.time() - t0)
        print(f"[scr] {n_done}/{len(jobs)} · {rate*60:.1f}/min · "
              f"ETA {(len(jobs)-n_done)/max(rate,1e-9)/60:.0f} min", flush=True)

    sink.close()
    rows = [json.loads(l) for l in open(args.out) if l.strip()]
    hits = [r for r in rows if r.get("stated_hit")]
    Path(args.out).parent.joinpath("screen_manifest.json").write_text(json.dumps({
        "experiment": "n11_deception_screen", "seed": SEED, "argv": sys.argv,
        "model": TARGET_MODEL, "max_new_gen": MAX_NEW_GEN, "batch_size": args.batch_size,
        "n_generations": len(rows), "n_stated_hit": len(hits),
        "elapsed_hours": round((time.time() - t0) / 3600, 3),
        "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2))
    print(f"[scr] done · {len(rows)} generations · {len(hits)} stated_hit", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
