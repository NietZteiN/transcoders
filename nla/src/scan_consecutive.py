"""Consecutive-position scan: watch layer-20 content accumulate token by token.

Reads EVERY token position across a contiguous window of the flagship L1b prompt
(the code region: function signature -> base cases -> recursive call) and verbalizes
each. The point: the read at position t reflects the whole prefix up to t, so the
descriptions should visibly accumulate structure as the window advances.

Output: data/nla/examples/consecutive_scan.json + printed transcript.
Run (nla-mi env; AV server on :30000):  python -m src.scan_consecutive --device cuda
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_NLA_ROOT = _HERE.parent
_PROJ = _NLA_ROOT.parent
sys.path.insert(0, str(_NLA_ROOT / "vendor" / "nla-repo"))
sys.path.insert(0, str(_HERE))

TARGET_MODEL = "Qwen/Qwen2.5-7B-Instruct"
LAYER_INDEX = 20
SEED = 20260724
N_SCAN = 44          # consecutive positions starting at the first code token

PREAMBLE = (
    "You are an expert software engineer taking part in a code comprehension study. "
    "You will be shown one function. Read it carefully, reason about what it computes "
    "step by step, and then predict the exact output of the given call. Be precise "
    "about values and types, and do not guess without tracing the computation. "
    "Here is the function and the call to evaluate:\n\n"
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--sglang-url", default="http://localhost:30000")
    ap.add_argument("--start", type=int, default=0, help="offset from first code token")
    ap.add_argument("--n", type=int, default=None, help="override N_SCAN")
    args = ap.parse_args()

    import torch
    torch.manual_seed(SEED)
    from extract import ActivationExtractor
    from nla_inference import NLAClient, NLACritic

    with open(_PROJ / "data" / "stimuli" / "dataset_a" / "dataset_a.jsonl") as f:
        row = next(json.loads(l) for l in f
                   if json.loads(l)["snippet_id"] == "JavaScript/63" and json.loads(l)["tier"] == "L1b")

    user = PREAMBLE + row["code"] + "\n\nWhat is the exact output of `smoothArea(14)`?"
    ex = ActivationExtractor(TARGET_MODEL, LAYER_INDEX, device=args.device)
    av = NLAClient(_NLA_ROOT / "data" / "checkpoints" / "av", sglang_url=args.sglang_url)
    ar = NLACritic(_NLA_ROOT / "data" / "checkpoints" / "ar", device=args.device)
    tokz = ex.tokenizer

    templ = tokz.apply_chat_template([{"role": "user", "content": user}],
                                     tokenize=False, add_generation_prompt=True)
    code_off = templ.index(row["code"])
    enc = tokz(templ, return_offsets_mapping=True, add_special_tokens=False)
    offsets = enc["offset_mapping"]
    first_code_pos = next(i for i, (s, e) in enumerate(offsets) if e > code_off and e > s)

    res = ex.extract_chat(user, None, text_id="scan")
    n_scan = args.n or N_SCAN
    start = first_code_pos + args.start
    positions = list(range(start, min(start + n_scan, len(res.positions))))

    rows_out = []
    print(f"scanning {len(positions)} consecutive positions from pos {first_code_pos} "
          f"(first code token); each read sees ONLY the prefix up to that token\n")
    for pos in positions:
        s, e = offsets[pos]
        tok_text = templ[s:e]
        prefix_tail = templ[max(code_off, e - 44):e]          # last chars of what this read has seen
        v = res.activations[pos]
        text = av.generate(v, temperature=0.0, max_new_tokens=150)
        _, cos_rt = ar.score(text, v)
        first_sentence = re.split(r"(?<=[.!?])\s", " ".join(text.split()))[0][:200]
        rows_out.append({"position": int(pos), "token_text": tok_text,
                         "prefix_tail": prefix_tail, "roundtrip_cos": round(float(cos_rt), 3),
                         "read": text})
        print(f"[{pos:>3}] …{prefix_tail!r:>48}  ->  {first_sentence}")

    ex.close()
    suffix = f"_start{args.start}" if args.start else ""
    out = _PROJ / "data" / "nla" / "examples" / f"consecutive_scan{suffix}.json"
    out.write_text(json.dumps({"model": TARGET_MODEL, "layer": LAYER_INDEX, "seed": SEED,
                               "snippet": "JavaScript/63 L1b", "rows": rows_out}, indent=1))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
