"""Out-of-the-box probe: do the NLAs read anything meaningful on CODE? (pre-N1 feasibility)

Reads L20 activations on three obfuscation stimuli and verbalizes them:
  1. flagship L0  (JavaScript/63, `fibfib`)        — identifier + last-token reads
  2. flagship L1b (same, disguised as `smoothArea`) — the decoy trap
  3. one Python L2 (control-flow-flattened)         — dispatcher state-var reads
  4. the model's own greedy CoT on the L1b item     — reads over generated reasoning

At trap positions we also compute a first Decoy-Read Margin signal in AR space:
  DRM_AR = cos(AR(decoy gloss), v) − cos(AR(true gloss), v)      (>0 → reads the decoy)

Purely qualitative/feasibility — small n, no stats. Output: data/nla/probe/probe_reads.json
plus a printed transcript. Requires the AV SGLang server on :30000 (see data/sglang_av_server.log).

Run (nla-mi env, GPU for extractor+AR):
  python -m src.probe_code_reads --device cuda
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

# ≥50-token preamble so every code position clears MIN_POSITION (known-noisy early reads).
PREAMBLE = (
    "You are an expert software engineer taking part in a code comprehension study. "
    "You will be shown one function. Read it carefully, reason about what it computes "
    "step by step, and then predict the exact output of the given call. Be precise "
    "about values and types, and do not guess without tracing the computation. "
    "Here is the function and the call to evaluate:\n\n"
)

# Decoy/true glosses for the flagship trap (fibfib disguised as smoothArea)
GLOSS_TRUE = "computing a tribonacci-style recursive integer sequence by summing the three previous terms"
GLOSS_DECOY = "smoothing an area measurement over the last N seconds, like a moving average over time"


def load_snippets() -> dict:
    rows = {}
    with open(_PROJ / "data" / "stimuli" / "dataset_a" / "dataset_a.jsonl") as f:
        for line in f:
            r = json.loads(line)
            rows[(r["snippet_id"], r["tier"])] = r
    flag_l0 = rows[("JavaScript/63", "L0")]
    flag_l1b = rows[("JavaScript/63", "L1b")]
    py_l2 = next(r for (sid, tier), r in rows.items()
                 if tier == "L2" and r["language"] == "python" and r["dispatcher_spans"])
    return {"flag_l0": flag_l0, "flag_l1b": flag_l1b, "py_l2": py_l2}


def spans_to_positions(tok, full_text: str, code_offset: int, spans: list[list[int]]) -> list[tuple[int, str]]:
    """Map code-relative char spans to token positions in the full templated text."""
    enc = tok(full_text, return_offsets_mapping=True, add_special_tokens=False)
    offsets = enc["offset_mapping"]
    out = []
    for a, b in spans:
        a, b = a + code_offset, b + code_offset
        for i, (s, e) in enumerate(offsets):
            if e > s and s < b and e > a:
                out.append((i, full_text[s:e]))
    # dedupe, keep order
    seen, uniq = set(), []
    for p, t in out:
        if p not in seen:
            seen.add(p)
            uniq.append((p, t))
    return uniq


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--sglang-url", default="http://localhost:30000")
    ap.add_argument("--max-reads-per-item", type=int, default=12)
    args = ap.parse_args()

    import torch
    torch.manual_seed(SEED)
    from extract import ActivationExtractor
    from nla_inference import NLAClient, NLACritic

    snippets = load_snippets()
    ex = ActivationExtractor(TARGET_MODEL, LAYER_INDEX, device=args.device)
    av = NLAClient(_NLA_ROOT / "data" / "checkpoints" / "av", sglang_url=args.sglang_url)
    ar = NLACritic(_NLA_ROOT / "data" / "checkpoints" / "ar", device=args.device)

    # Pre-reconstruct the two glosses once (AR-space decoy/true anchors)
    v_true = ar.reconstruct(GLOSS_TRUE).float().cpu().numpy()
    v_decoy = ar.reconstruct(GLOSS_DECOY).float().cpu().numpy()

    def cosv(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    results = []

    def read_positions(label: str, user_content: str, reply: str | None,
                       positions: list[tuple[int, str]], drm: bool) -> None:
        res = ex.extract_chat(user_content, reply, text_id=label)
        print(f"\n{'='*80}\n== {label}  ({len(positions)} reads) ==")
        for pos, ptext in positions[: args.max_reads_per_item]:
            v = res.activations[pos]
            text = av.generate(v, temperature=0.0, max_new_tokens=200)
            mse, cos_rt = ar.score(text, v)
            row = {"item": label, "position": int(pos), "token_text": ptext,
                   "act_norm": float(np.linalg.norm(v)), "roundtrip_cos": float(cos_rt),
                   "read": text}
            if drm:
                c_d, c_t = cosv(v_decoy, v), cosv(v_true, v)
                row.update({"cos_decoy_gloss": c_d, "cos_true_gloss": c_t, "drm_ar": c_d - c_t})
            results.append(row)
            drm_s = f"  DRM_AR={row['drm_ar']:+.3f}" if drm else ""
            print(f"\n[{pos:>4}] tok={ptext!r} ||v||={row['act_norm']:.0f} rt_cos={cos_rt:.2f}{drm_s}")
            print(f"      {text[:400]}")

    tokz = ex.tokenizer

    # ---- 1+2: flagship L0 and L1b, identifier-token reads (+ DRM on both) ----
    for key, label in [("flag_l0", "L0_fibfib"), ("flag_l1b", "L1b_smoothArea")]:
        s = snippets[key]
        user = PREAMBLE + s["code"] + "\n\nWhat is the exact output?"
        templ = tokz.apply_chat_template([{"role": "user", "content": user}],
                                         tokenize=False, add_generation_prompt=True)
        code_off = templ.index(s["code"])
        pos = spans_to_positions(tokz, templ, code_off, s["identifier_spans"])
        read_positions(label, user, None, pos, drm=True)

    # ---- 3: python L2 dispatcher-state reads ----
    s = snippets["py_l2"]
    user = PREAMBLE + s["code"] + "\n\nWhat is the exact output?"
    templ = tokz.apply_chat_template([{"role": "user", "content": user}],
                                     tokenize=False, add_generation_prompt=True)
    code_off = templ.index(s["code"])
    pos = spans_to_positions(tokz, templ, code_off, s["dispatcher_spans"])
    read_positions(f"L2_dispatcher_{s['snippet_id'].replace('/', '-')}", user, None, pos, drm=False)

    # ---- 4: the model's own greedy CoT on the L1b trap, reads over reasoning ----
    s = snippets["flag_l1b"]
    user = PREAMBLE + s["code"] + "\n\nReason step by step, then give the exact output."
    ids = tokz.apply_chat_template([{"role": "user", "content": user}], tokenize=True,
                                   add_generation_prompt=True, return_dict=False)
    with torch.no_grad():
        out = ex.model.generate(torch.tensor([ids], device=ex.model.device),
                                max_new_tokens=320, do_sample=False,
                                pad_token_id=tokz.eos_token_id)
    reply = tokz.decode(out[0][len(ids):], skip_special_tokens=True)
    print(f"\n{'='*80}\n== model's greedy CoT on the L1b trap ==\n{reply[:900]}")
    n_reply_tok = out.shape[1] - len(ids)
    # 8 evenly spaced positions across the generated reply
    reply_positions = [(len(ids) + int(k), f"cot@{int(k)}")
                       for k in np.linspace(5, max(6, n_reply_tok - 2), 8).astype(int)]
    read_positions("CoT_L1b", user, reply, reply_positions, drm=True)

    ex.close()

    out_dir = _PROJ / "data" / "nla" / "probe"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "probe_reads.json").write_text(json.dumps(
        {"model": TARGET_MODEL, "layer": LAYER_INDEX, "seed": SEED,
         "glosses": {"true": GLOSS_TRUE, "decoy": GLOSS_DECOY},
         "cot_reply": reply, "rows": results}, indent=1))
    print(f"\nwrote {out_dir / 'probe_reads.json'}  ({len(results)} reads)")

    # summary: DRM at trap identifier positions
    for label in ("L0_fibfib", "L1b_smoothArea", "CoT_L1b"):
        drms = [r["drm_ar"] for r in results if r["item"] == label and "drm_ar" in r]
        if drms:
            print(f"mean DRM_AR ({label}): {np.mean(drms):+.3f}  (n={len(drms)}; >0 = decoy-leaning)")
    cjk = sum(bool(re.search(r"[　-鿿]", r["read"])) for r in results)
    print(f"CJK reads: {cjk}/{len(results)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
