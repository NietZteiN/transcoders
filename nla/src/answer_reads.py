"""Add ANSWER-LINE reads to the capture gallery and rebuild EXAMPLES.md properly.

For each captured task (data/nla/examples/captures.json): find the final answer line in
the model's reply (`Output: ...` / `Lines: [...]`), verbalize the L20 activations at up to
3 token positions ON that line (the value tokens — the moment the model commits to its
answer), and store them as `answer_reads`. Then regenerate EXAMPLES.md with FULL reads
(no truncation, no table-breaking) and the answer reads shown right next to the model
answer — "what the model said" vs "what its layer-20 state says it was thinking".

Run (nla-mi env; AV server on :30000):
  python -m src.answer_reads --device cuda
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
CAPTURES = _PROJ / "data" / "nla" / "examples" / "captures.json"
GALLERY = _PROJ / "data" / "nla" / "examples" / "EXAMPLES.md"

# Must match capture_examples.py so token positions line up with the stored replies.
PREAMBLE = (
    "You are an expert software engineer taking part in a code comprehension study. "
    "You will be given one small task about a piece of code. Read the code carefully, "
    "reason about what it does step by step, and then answer in exactly the format "
    "requested. Be precise about values and types; trace the computation rather than "
    "guessing from names. Here is the task:\n\n"
)


def build_user(c: dict) -> str:
    if c["kind"] == "output_prediction":
        return (PREAMBLE + c["code"] +
                f"\n\nWhat is the exact output of `{c['call']}`? Reason step by step, "
                f"then end with one line exactly of the form `Output: <value>`.")
    return (PREAMBLE + "Line-numbered program:\n\n" + c["code"] +
            f"\n\nWhich line numbers actually execute AND affect the value printed on the "
            f"final line (the dynamic backward slice for `{c['target']}`)? Reason step by "
            f"step, then end with one line exactly of the form `Lines: [n1, n2, ...]`.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--sglang-url", default="http://localhost:30000")
    args = ap.parse_args()

    from extract import ActivationExtractor
    from nla_inference import NLAClient, NLACritic

    data = json.loads(CAPTURES.read_text())
    ex = ActivationExtractor(TARGET_MODEL, LAYER_INDEX, device=args.device)
    av = NLAClient(_NLA_ROOT / "data" / "checkpoints" / "av", sglang_url=args.sglang_url)
    ar = NLACritic(_NLA_ROOT / "data" / "checkpoints" / "ar", device=args.device)
    tokz = ex.tokenizer

    for c in data["captures"]:
        reply = c["model_reply"]
        m = (re.search(r"Output:.*$", reply, re.M) if c["kind"] == "output_prediction"
             else re.search(r"Lines:.*$", reply))
        if not m:
            c["answer_reads"] = []
            print(f"{c['task_id']}: no answer line in reply (truncated run) — skipped")
            continue

        user = build_user(c)
        templ = tokz.apply_chat_template([{"role": "user", "content": user}],
                                         tokenize=False, add_generation_prompt=True)
        full = templ + reply
        a = len(templ) + m.start()
        b = len(templ) + m.end()
        enc = tokz(full, return_offsets_mapping=True, add_special_tokens=False)
        line_positions = [i for i, (s, e) in enumerate(enc["offset_mapping"])
                          if e > s and s < b and e > a]
        # up to 3 reads spread across the answer line (start, middle, last value token)
        picks = sorted(set(np.linspace(0, len(line_positions) - 1, 3).astype(int)))
        picks = [line_positions[i] for i in picks]

        res = ex.extract_chat(user, reply)
        reads = []
        for pos in picks:
            if pos >= len(res.positions):
                continue
            v = res.activations[pos]
            text = av.generate(v, temperature=0.0, max_new_tokens=180)
            _, cos_rt = ar.score(text, v)
            s, e = enc["offset_mapping"][pos]
            reads.append({"position": int(pos), "token_text": full[s:e],
                          "roundtrip_cos": round(float(cos_rt), 3), "read": text})
        c["answer_reads"] = reads
        print(f"{c['task_id']}: {len(reads)} answer-line reads (line: {m.group(0)[:60]!r})")

    ex.close()
    CAPTURES.write_text(json.dumps(data, indent=1))

    # ---- rebuild the gallery: full reads, no tables ------------------------------
    md = ["# NLA out-of-the-box captures — simple code tasks", "",
          f"Model: `{data['model']}` · layer {data['layer']} · greedy · seed {data['seed']}.",
          "Each *read* is the NLA verbalizer's description of the layer-20 residual vector at that",
          "single token (temp 0). `rt` = AR round-trip cosine (how much of the vector the text recovers).", ""]
    for c in data["captures"]:
        head = "✅ correct" if c["correct"] else "❌ wrong"
        md += [f"## {c['task_id']}  — {c['kind'].replace('_', ' ')}, {head}", "",
               "```" + ("python" if c["language"] == "python" else "javascript"),
               c["code"], "```", "",
               f"**Question:** `{c.get('prompt_question') or 'dynamic slice for ' + str(c.get('target'))}`",
               f"**Ground truth:** `{c['truth']}`",
               f"**Model answer:** `{c['model_answer']}`", ""]
        if c.get("answer_reads"):
            md += ["### 🧠 NLA reads AT the answer line (the moment the model commits)", ""]
            for r in c["answer_reads"]:
                md += [f"**@ token `{r['token_text']}` (pos {r['position']}, rt {r['roundtrip_cos']}):**",
                       "", "> " + r["read"].replace("\n", "\n> "), ""]
        elif c.get("answer_reads") == []:
            md += ["*(no answer line — reply truncated before committing; no answer reads)*", ""]
        md += ["### NLA reads at code tokens and across the CoT", ""]
        for r in c["reads"]:
            md += [f"**@ `{r['where']}` (pos {r['position']}, rt {r['roundtrip_cos']}):**",
                   "", "> " + r["read"].replace("\n", "\n> "), ""]
        md += ["<details><summary>full model reply</summary>", "", "```",
               c["model_reply"].strip(), "```", "", "</details>", "", "---", ""]
    GALLERY.write_text("\n".join(md))
    n_ans = sum(len(c.get("answer_reads") or []) for c in data["captures"])
    print(f"\nrebuilt {GALLERY} · {n_ans} answer-line reads added")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
