"""Capture gallery: out-of-the-box model answers + NLA reads on simple code tasks.

Two task kinds:
  * output prediction — short L0 snippets from Dataset A (ground truth = expected_output)
  * slice prediction  — 3 hand-built numbered programs; predict the DYNAMIC backward
    slice (executed lines that affect the printed value); ground truth derived by hand

For each task: greedy model answer (graded), then L20 reads at (a) key code-token
positions, (b) evenly spaced CoT positions, verbalized by the AV and round-trip-scored
by the AR. Output: data/nla/examples/captures.json + a human-readable EXAMPLES.md.

Run (nla-mi env; AV server on :30000):
  python -m src.capture_examples --device cuda
"""
from __future__ import annotations

import argparse
import ast
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
N_OUTPUT_TASKS = 4
COT_READS = 5
CODE_READS = 5

PREAMBLE = (
    "You are an expert software engineer taking part in a code comprehension study. "
    "You will be given one small task about a piece of code. Read the code carefully, "
    "reason about what it does step by step, and then answer in exactly the format "
    "requested. Be precise about values and types; trace the computation rather than "
    "guessing from names. Here is the task:\n\n"
)

SLICE_TASKS = [
    {
        "task_id": "slice_1_straightline",
        "code": ("1  a = 3\n2  b = 4\n3  c = a * 2\n4  d = b + 1\n"
                 "5  e = c + a\n6  print(e)"),
        "target": "e",
        "truth": [1, 3, 5],
        "key_names": ["e", "c", "a"],
    },
    {
        "task_id": "slice_2_loop",
        "code": ("1  n = 5\n2  total = 0\n3  count = 0\n4  for i in range(n):\n"
                 "5      total += i\n6  count = count + 2\n7  print(total)"),
        "target": "total",
        "truth": [1, 2, 4, 5],
        "key_names": ["total", "n", "count"],
    },
    {
        "task_id": "slice_3_branch",
        "code": ("1  x = 2\n2  y = 10\n3  z = 0\n4  if x > 1:\n5      z = y + x\n"
                 "6  else:\n7      z = y - x\n8  w = z * 2\n9  print(w)"),
        "target": "w",
        "truth": [1, 2, 4, 5, 8],
        "key_names": ["w", "z", "x"],
    },
]


def word_spans(text: str, name: str) -> list[tuple[int, int]]:
    return [(m.start(), m.end()) for m in
            re.finditer(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])", text)]


def pick_output_tasks() -> list[dict]:
    """Shortest Dataset-A L0 snippets with parseable arg-list inputs."""
    rows = []
    with open(_PROJ / "data" / "stimuli" / "dataset_a" / "dataset_a.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if r["tier"] != "L0":
                continue
            try:
                args = ast.literal_eval(r["meta"]["input"])
                assert isinstance(args, list)
            except Exception:
                continue
            fn = r["meta"].get("fn_name")
            if not fn:
                continue
            call = f"{fn}({', '.join(json.dumps(a) for a in args)})"
            rows.append({"task_id": r["snippet_id"], "language": r["language"],
                         "code": r["code"], "call": call,
                         "truth": r["expected_output"],
                         "key_names": sorted({si["name"] for si in r["meta"]["span_info"]},
                                             key=len, reverse=True)[:3]})
    rows.sort(key=lambda r: len(r["code"]))
    # 2 shortest per language for variety
    by_lang: dict[str, list] = {}
    for r in rows:
        by_lang.setdefault(r["language"], []).append(r)
    picked = []
    for lang in sorted(by_lang):
        picked.extend(by_lang[lang][:2])
    return picked[:N_OUTPUT_TASKS]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--sglang-url", default="http://localhost:30000")
    args = ap.parse_args()

    import torch
    torch.manual_seed(SEED)
    from extract import ActivationExtractor
    from nla_inference import NLAClient, NLACritic

    ex = ActivationExtractor(TARGET_MODEL, LAYER_INDEX, device=args.device)
    av = NLAClient(_NLA_ROOT / "data" / "checkpoints" / "av", sglang_url=args.sglang_url)
    ar = NLACritic(_NLA_ROOT / "data" / "checkpoints" / "ar", device=args.device)
    tokz = ex.tokenizer

    def generate_reply(user: str, max_new: int = 900) -> str:
        ids = tokz.apply_chat_template([{"role": "user", "content": user}], tokenize=True,
                                       add_generation_prompt=True, return_dict=False)
        with torch.no_grad():
            out = ex.model.generate(torch.tensor([ids], device=ex.model.device),
                                    max_new_tokens=max_new, do_sample=False,
                                    pad_token_id=tokz.eos_token_id)
        return tokz.decode(out[0][len(ids):], skip_special_tokens=True)

    def read_at(user: str, reply: str, code_in_user: str, key_names: list[str]) -> list[dict]:
        """CODE_READS positions at key-name tokens + COT_READS spread over the reply."""
        res = ex.extract_chat(user, reply)
        templ = tokz.apply_chat_template([{"role": "user", "content": user}],
                                         tokenize=False, add_generation_prompt=True)
        full = templ + reply
        enc = tokz(full, return_offsets_mapping=True, add_special_tokens=False)
        offsets = enc["offset_mapping"]
        code_off = full.index(code_in_user)

        code_positions: list[tuple[int, str]] = []
        for name in key_names:
            for a, b in word_spans(code_in_user, name):
                a, b = a + code_off, b + code_off
                for i, (s, e) in enumerate(offsets):
                    if e > s and s < b and e > a and len(full[s:e].strip()) > 1:
                        code_positions.append((i, full[s:e]))
                        break                      # first token of each occurrence only
        code_positions = code_positions[:CODE_READS]

        reply_start = len(tokz(templ, add_special_tokens=False)["input_ids"])
        n_total = len(res.positions)
        cot_positions = [(int(p), f"cot@{int(p) - reply_start}") for p in
                         np.linspace(reply_start + 4, n_total - 2, COT_READS).astype(int)]

        reads = []
        for pos, label in code_positions + cot_positions:
            if pos >= n_total:
                continue
            v = res.activations[pos]
            text = av.generate(v, temperature=0.0, max_new_tokens=180)
            _, cos_rt = ar.score(text, v)
            reads.append({"position": int(pos), "where": label,
                          "roundtrip_cos": round(float(cos_rt), 3), "read": text})
        return reads

    captures = []

    # ---- output-prediction tasks ------------------------------------------------
    for t in pick_output_tasks():
        user = (PREAMBLE + t["code"] +
                f"\n\nWhat is the exact output of `{t['call']}`? Reason step by step, "
                f"then end with one line exactly of the form `Output: <value>`.")
        reply = generate_reply(user)
        m = re.search(r"Output:\s*(.+?)\s*$", reply, re.M)
        answer = m.group(1).strip() if m else None
        norm = lambda s: re.sub(r"[\s'\"`]", "", str(s)).lower()   # backticks too — model wraps values in `…`
        correct = answer is not None and norm(answer) == norm(t["truth"])
        print(f"[output] {t['task_id']}: answer={answer!r} truth={t['truth']!r} correct={correct}")
        captures.append({"kind": "output_prediction", **t, "prompt_question": t["call"],
                         "model_answer": answer, "correct": bool(correct),
                         "model_reply": reply,
                         "reads": read_at(user, reply, t["code"], t["key_names"])})

    # ---- slice-prediction tasks -------------------------------------------------
    for t in SLICE_TASKS:
        user = (PREAMBLE + "Line-numbered program:\n\n" + t["code"] +
                f"\n\nWhich line numbers actually execute AND affect the value printed on the "
                f"final line (the dynamic backward slice for `{t['target']}`)? Reason step by "
                f"step, then end with one line exactly of the form `Lines: [n1, n2, ...]`.")
        reply = generate_reply(user)
        m = re.search(r"Lines:\s*\[([0-9,\s]*)\]", reply)
        answer = sorted(int(x) for x in m.group(1).split(",") if x.strip()) if m else None
        correct = answer == t["truth"]
        print(f"[slice ] {t['task_id']}: answer={answer} truth={t['truth']} correct={correct}")
        captures.append({"kind": "slice_prediction", **t, "language": "python",
                         "model_answer": str(answer), "correct": bool(correct),
                         "model_reply": reply,
                         "reads": read_at(user, reply, t["code"], t["key_names"])})

    ex.close()

    out_dir = _PROJ / "data" / "nla" / "examples"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "captures.json").write_text(json.dumps(
        {"model": TARGET_MODEL, "layer": LAYER_INDEX, "seed": SEED, "captures": captures}, indent=1))

    # ---- readable gallery -------------------------------------------------------
    md = ["# NLA out-of-the-box captures — simple code tasks",
          "", f"Model: `{TARGET_MODEL}` · layer {LAYER_INDEX} · greedy · seed {SEED}. ",
          "Reads = AV verbalization of the L20 residual at that token (temp 0); `rt_cos` = AR round-trip cosine.", ""]
    for c in captures:
        md += [f"## {c['task_id']}  ({c['kind']}, {'✅ correct' if c['correct'] else '❌ wrong'})", "",
               "```" + ("python" if c["language"] == "python" else "javascript"),
               c["code"], "```", "",
               f"**Question:** `{c.get('prompt_question', 'dynamic slice for ' + c.get('target', '?'))}`  ",
               f"**Model answer:** `{c['model_answer']}` · **Truth:** `{c['truth']}`", "",
               "<details><summary>model reply</summary>", "", "```", c["model_reply"].strip()[:1200], "```", "</details>", "",
               "| where | rt_cos | NLA read |", "|---|---|---|"]
        for r in c["reads"]:
            read_one_line = " ".join(r["read"].split())[:260]
            md.append(f"| `{r['where']}` @ {r['position']} | {r['roundtrip_cos']} | {read_one_line} |")
        md.append("")
    (out_dir / "EXAMPLES.md").write_text("\n".join(md))

    n_ok = sum(c["correct"] for c in captures)
    n_reads = sum(len(c["reads"]) for c in captures)
    cjk = sum(bool(re.search(r"[　-鿿]", r["read"])) for c in captures for r in c["reads"])
    print(f"\n{len(captures)} tasks ({n_ok} correct) · {n_reads} reads · CJK {cjk}")
    print(f"wrote {out_dir / 'captures.json'} and {out_dir / 'EXAMPLES.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
