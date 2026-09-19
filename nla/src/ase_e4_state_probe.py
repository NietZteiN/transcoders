"""E4 — dispatcher state-binding probe (CLAUDE.md §3; charter's cheap, panel-agnostic experiment).

Pre-registered in log/nla-harness/2026-09-19_e4-state-probe-prereg.md.

THE QUESTION. Control-flow flattening replaces straight-line code with a `switch (__state)` dispatcher
whose case labels are PERMUTED, so a case's printed position says nothing about when it runs. Recovering
the execution order requires following the chain of `__state = N;` assignments. E4 asks whether that
recovered order is present in the residual stream: **is a case's EXECUTION RANK linearly decodable at the
token where its label appears?** If yes, the model binds each case to its place in the simulated
sequence; if no, it is reading the text without building the dispatcher's order.

This needs no SAE and no pretrained dictionary, which is why it is the one committed experiment that was
runnable while the Instrument-3 env was broken.

THREE TARGETS, AND TWO OF THEM ARE CONTROLS. A probe that "works" proves nothing on its own:
  * `successor`   -- the next state value, which is written LITERALLY in the case body (`__state = 7;`).
                     A positive control: if this is not decodable the pipeline is broken, not the model.
  * `printed_rank`-- the case's ordinal position in the text. A positional control; transformers encode
                     position, so this should be easy and is NOT evidence of state binding.
  * `exec_rank`   -- the case's position in EXECUTION order. The question. It is dissociated from
                     `printed_rank` by construction (identity permutations were rejected when the corpus
                     was generated), so a probe reading position alone scores chance on it.

Plus a **shuffled-label control**: execution ranks permuted within each snippet, which must land at
chance. Without it, a probe that has merely memorised per-snippet idiosyncrasies looks like a finding.

Splits are GROUPED BY SNIPPET, so no case from a program appears in both train and test.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

TAG = "[E4]"
SEED = 20260724


def dispatcher_truth(src: str) -> dict | None:
    """Recover printed order, entry state and the successor chain from a generated variant."""
    m_entry = re.search(r"int __state = (\d+);", src)
    if not m_entry:
        return None
    entry = int(m_entry.group(1))
    printed = [int(m.group(1)) for m in re.finditer(r"case (\d+):", src)]
    blocks = re.split(r"case (\d+):", src)
    succ: dict[int, int | None] = {}
    for i in range(1, len(blocks), 2):
        lab, body = int(blocks[i]), blocks[i + 1]
        m = re.search(r"__state = (\d+);", body)
        succ[lab] = int(m.group(1)) if m else None
    order, s = [], entry
    while s is not None and s not in order:
        order.append(s)
        s = succ.get(s)
    if len(order) != len(printed):        # a chain that does not cover every case is not usable truth
        return None
    return {"printed": printed, "entry": entry, "succ": succ,
            "exec_rank": {lab: order.index(lab) for lab in printed},
            "printed_rank": {lab: i for i, lab in enumerate(printed)}}


@torch.no_grad()
def capture(lm, prompt: str, layers: list[int], case_labels: list[int]) -> dict[int, dict[int, np.ndarray]]:
    """Residual stream at each layer, at the token where each `case <k>:` label sits."""
    tok = lm.tokenizer
    enc = tok(prompt, return_tensors="pt", return_offsets_mapping=True)
    offsets = enc.pop("offset_mapping")[0].tolist()
    enc = {k: v.to(lm.model.device) for k, v in enc.items()}
    out = lm.model.model(**enc, output_hidden_states=True, use_cache=False, return_dict=True)
    pos: dict[int, int] = {}
    for lab in case_labels:
        m = re.search(rf"case {lab}:", prompt)
        if not m:
            continue
        # the token containing the label digits (search start of `case N:` + 5 = the digit)
        ch = m.start() + 5
        for ti, (a, b) in enumerate(offsets):
            if a <= ch < b:
                pos[lab] = ti
                break
    return {L: {lab: out.hidden_states[L + 1][0, p].float().cpu().numpy() for lab, p in pos.items()}
            for L in layers}


def probe(X: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int) -> dict:
    """Grouped 5-fold multinomial logistic regression; accuracy and Spearman on held-out cases."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from scipy.stats import spearmanr
    accs, rhos, chance = [], [], []
    gkf = GroupKFold(n_splits=5)
    for tr, te in gkf.split(X, y, groups):
        if len(np.unique(y[tr])) < 2:
            continue
        clf = LogisticRegression(max_iter=2000, C=1.0, random_state=seed)
        Xs = (X - X.mean(0)) / (X.std(0) + 1e-6)
        clf.fit(Xs[tr], y[tr])
        p = clf.predict(Xs[te])
        accs.append(float((p == y[te]).mean()))
        r = spearmanr(p, y[te]).statistic
        rhos.append(float(r) if r == r else 0.0)
        # chance = predicting the most frequent training class
        vals, cnt = np.unique(y[tr], return_counts=True)
        chance.append(float((y[te] == vals[cnt.argmax()]).mean()))
    return {"acc": float(np.mean(accs)), "acc_sd": float(np.std(accs)),
            "spearman": float(np.mean(rhos)), "majority_baseline": float(np.mean(chance)),
            "n": int(len(y)), "folds": len(accs)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifact", default="/scratch/juno/jvl210002/ase2026/LLM-Attention-Fixation_submission")
    ap.add_argument("--flat-dir", default="/scratch/juno/jvl210002/ase2026/flat_humaneval_java")
    ap.add_argument("--packs", default="/scratch/juno/jvl210002/ase2026/full/packs_flat_paired.jsonl")
    ap.add_argument("--model-id", default="codellama/CodeLlama-7b-Instruct-hf")
    ap.add_argument("--layers", default="4,10,16,22,28")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    low = args.model_id.lower()
    if any(v in low for v in ("qwen", "deepseek", "yi-", "glm", "internlm", "baichuan")):
        print(f"{TAG} REFUSED: {args.model_id} is a Chinese model."); return 2

    sys.path.insert(0, args.artifact)
    from ase_steer_run import ANSWER_PREFIX, TEMP, TOP_P, install_chat_template  # noqa: E402
    import counterfactual_eval as ce                                            # noqa: E402
    from models import SteeredCausalLM                                          # noqa: E402

    layers = [int(x) for x in args.layers.split(",")]
    packs = [json.loads(l) for l in open(args.packs) if l.strip()]
    lm = SteeredCausalLM()
    # cache_dir must come from HF_HOME, as every other runner in this project does. Passing None sends
    # their loader to the default ~/.cache path, which under HF_HUB_OFFLINE=1 fails with "couldn't
    # connect to huggingface.co" even though the weights are cached (job 412666).
    cache_dir = os.path.join(os.environ["HF_HOME"], "hub") if os.environ.get("HF_HOME") else None
    lm.config(model_name=args.model_id, max_new_tokens=8, temperature=TEMP, top_p=TOP_P,
              key_scope="prompt", cache_dir=cache_dir)
    lm.build()
    install_chat_template(lm)
    print(f"{TAG} {len(packs)} flattened snippets · layers {layers} · {args.model_id}", flush=True)

    rows = {L: {"X": [], "exec_rank": [], "printed_rank": [], "successor": [], "g": []} for L in layers}
    used = skipped = 0
    for i, rec in enumerate(packs, 1):
        src = Path(args.flat_dir, f"{rec['snippet']}.java").read_text()
        truth = dispatcher_truth(src)
        if truth is None:
            skipped += 1; continue
        prompt = lm._build_prompt(src, instruction=ce.build_counterfactual_instruction(rec["pack"]),
                                  language="java", answer_prefix=ANSWER_PREFIX)
        caps = capture(lm, prompt, layers, truth["printed"])
        if not caps[layers[0]]:
            skipped += 1; continue
        for L in layers:
            for lab, vec in caps[L].items():
                s = truth["succ"].get(lab)
                rows[L]["X"].append(vec)
                rows[L]["exec_rank"].append(truth["exec_rank"][lab])
                rows[L]["printed_rank"].append(truth["printed_rank"][lab])
                rows[L]["successor"].append(-1 if s is None else truth["printed_rank"].get(s, -1))
                rows[L]["g"].append(rec["snippet"])
        used += 1
        if i % 25 == 0:
            print(f"{TAG} {i}/{len(packs)}", flush=True)

    rng = np.random.default_rng(SEED)
    report = {"model": args.model_id, "n_snippets": used, "skipped": skipped, "layers": {}}
    print(f"\n{TAG} captured {used} snippets ({skipped} skipped) · "
          f"{len(rows[layers[0]]['X'])} case positions\n")
    hdr = f"{'layer':>6s} {'target':>14s} {'acc':>7s} {'majority':>9s} {'spearman':>9s}"
    print(hdr); print("-" * len(hdr))
    for L in layers:
        X = np.stack(rows[L]["X"]); g = np.array(rows[L]["g"])
        rep = {}
        for target in ("successor", "printed_rank", "exec_rank"):
            y = np.array(rows[L][target])
            rep[target] = probe(X, y, g, SEED)
            print(f"{L:6d} {target:>14s} {rep[target]['acc']:7.3f} "
                  f"{rep[target]['majority_baseline']:9.3f} {rep[target]['spearman']:9.3f}")
        # shuffled control: exec_rank permuted WITHIN snippet -> must be at chance
        y = np.array(rows[L]["exec_rank"]).copy()
        for s in np.unique(g):
            m = g == s
            y[m] = rng.permutation(y[m])
        rep["exec_rank_shuffled"] = probe(X, y, g, SEED)
        print(f"{L:6d} {'exec(shuffled)':>14s} {rep['exec_rank_shuffled']['acc']:7.3f} "
              f"{rep['exec_rank_shuffled']['majority_baseline']:9.3f} "
              f"{rep['exec_rank_shuffled']['spearman']:9.3f}")
        report["layers"][str(L)] = rep
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1))
    print(f"\n{TAG} wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
