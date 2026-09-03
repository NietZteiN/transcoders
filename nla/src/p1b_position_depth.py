"""P1b — position x depth: when does correctness become readable from the residual stream?

Pre-registration: log/nla-harness/2026-08-30_p1b-position-depth-prereg.md. Every constant is
quoted from it and none is tuned here.

Every item-level read in this programme has been taken at `last_prompt`. Two independent results
say the signal lives late: `rt_cos` pays at the `answer_line` and nowhere in the reasoning trace
(q = 8e-05), and dense correctness decodability climbs monotonically to the final layer (0.7612).
If correctness only becomes readable late in BOTH senses, the ledger of item-level nulls was
measuring too early rather than measuring absence.

THE SCEPTICAL BRANCH IS THE POINT. Layer 27 is one block from the logits and the answer line is
one token from the value, so a signal confined there is close to reading the output rather than
predicting it. H-PD2 splits EARLY KNOWLEDGE from LATE READOUT ONLY on `reply_q50`, and the rule
was frozen before any number existed.

Env `nla-mi`, one GPU.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics as st
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_NLA_ROOT = _HERE.parent
_PROJ = _NLA_ROOT.parent
sys.path.insert(0, str(_NLA_ROOT / "vendor" / "nla-repo"))
sys.path.insert(0, str(_HERE))

SEED = 20260724
MAX_NEW_GEN = 2048
C_FROZEN, N_FOLDS, N_BOOT, N_PERM = 1.0, 5, 2000, 200
DELTA_POS, DELTA_LEN = 0.10, 0.05
PRIMARY_LAYER = 20
POSITIONS = ["last_prompt", "reply_q25", "reply_q50", "reply_q75", "answer_line", "last_token"]

from p1b_read_probe import grouped_cv_auc  # noqa: E402


def boot_ci_diff(y, a, b, groups, seed=SEED):
    from sklearn.metrics import roc_auc_score
    rng = random.Random(seed)
    uniq = sorted(set(groups))
    by = {g: [i for i, gg in enumerate(groups) if gg == g] for g in uniq}
    out = []
    for _ in range(N_BOOT):
        idx = [i for _ in range(len(uniq)) for i in by[uniq[rng.randrange(len(uniq))]]]
        yy = np.asarray(y)[idx]
        if len(set(yy.tolist())) < 2:
            continue
        out.append(roc_auc_score(yy, np.asarray(a)[idx]) - roc_auc_score(yy, np.asarray(b)[idx]))
    out.sort()
    return [round(out[int(0.025 * len(out))], 4), round(out[int(0.975 * len(out))], 4)] \
        if out else [None, None]


def token_char_offsets(tokz, ids: list[int]) -> list[int]:
    """Cumulative character offset at which each generated token's text begins.

    Built by decoding tokens one at a time rather than re-tokenising the decoded string: the ids
    that were actually generated are the ground truth, and a re-tokenisation can merge differently
    and shift every index.
    """
    offs, n = [], 0
    for t in ids:
        offs.append(n)
        n += len(tokz.decode([t], skip_special_tokens=True))
    return offs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    # The grid was written for L1b. L2 is where the ladder found signal, and a dispatcher's state
    # is a thing that changes DURING simulation — so if state tracking exists at all it lives in
    # the reply positions, which have never been probed on any tier but L1b.
    ap.add_argument("--tier", default="L1b", choices=["L0", "L1", "L1b", "L2", "L3"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--acts-out", default=None)
    args = ap.parse_args()
    tag = "" if args.tier == "L1b" else f"_{args.tier}"
    if args.out is None:
        args.out = str(_PROJ / f"data/nla/p0/p1b/position_depth{tag}.json")
    if args.acts_out is None:
        args.acts_out = str(_PROJ / f"data/nla/p0/p1b/pos_acts{tag}.npz")

    import torch
    from p1b_ladder import load_tier
    from steer_run import ANSWER_RE, SEED as RSEED, TARGET_MODEL, build_user, graded, load_pairs
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(SEED)
    if args.tier == "L1b":
        pairs = load_pairs(args.limit, random.Random(RSEED))
    else:
        # load_tier yields {snippet_id, code, call, truth}; rename to the keys used below so the
        # rest of the grid is untouched.
        pairs = [{"snippet_id": r["snippet_id"], "code_l1b": r["code"],
                  "call_l1b": r["call"], "truth": r["truth"]}
                 for r in load_tier(args.tier, random.Random(RSEED))]
        if args.limit:
            pairs = pairs[:args.limit]
    tokz = AutoTokenizer.from_pretrained(TARGET_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        TARGET_MODEL, torch_dtype=torch.bfloat16, device_map=args.device).eval()

    # Same latent Gemma-3 bug as multilayer_vectors had: Gemma3Config carries neither field at
    # the top level. Harmless on Qwen/Llama, fatal on a Gemma host.
    from steer_multilayer import model_dims
    d_model, n_layers = model_dims(model)
    P = len(POSITIONS)
    acts = np.zeros((len(pairs), P, n_layers, d_model), dtype=np.float32)
    valid = np.zeros((len(pairs), P), dtype=bool)
    y, groups, lengths, replies = [], [], [], []

    for i, p in enumerate(pairs):
        user = build_user(p["code_l1b"], p["call_l1b"])
        ids = tokz.apply_chat_template([{"role": "user", "content": user}], tokenize=True,
                                       add_generation_prompt=True, return_dict=False)
        with torch.no_grad():
            o = model.generate(torch.tensor([ids], device=model.device),
                               max_new_tokens=MAX_NEW_GEN, do_sample=False,
                               pad_token_id=tokz.eos_token_id)
        full = o[0].tolist()
        gen = full[len(ids):]
        text = tokz.decode(gen, skip_special_tokens=True)
        got, ok = graded(text, p["truth"])
        y.append(int(ok))
        groups.append(p["snippet_id"])
        lengths.append(len(gen))
        replies.append({"snippet_id": p["snippet_id"], "correct": bool(ok), "answer": got,
                        "n_gen": len(gen), "chars": len(text)})

        # One forward pass over prompt+reply; hidden_states[K+1] is layer K (P0.1's gate).
        with torch.no_grad():
            out = model(torch.tensor([full], device=model.device), output_hidden_states=True)
        hs = out.hidden_states
        plen, ng = len(ids), len(gen)

        idx = {"last_prompt": plen - 1,
               "reply_q25": plen + int(0.25 * ng) if ng else None,
               "reply_q50": plen + int(0.50 * ng) if ng else None,
               "reply_q75": plen + int(0.75 * ng) if ng else None,
               "last_token": len(full) - 1,
               "answer_line": None}
        m = ANSWER_RE.search(text)
        if m and ng:
            offs = token_char_offsets(tokz, gen)
            # the token whose text begins at or before the answer value's first character
            k = max([j for j, o_ in enumerate(offs) if o_ <= m.start(1)], default=0)
            idx["answer_line"] = plen + max(k - 1, 0)

        for pi, tag in enumerate(POSITIONS):
            t = idx.get(tag)
            if t is None or not (0 <= t < len(full)):
                continue
            for L in range(n_layers):
                acts[i, pi, L] = hs[L + 1][0, t].float().cpu().numpy()
            valid[i, pi] = True

        if (i + 1) % 10 == 0:
            print(f"[pd] {i+1}/{len(pairs)} · acc so far {st.mean(y):.3f}", flush=True)

    y = np.array(y); groups = np.array(groups); lengths = np.array(lengths, dtype=float)
    Path(args.acts_out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.acts_out, acts=acts, valid=valid, y=y, groups=groups,
                        lengths=lengths, positions=np.array(POSITIONS))
    print(f"[pd] acts {acts.shape} · accuracy {y.mean():.4f} · "
          f"answer_line valid {valid[:, POSITIONS.index('answer_line')].sum()}/{len(y)}",
          flush=True)

    rep = {"experiment": "p1b_position_depth", "tier": args.tier,
           "prereg": "log/nla-harness/2026-08-30_p1b-position-depth-prereg.md",
           "model": TARGET_MODEL, "seed": SEED, "max_new_gen": MAX_NEW_GEN, "C": C_FROZEN,
           "n_items": int(len(y)), "n_correct": int(y.sum()),
           "positions": POSITIONS, "n_layers": int(n_layers),
           "grid": {}, "replies": replies}

    oof: dict[tuple[str, int], np.ndarray] = {}
    for pi, tag in enumerate(POSITIONS):
        v = valid[:, pi]
        rep["grid"][tag] = {"n": int(v.sum()), "layers": []}
        if v.sum() < 30 or len(set(y[v].tolist())) < 2:
            rep["grid"][tag]["note"] = "too few valid items"
            continue
        # length-only baseline at this position, same folds
        len_auc, _ = grouped_cv_auc(lengths[v].reshape(-1, 1), y[v], groups[v])
        rep["grid"][tag]["length_only_auc"] = round(len_auc, 4)
        for L in range(n_layers):
            a, o = grouped_cv_auc(acts[v, pi, L], y[v], groups[v])
            oof[(tag, L)] = o
            rep["grid"][tag]["layers"].append({"layer": L, "auc": round(a, 4)})
        best = max(rep["grid"][tag]["layers"], key=lambda r: r["auc"])
        rep["grid"][tag]["argmax_layer"] = best["layer"]
        rep["grid"][tag]["argmax_auc"] = best["auc"]
        print(f"[pd] {tag:<12} n={int(v.sum()):>3} lengthAUC {len_auc:.4f} · "
              f"L{PRIMARY_LAYER} {rep['grid'][tag]['layers'][PRIMARY_LAYER]['auc']:.4f} · "
              f"best L{best['layer']} {best['auc']:.4f}", flush=True)

    def cell(tag):
        g = rep["grid"].get(tag, {})
        return g["layers"][PRIMARY_LAYER]["auc"] if g.get("layers") else None

    def contrast(tag):
        """tag vs last_prompt at the primary layer, paired on items valid for BOTH."""
        a, b = POSITIONS.index(tag), POSITIONS.index("last_prompt")
        both = valid[:, a] & valid[:, b]
        if both.sum() < 30:
            return {"verdict": "NEEDS_DATA", "n": int(both.sum())}
        ya, ga = y[both], groups[both]
        auc_a, oa = grouped_cv_auc(acts[both, a, PRIMARY_LAYER], ya, ga)
        auc_b, ob = grouped_cv_auc(acts[both, b, PRIMARY_LAYER], ya, ga)
        lb, _ = grouped_cv_auc(lengths[both].reshape(-1, 1), ya, ga)
        d = auc_a - auc_b
        ci = boot_ci_diff(ya, oa, ob, ga)
        beats_len = auc_a - lb
        ok = (d >= DELTA_POS and ci[0] is not None and ci[0] > 0 and beats_len >= DELTA_LEN)
        return {"n": int(both.sum()), "auc_here": round(auc_a, 4),
                "auc_last_prompt": round(auc_b, 4), "delta": round(d, 4), "ci95": ci,
                "length_only_auc": round(lb, 4), "beats_length_by": round(beats_len, 4),
                "thresholds": {"delta": DELTA_POS, "beats_length": DELTA_LEN},
                "passes": bool(ok)}

    prim = contrast("answer_line")
    rep["primary_H_PD1"] = {**prim,
                            "verdict": ("POSITION MATTERS" if prim.get("passes")
                                        else "POSITION DOES NOT MATTER")}
    if prim.get("passes"):
        mid = contrast("reply_q50")
        rep["secondary_H_PD2"] = {
            "reply_q50": mid,
            "verdict": ("EARLY KNOWLEDGE — correctness is represented before the answer is "
                        "committed" if mid.get("passes") else
                        "LATE READOUT ONLY — the probe reads a decided answer; no predictive "
                        "claim is licensed"),
        }
    else:
        rep["secondary_H_PD2"] = {"verdict": "NOT EVALUATED",
                                  "reason": "defined only when H-PD1 returns POSITION MATTERS"}

    # permutation null: a DISTRIBUTION, at the primary cell only (200 grouped-CV refits)
    v = valid[:, POSITIONS.index("answer_line")]
    Xp = acts[v, POSITIONS.index("answer_line"), PRIMARY_LAYER]
    rng = np.random.default_rng(SEED)
    null = [grouped_cv_auc(Xp, rng.permutation(y[v]), groups[v])[0] for _ in range(N_PERM)]
    obs = rep["grid"]["answer_line"]["layers"][PRIMARY_LAYER]["auc"]
    rep["permutation_control"] = {
        "cell": f"answer_line/L{PRIMARY_LAYER}", "n_perm": N_PERM,
        "observed": obs, "null_mean": round(st.mean(null), 4),
        "null_sd": round(st.pstdev(null), 4), "null_max": round(max(null), 4),
        "p_empirical": round((sum(1 for x in null if x >= obs) + 1) / (N_PERM + 1), 5),
        "leaks": bool(st.mean(null) > 0.60)}

    rep["finished_utc"] = datetime.now(timezone.utc).isoformat()
    Path(args.out).write_text(json.dumps(rep, indent=2))
    print(f"\n[pd] PRIMARY: {rep['primary_H_PD1']['verdict']}")
    print(f"[pd] H-PD2:   {rep['secondary_H_PD2']['verdict']}")
    print(f"[pd] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
