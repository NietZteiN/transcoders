#!/usr/bin/env python
"""L3 -- what carries the per-item obfuscation cost G?

Rule frozen in log/nla-harness/2026-09-03_g-variance-prereg.md BEFORE this ran.
Six predictors fixed there; no predictor may be added after seeing output.

G = [logp(y_clean|x_l0) - logp(y_clean|x_l1b)] / |y_clean|, the R2 clean-half unit.
Because G divides by |y_clean|, longer replies average over more tokens and shrink G toward its
mean, so a raw correlation with reply length is partly MECHANICAL. Every predictor is therefore a
partial Spearman controlling rank(|y_clean|); reply length is reported as a diagnostic only.
"""
from __future__ import annotations
import argparse, difflib, json, pathlib, random, sys
import numpy as np

SEED, N_BOOT, N_PERM, RHO_MIN = 20260724, 10000, 10000, 0.30
_PROJ = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJ / "nla" / "src"))
from length_confound import rank, pearson, spearman, partial_spearman, boot, bh  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default=str(_PROJ / "data/nla/p0/trace_llr/gemma12b/llr_rows.jsonl"))
    ap.add_argument("--traces", default=str(_PROJ / "data/nla/p0/trace_llr/gemma12b/traces.jsonl"))
    ap.add_argument("--out", default=str(_PROJ / "data/nla/p0/trace_llr/gemma12b/g_variance.json"))
    a = ap.parse_args()
    from steer_run import SEED as S_SEED, load_pairs
    rng = np.random.default_rng(SEED)

    clean, ntok = {}, {}
    for line in open(a.rows):
        r = json.loads(line); sid, *tag, which = r["key"].split("|")
        if which == "clean":
            clean[(sid, "|".join(tag))] = r["logp_sum"] / r["n_tok"]
            ntok[sid] = r["n_tok"]
    traces = {json.loads(l)["snippet_id"]: json.loads(l) for l in open(a.traces)}
    pairs = {p["snippet_id"]: p for p in load_pairs(None, random.Random(S_SEED))}

    G, L, cols, sids = [], [], {k: [] for k in
                               ("snippet_size", "id_surface", "decoy_overlap",
                                "decoy_verbosity", "language_js", "l0_correct")}, []
    for sid, t in traces.items():
        p = pairs.get(sid)
        if p is None or (sid, "l0prompt") not in clean:
            continue
        ids0, ids1 = t["l0_prompt_ids"], t["l1b_prompt_ids"]
        # identifier surface = share of the L1b prompt occupied by tokens that differ from L0
        sm = difflib.SequenceMatcher(a=ids0, b=ids1, autojunk=False)
        idt = sum(j1 - j0 for tag, i0, i1, j0, j1 in sm.get_opcodes() if tag in ("replace", "insert"))
        tt, td = set(p.get("terms_true") or []), set(p.get("terms_decoy") or [])
        dec = list((p.get("rename_map") or {}).values())
        sids.append(sid)
        G.append(clean[(sid, "l0prompt")] - clean[(sid, "noop#1")])
        L.append(ntok[sid])
        cols["snippet_size"].append(len(ids0))
        cols["id_surface"].append(idt / len(ids1))
        cols["decoy_overlap"].append(len(tt & td) / len(tt) if tt else 0.0)
        cols["decoy_verbosity"].append(float(np.mean([len(d) for d in dec])) if dec else 0.0)
        cols["language_js"].append(1.0 if (p.get("language") or "").lower().startswith("java") else 0.0)
        cols["l0_correct"].append(1.0 if t.get("l0_correct") else 0.0)
    G, L = np.asarray(G, float), np.asarray(L, float)
    X = {k: np.asarray(v, float) for k, v in cols.items()}

    per, ps = {}, []
    for k, x in X.items():
        r = partial_spearman(G, x, L)
        lo, hi = boot(lambda g, xx, ll: partial_spearman(g, xx, ll), [G, x, L], rng)
        obs = abs(r)
        cnt = sum(abs(partial_spearman(rng.permutation(G), x, L)) >= obs for _ in range(N_PERM // 10))
        pv = (cnt + 1) / (N_PERM // 10 + 1)
        per[k] = {"rho_partial": r, "ci95": [lo, hi], "p_perm": pv,
                  "rho_raw": spearman(G, x)}
        ps.append(pv)
    for k, q in zip(X, bh(ps)):
        per[k]["q_bh"] = float(q)

    # joint rank regression, adjusted R^2 -- reported with the overfitting caveat, not as a result
    A = np.c_[np.ones(len(G)), np.column_stack([rank(X[k]) for k in X]), rank(L)]
    beta, *_ = np.linalg.lstsq(A, rank(G), rcond=None)
    resid = rank(G) - A @ beta
    ss = ((rank(G) - rank(G).mean()) ** 2).sum()
    r2 = 1 - (resid ** 2).sum() / ss
    n, k_ = len(G), A.shape[1] - 1
    r2adj = 1 - (1 - r2) * (n - 1) / (n - k_ - 1)

    sig = [k for k, v in per.items() if v["q_bh"] < 0.05 and abs(v["rho_partial"]) >= RHO_MIN]
    weak = [k for k, v in per.items() if v["q_bh"] < 0.05]
    verdict = "L3-SIGNAL" if sig else ("L3-WEAK" if weak else "L3-NULL")

    res = {"experiment": "L3_g_variance", "seed": SEED, "n": len(G), "rho_min": RHO_MIN,
           "control": "clean_reply_tokens", "per_predictor": per,
           "diagnostic_reply_length": {"rho_raw_G_vs_len": spearman(G, L),
                                       "note": "partly mechanical: |y_clean| is G's denominator"},
           "joint_rank_regression": {"r2": float(r2), "r2_adj": float(r2adj), "k": int(k_),
                                     "caveat": "7 terms on 60 items; adjusted R2 only"},
           "verdict": verdict, "significant": sig}
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=2)

    print(f"n = {len(G)} · mean G {G.mean():+.4f} · control = clean reply tokens "
          f"(diagnostic raw rho(G, len) = {spearman(G, L):+.3f}, partly mechanical)")
    print(f"{'predictor':<18}{'rho|len':>9}{'ci95':>20}{'p':>8}{'q':>8}{'rho_raw':>9}")
    for k, v in sorted(per.items(), key=lambda kv: -abs(kv[1]["rho_partial"])):
        print(f"{k:<18}{v['rho_partial']:>+9.3f}"
              f"{f'[{v[chr(99)+chr(105)+chr(57)+chr(53)][0]:+.3f},{v[chr(99)+chr(105)+chr(57)+chr(53)][1]:+.3f}]':>20}"
              f"{v['p_perm']:>8.4f}{v['q_bh']:>8.4f}{v['rho_raw']:>+9.3f}")
    print(f"\njoint rank regression: R2 {r2:.3f} · adjusted R2 {r2adj:.3f} (7 terms, 60 items)")
    print(f"VERDICT {verdict} · significant {sig or 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
