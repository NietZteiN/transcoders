#!/usr/bin/env python
"""L — does the token inflation caused by adversarial renaming do independent work?

Rule frozen in log/nla-harness/2026-09-03_length-confound-prereg.md BEFORE this ran (commit ef246c9).

Adversarial renaming inflates L1b prompts by a median +27 tokens (2026-09-03_patch-alignment-
impossible.md). Nothing in Papers 2-3 controls for that. If the extra LENGTH does work beyond the
extra RENAMING, then any length-sensitive L0-vs-L1b measure carries a confound and a length-matched
stimulus tier is justified on its own merits.

Readout is G_i, the R2 unit: [logp(y_clean|x_l0) - logp(y_clean|x_l1b)] / |y_clean|. Both terms share
the denominator |y_clean|, so G has no mechanical dependence on prompt length -- that is the whole
reason it is the readout here rather than a raw logprob.
"""
from __future__ import annotations
import argparse, json, pathlib, random, sys
import numpy as np

SEED, N_BOOT = 20260724, 10000
RHO_MIN = 0.25            # |partial rho| below this is "no independent work" per the frozen rule
_PROJ = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJ / "nla" / "src"))


def rank(x):
    """Average ranks, so ties (common in n_r) don't bias the partial regression."""
    x = np.asarray(x, float); order = x.argsort(); r = np.empty(len(x), float)
    r[order] = np.arange(len(x), dtype=float)
    _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    means = np.zeros(len(cnt))
    np.add.at(means, inv, r); means /= cnt
    return means[inv]


def pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a - a.mean(), b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else float("nan")


def spearman(a, b):
    return pearson(rank(a), rank(b))


def partial_spearman(y, x, z):
    """Spearman of y and x with z partialled out, on ranks (residuals of OLS on rank(z))."""
    ry, rx, rz = rank(y), rank(x), rank(z)
    A = np.c_[np.ones(len(rz)), rz]
    res = lambda v: v - A @ np.linalg.lstsq(A, v, rcond=None)[0]
    return pearson(res(ry), res(rx))


def boot(fn, arrays, rng, n=N_BOOT):
    m = len(arrays[0])
    out = np.empty(n)
    for k in range(n):
        idx = rng.integers(0, m, m)
        out[k] = fn(*[a[idx] for a in arrays])
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def bh(pvals):
    """Benjamini-Hochberg adjusted p-values, charter stats stack."""
    p = np.asarray(pvals, float); o = p.argsort(); m = len(p)
    adj = np.empty(m); run = 1.0
    for i in range(m - 1, -1, -1):
        run = min(run, p[o[i]] * m / (i + 1)); adj[o[i]] = run
    return adj


def perm_p(stat_fn, y, x, rng, n=N_BOOT):
    """Two-sided permutation p: shuffle y against x, keeping any control vector fixed."""
    obs = abs(stat_fn(y, x))
    cnt = sum(abs(stat_fn(rng.permutation(y), x)) >= obs for _ in range(n))
    return (cnt + 1) / (n + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default=str(_PROJ / "data/nla/p0/trace_llr/gemma12b/llr_rows.jsonl"))
    ap.add_argument("--traces", default=str(_PROJ / "data/nla/p0/trace_llr/gemma12b/traces.jsonl"))
    ap.add_argument("--out", default=str(_PROJ / "data/nla/p0/trace_llr/gemma12b/length_confound.json"))
    a = ap.parse_args()
    from steer_run import SEED as S_SEED, load_pairs
    rng = np.random.default_rng(SEED)

    clean = {}
    for line in open(a.rows):
        r = json.loads(line); sid, *tag, which = r["key"].split("|")
        if which == "clean":
            clean[(sid, "|".join(tag))] = r["logp_sum"] / r["n_tok"]
    traces = {json.loads(l)["snippet_id"]: json.loads(l) for l in open(a.traces)}
    pairs = {p["snippet_id"]: p for p in load_pairs(None, random.Random(S_SEED))}

    sids, G, D, NR, FLIP = [], [], [], [], []
    for sid, t in traces.items():
        p = pairs.get(sid)
        nr = len(p.get("rename_map") or {}) if p else 0
        if nr == 0 or (sid, "l0prompt") not in clean:
            continue
        sids.append(sid)
        G.append(clean[(sid, "l0prompt")] - clean[(sid, "noop#1")])
        D.append(len(t["l1b_prompt_ids"]) - len(t["l0_prompt_ids"]))
        NR.append(nr)
        FLIP.append(bool(t["l0_correct"]) and not bool(t["l1b_correct"]))
    G, D, NR, FLIP = map(np.asarray, (G, D, NR, np.array(FLIP, bool)))
    INFL = D / NR

    r1 = spearman(G, D)
    ci1 = boot(spearman, [G, D], rng)
    p1 = perm_p(spearman, G, D, rng)

    r2 = partial_spearman(G, INFL, NR)
    f2 = lambda g, i, n: partial_spearman(g, i, n)
    ci2 = boot(f2, [G, INFL, NR], rng)
    # permutation for the partial: shuffle G, keep (INFL, NR) paired
    obs2 = abs(r2)
    cnt = sum(abs(partial_spearman(rng.permutation(G), INFL, NR)) >= obs2 for _ in range(N_BOOT // 10))
    p2 = (cnt + 1) / (N_BOOT // 10 + 1)

    q1, q2 = bh([p1, p2])
    confound = bool(abs(r2) >= RHO_MIN and (ci2[0] > 0 or ci2[1] < 0))
    verdict = "L-CONFOUND" if confound else "L-CLEAN"

    sec = {"n_flip": int(FLIP.sum()),
           "spearman_D_flip": spearman(D, FLIP.astype(float)),
           "partial_infl_flip": partial_spearman(FLIP.astype(float), INFL, NR),
           "note": "PRE-LABELLED UNDERPOWERED: 6 flips cannot support a claim either way"}
    res = {"experiment": "L_length_confound", "seed": SEED, "n": len(sids), "n_boot": N_BOOT,
           "rho_min": RHO_MIN,
           "H_L1": {"spearman_G_delta": r1, "ci95": ci1, "p_perm": p1, "q_bh": float(q1)},
           "H_L2": {"partial_spearman_G_infl_given_nr": r2, "ci95": ci2, "p_perm": p2, "q_bh": float(q2)},
           "descriptives": {"mean_G": float(G.mean()), "median_delta": float(np.median(D)),
                            "median_n_renames": float(np.median(NR)),
                            "median_infl_per_rename": float(np.median(INFL)),
                            "spearman_delta_nr": spearman(D, NR)},
           "secondary_accuracy": sec, "verdict": verdict}
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=2)

    print(f"n = {len(sids)} · mean G {G.mean():+.4f} · median Δ {np.median(D):+.0f} tok · "
          f"median renames {np.median(NR):.0f} · median inflation/rename {np.median(INFL):+.2f} tok")
    print(f"  Spearman(Δ, n_renames) = {spearman(D, NR):+.3f}   [how tangled the two predictors are]")
    print(f"H-L1  rho(G, Δ)                   = {r1:+.3f}  CI95 [{ci1[0]:+.3f},{ci1[1]:+.3f}]  "
          f"p {p1:.4f}  q {q1:.4f}")
    print(f"H-L2  rho_partial(G, infl | n_r)  = {r2:+.3f}  CI95 [{ci2[0]:+.3f},{ci2[1]:+.3f}]  "
          f"p {p2:.4f}  q {q2:.4f}   threshold |rho| >= {RHO_MIN}")
    print(f"secondary (UNDERPOWERED, {sec['n_flip']} flips): rho(Δ, flip) {sec['spearman_D_flip']:+.3f} · "
          f"partial(infl, flip | n_r) {sec['partial_infl_flip']:+.3f}")
    print(f"\nVERDICT {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
