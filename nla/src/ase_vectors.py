"""H-R6 vector fitting (CPU): from the pool capture, build the ORACLE-FREE vectors the `ridge_map` and
`role_proto` arms write into the 50 test snippets (nla/configs/ase_vectors.yaml; prereg
log/nla-harness/2026-09-14_better-vector-prereg.md).

SPLIT. TEST = the snippets of `paths.test_subset` (the bake-off's 50). POOL = every other aligned
snippet. Every vector written to a test snippet is a function of POOL states and of the test snippet's
own DECOY state / tags only -- its original prompt, clean state and true names never enter. This
replaces the bake-off's leave-one-item-out `erasure` (which fit on the other 49 test items): a bigger
pool and no leakage question at all.

`ridge_map`: reduced-rank ridge regression of delta = h0 - h1b on the centred decoy state,
    delta_hat = (h1b - mu) A + c,   v = h1b + delta_hat.
  A = 0 is exactly the mean-difference (`erasure`) construction, so the arm nests it; the map can
  only add what the decoy state predicts about its own repair (declared type and usage are in that
  state through context). Solved in the dual (n x n), rank-truncated on the fitted values (RRR).
  lambda / rank by grouped 5-fold CV on the pool (groups = snippets, criterion = held-out cosine to
  the true delta), the mean-only model scored the same way: the FROZEN PRE-GPU GATE is
  cos(best map) - cos(mean-only) >= gate_cos_margin, else the arm is not run.
`role_proto`: v = mean pool h0_mean over spans sharing the test span's `type|role` tag, falling back to
  `type`, then `kind` (method/variable) when a bucket has < min_bucket spans; fallback share reported.
"""
from __future__ import annotations

import argparse
import json
import zlib
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml


def rrr_fit(Xc: np.ndarray, Yc: np.ndarray, lam: float, rank: int) -> np.ndarray:
    """Ridge in the dual then reduced-rank truncation; returns A (d x d) with rank <= `rank`."""
    n = Xc.shape[0]
    G = Xc @ Xc.T
    alpha = np.linalg.solve(G + lam * np.eye(n), Yc)           # n x d
    A = Xc.T @ alpha                                          # d x d, full ridge solution
    if rank <= 0:
        return np.zeros_like(A)
    Yhat = Xc @ A
    # right singular vectors of the fitted values span the reduced-rank target subspace
    _, _, Vt = np.linalg.svd(Yhat, full_matrices=False)
    V = Vt[:rank].T
    return A @ V @ V.T


def cos_rows(P: np.ndarray, T: np.ndarray) -> np.ndarray:
    num = (P * T).sum(1)
    den = np.linalg.norm(P, axis=1) * np.linalg.norm(T, axis=1) + 1e-12
    return num / den


def cv_select(X: np.ndarray, Y: np.ndarray, groups: list[str], lambdas, ranks, folds: int, seed: int) -> dict:
    """Grouped K-fold over snippets. Returns per-(lambda, rank) mean held-out cosine and the mean-only score."""
    uniq = sorted(set(groups))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(uniq))
    fold_of = {uniq[i]: int(f) for f, i in zip(np.arange(len(uniq)) % folds, perm)}
    g = np.array([fold_of[s] for s in groups])
    scores: dict[tuple, list] = defaultdict(list)
    mean_only: list[float] = []
    for f in range(folds):
        tr, te = g != f, g == f
        mu, c = X[tr].mean(0), Y[tr].mean(0)
        Xc, Yc = X[tr] - mu, Y[tr] - c
        mean_only.append(float(cos_rows(np.broadcast_to(c, Y[te].shape), Y[te]).mean()))
        for lam in lambdas:
            # YAML 1.1 (PyYAML) reads `1.0e2` as a STRING (its float regex wants a signed exponent);
            # cast here so the config's ladder cannot silently poison the solve again (job 401200).
            lam = float(lam)
            n = Xc.shape[0]
            alpha = np.linalg.solve(Xc @ Xc.T + lam * np.eye(n), Yc)
            A_full = Xc.T @ alpha
            Yhat = Xc @ A_full
            _, _, Vt = np.linalg.svd(Yhat, full_matrices=False)
            for r in ranks:
                V = Vt[:r].T
                A = A_full @ V @ V.T
                pred = (X[te] - mu) @ A + c
                scores[(float(lam), int(r))].append(float(cos_rows(pred, Y[te]).mean()))
    table = {f"{lam:g}|{r}": float(np.mean(v)) for (lam, r), v in scores.items()}
    best = max(scores, key=lambda k: np.mean(scores[k]))
    return {"table": table, "best": {"lambda": best[0], "rank": best[1], "cos": float(np.mean(scores[best]))},
            "mean_only_cos": float(np.mean(mean_only)), "folds": folds}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs" / "ase_vectors.yaml"))
    ap.add_argument("--pool", default=None); ap.add_argument("--out", default=None)
    ap.add_argument("--report", default=None); ap.add_argument("--test-subset", default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config)); P = cfg["paths"]
    pool = torch.load(args.pool or P["pool"], weights_only=False)
    spans = pool["spans"]
    test_ids = {json.loads(l)["snippet"] for l in open(args.test_subset or P["test_subset"]) if l.strip()}
    fit = [s for s in spans if s["snippet"] not in test_ids]
    test = [s for s in spans if s["snippet"] in test_ids]
    n_fit_snip = len({s["snippet"] for s in fit})
    rep: dict = {"layer": pool["layer"], "model": pool["model"], "n_pool_spans": len(fit),
                 "n_pool_snippets": n_fit_snip, "n_test_spans": len(test),
                 "n_test_snippets": len({s["snippet"] for s in test}), "seed": cfg["seed"]}
    if n_fit_snip < int(cfg["split"]["min_pool_snippets"]):
        rep["verdict"] = "POOL-TOO-SMALL"; _dump(rep, args.report or P["fit_report"])
        print(rep); return 3
    assert not ({s["snippet"] for s in fit} & test_ids)

    X = np.stack([s["h1b_mean"].float().numpy() for s in fit]).astype(np.float64)
    H0 = np.stack([s["h0_mean"].float().numpy() for s in fit]).astype(np.float64)
    Y = H0 - X
    groups = [s["snippet"] for s in fit]
    R = cfg["ridge_map"]
    cv = cv_select(X, Y, groups, [float(l) for l in R["lambdas"]], [int(r) for r in R["ranks"]],
                   int(R["cv_folds"]), int(cfg["seed"]))
    cv["margin"] = cv["best"]["cos"] - cv["mean_only_cos"]
    cv["gate_pass"] = bool(cv["margin"] >= float(R["gate_cos_margin"]))
    rep["ridge_map"] = cv
    # final fit on the whole pool with the selected (lambda, rank)
    mu, c = X.mean(0), Y.mean(0)
    A = rrr_fit(X - mu, Y - c, cv["best"]["lambda"], cv["best"]["rank"])
    rep["ridge_map"]["mean_delta_norm"] = float(np.linalg.norm(c))
    rep["ridge_map"]["erasure_cos_check"] = float(cos_rows(np.broadcast_to(c, Y.shape), Y).mean())

    # role prototypes
    RP = cfg["role_proto"]; levels = RP["tag_levels"]; mb = int(RP["min_bucket"])
    def key(s, lvl):
        return {"type_role": f"{s['type']}|{s['role']}", "type": s["type"], "kind": s["kind"]}[lvl]
    buckets = {lvl: defaultdict(list) for lvl in levels}
    for s in fit:
        for lvl in levels:
            buckets[lvl][key(s, lvl)].append(s["h0_mean"].float().numpy())
    protos = {lvl: {k: np.mean(v, 0) for k, v in b.items() if len(v) >= mb} for lvl, b in buckets.items()}
    rep["role_proto"] = {"bucket_sizes": {lvl: {k: len(v) for k, v in b.items()} for lvl, b in buckets.items()},
                         "test_tag_counts": dict(Counter(key(s, "type_role") for s in test))}

    vec: dict = {"ridge_map": defaultdict(dict), "role_proto": defaultdict(dict)}
    resolved = Counter(); unresolved = []
    for s in test:
        sid, j = s["snippet"], s["span_idx"]
        x = s["h1b_mean"].float().numpy().astype(np.float64)
        v = x + (x - mu) @ A + c
        vec["ridge_map"][sid][j] = torch.tensor(v, dtype=torch.float32)
        for lvl in levels:
            k = key(s, lvl)
            if k in protos[lvl]:
                vec["role_proto"][sid][j] = torch.tensor(protos[lvl][k], dtype=torch.float32)
                resolved[lvl] += 1; break
        else:
            unresolved.append((sid, j, key(s, "type_role")))
    rep["role_proto"]["resolved_at"] = dict(resolved)
    rep["role_proto"]["unresolved"] = unresolved
    if unresolved:
        # a span with no prototype at any level would be an un-written span inside a written arm;
        # arm_guard would then refuse the contrast, so refuse here first
        rep["verdict"] = "PROTO-UNRESOLVED"; _dump(rep, args.report or P["fit_report"])
        print(f"[VEC] FATAL: {len(unresolved)} test spans without a prototype: {unresolved[:5]}"); return 3
    rep["verdict"] = "OK"
    rep["ridge_gate"] = "RUN ridge_map" if cv["gate_pass"] else "MAP-LEARNS-NOTHING (ridge_map not run)"
    rep["crc"] = zlib.crc32(json.dumps(cv["table"], sort_keys=True).encode())
    out = Path(args.out or P["vectors"])
    torch.save({"layer": pool["layer"], "config_sha": pool.get("config_sha"),
                "arms": {a: {sid: dict(d) for sid, d in m.items()} for a, m in vec.items()},
                "selected": cv["best"], "gate_pass": cv["gate_pass"]}, out)
    _dump(rep, args.report or P["fit_report"])
    print(f"[VEC] pool {len(fit)} spans / {n_fit_snip} snippets · test {len(test)} spans · "
          f"ridge best lambda={cv['best']['lambda']:g} rank={cv['best']['rank']} cos={cv['best']['cos']:.4f} "
          f"vs mean-only {cv['mean_only_cos']:.4f} (margin {cv['margin']:+.4f}, gate "
          f"{'PASS' if cv['gate_pass'] else 'FAIL'}) · proto resolved {dict(resolved)} · {out}")
    return 0


def _dump(rep: dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(rep, indent=1, default=str))


if __name__ == "__main__":
    raise SystemExit(main())
