"""Experiment W, stage 0 — can the NLA represent an obfuscated identifier position at all?

Pre-registered in `log/nla-harness/2026-09-03_nla-writeback-prereg.md`. This script decides
whether stages 1-2 are licensed to run; it is a **gate**, and a refusal here is a result.

The question. Every steering arm in this programme has written a direction built from a
*template string*. The NLA's own protocol is read -> edit -> reconstruct -> write back, and it
has never been executed on this host. Before spending GPU on the write-back, ask the cheaper
question it presupposes: **does the round trip AR(AV(h)) preserve h at these positions?** If the
verbalizer cannot describe an obfuscated identifier's residual distinctly enough for the
reconstructor to find its way back, then editing the read is meaningless and no write-back
experiment can be interpreted.

Two gates, in increasing strictness (both frozen in the prereg):

  H-W0a  matched cos > CROSS-ITEM mismatched cos     -> the round trip carries item identity
  H-W0b  matched cos > WITHIN-ITEM mismatched cos    -> it distinguishes THIS identifier from
                                                        another identifier in the same snippet

W0b is the one that licenses stage 1: a per-span edit needs a per-span referent. W0a alone
would be satisfied by a round trip that only recovers "some obfuscated Python", which is not
something you can edit toward a specific true meaning.

Why the within-item null is the hard one, and why both are reported: activations at nearby
positions of one prompt are strongly correlated by construction, so within-item mismatched
cosines are an *adversarial* baseline, while cross-item mismatched cosines are a lenient one.
Reporting only the lenient null would let shared-context similarity masquerade as identity.

DESIGN CHOICE - the span vector is the span's LAST token, not its mean.
Rejected: mean-pooling over the span's tokens. Under causal masking the last token of an
identifier is the only position that has seen the whole identifier, and the AV/AR pair was
trained on individual residual vectors, never on pooled ones - a pooled vector is off the
distribution both halves of the NLA were fit to, which would confound "the NLA cannot represent
identifiers" with "the NLA cannot represent averages". Recorded here because the rejected path
looks equally reasonable until you ask what the AV was trained on.

Secondary, descriptive, no gate (H-W3): do the reads mention the DECOY semantic field more often
than the TRUE one? That is the E1 semantic-capture question asked through Instrument 2 instead of
through an SAE, and it costs nothing once the reads exist.

Host note: this is the FIRST use of the Gemma AV anywhere in the project - all 16 other AV call
sites hardcode the Qwen checkpoint (d_model 3584). The Gemma pair is d_model 3840,
injection_scale 80000 vs Qwen's 150. The CJK injection-failure alarm is therefore load-bearing
rather than decorative: a wrong injection scale puts the vector off-distribution and the
verbalizer free-associates in Chinese instead of failing loudly.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_NLA_ROOT = _HERE.parent
_PROJ = _NLA_ROOT.parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_NLA_ROOT / "vendor" / "nla-repo"))

from extract import ActivationExtractor  # noqa: E402
from span_positions import span_token_positions  # noqa: E402
from steer_run import (HOSTS, AR_CHECKPOINTS, _hf_snapshot, build_user,  # noqa: E402
                       load_pairs, SEED)

# The AR already has a per-host map; the AV never needed one because no AV run had ever left the
# Qwen host. Mirrored here rather than hardcoded so the two halves cannot silently disagree about
# which model they are describing - an AR/AV mismatch produces cosines that mean nothing at all.
AV_CHECKPOINTS = {
    "qwen7b":   _NLA_ROOT / "data" / "checkpoints" / "av",
    "gemma12b": _hf_snapshot("kitft/nla-gemma3-12b-L32-av"),
}

MAX_NEW_READ = 96          # matches capture_core.MAX_NEW_READ
N_CROSS = 5                # cross-item mismatched draws per span
N_BOOT = 10_000            # prereg
CJK_ABORT_RATE = 0.5       # >50% of a 100-read window mostly-CJK => injection failure, abort


def cosv(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def cluster_boot(by_item: list[np.ndarray], rng: np.random.Generator,
                 n: int = N_BOOT) -> tuple[float, float, float]:
    """Percentile CI over a mean, resampling ITEMS not spans.

    Spans inside one snippet share a prompt and are not independent draws; bootstrapping over
    spans would shrink the CI by the average spans-per-item and could manufacture a separation
    between matched and mismatched that the item-level data does not support.
    """
    vals = np.concatenate(by_item)
    obs = float(vals.mean())
    if len(by_item) < 2:
        return obs, float("nan"), float("nan")
    idx = np.arange(len(by_item))
    draws = np.empty(n)
    for i in range(n):
        pick = rng.choice(idx, size=len(idx), replace=True)
        draws[i] = np.concatenate([by_item[j] for j in pick]).mean()
    return obs, float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def mentions(text: str, terms: list[str]) -> bool:
    """Frozen, deterministic, no judge: case-insensitive substring of any term >= 3 chars.

    Substring rather than word-boundary because identifiers are routinely embedded in the
    verbalizer's prose as morphology ("a fibfib-like sequence"), which a \\b match would miss.
    The >=3 filter keeps one-letter loop variables from matching essentially every read.
    """
    low = text.lower()
    return any(len(t) >= 3 and t.lower() in low for t in terms)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma12b", choices=sorted(HOSTS))
    ap.add_argument("--allow-banked-host", action="store_true",
                    help="required for qwen7b (2026-09-02 model constraint: no Chinese models)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--sglang-url", default="http://localhost:30000")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ar-device", default=None,
                    help="device for the AR; defaults to --device. Set 'cpu' when the subject, "
                         "the sglang AV and the AR do not co-fit on one card (three 12B models).")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-spans-per-item", type=int, default=8,
                    help="matches identifier_terms()'s own [:8] cap, so spans and terms agree")
    ap.add_argument("--smoke", action="store_true",
                    help="3 items; asserts reads are non-CJK and matched cos > 0 before the "
                         "full run is allowed to cost anything")
    ap.add_argument("--max-hours", type=float, default=4.0)
    args = ap.parse_args()

    if args.model == "qwen7b" and not args.allow_banked_host:
        print("[W0] REFUSED: qwen7b is a Chinese model; pass --allow-banked-host to override.")
        return 2

    model_name, layer = HOSTS[args.model]
    av_dir, ar_dir = AV_CHECKPOINTS.get(args.model), AR_CHECKPOINTS.get(args.model)
    if av_dir is None or ar_dir is None:
        print(f"[W0] REFUSED: no NLA pair for host {args.model} "
              f"(av={av_dir}, ar={ar_dir})")
        return 2

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    from nla_inference import NLAClient, NLACritic  # noqa: E402

    print(f"[W0] host={args.model} {model_name} L{layer}  av={av_dir}  ar={ar_dir}", flush=True)
    ex = ActivationExtractor(model_name, layer, device=args.device)
    av = NLAClient(av_dir, sglang_url=args.sglang_url)
    ar = NLACritic(ar_dir, device=args.ar_device or args.device)
    tokz = ex.tokenizer

    pairs = load_pairs(args.limit or (3 if args.smoke else None), random.Random(SEED))
    if args.smoke:
        pairs = pairs[:3]

    rows: list[dict] = []
    cjk_window: list[bool] = []
    skipped: dict[str, int] = {}

    for pi, p in enumerate(pairs):
        if time.time() - t0 > args.max_hours * 3600:
            skipped["wall_clock"] = skipped.get("wall_clock", 0) + 1
            break
        user = build_user(p["code_l1b"], p["call_l1b"])
        spans = (p.get("id_spans_l1b") or [])[: args.max_spans_per_item]
        if not spans:
            skipped["no_spans"] = skipped.get("no_spans", 0) + 1
            continue
        # One forward pass per item; every span indexes into the same hidden states.
        full = ex.extract_chat(user, positions=None, text_id=p["snippet_id"])
        H = full.activations                                   # [T, d]

        for si, sp in enumerate(spans):
            pos, diag = span_token_positions(tokz, user, p["code_l1b"], [sp])
            if not pos:
                skipped[f"span:{diag.get('reason')}"] = \
                    skipped.get(f"span:{diag.get('reason')}", 0) + 1
                continue
            last = max(pos)                                    # see DESIGN CHOICE in docstring
            if last >= len(H):
                skipped["span_past_end"] = skipped.get("span_past_end", 0) + 1
                continue
            h = H[last].astype(np.float32)
            text = av.generate(h, temperature=0.0, max_new_tokens=MAX_NEW_READ)
            rec = ar.reconstruct(text).numpy().astype(np.float32)
            cjk = sum(1 for ch in text if "一" <= ch <= "鿿") / max(len(text), 1)
            cjk_window.append(cjk > 0.2)
            if len(cjk_window) > 100:
                cjk_window.pop(0)
            if len(cjk_window) >= 20 and sum(cjk_window) / len(cjk_window) > CJK_ABORT_RATE:
                print(f"[W0] ABORT: CJK rate {sum(cjk_window)}/{len(cjk_window)} — the Gemma AV "
                      f"injection is failing (wrong injection_scale?). Not a null; a broken "
                      f"instrument.", flush=True)
                json.dump({"verdict": "W0-INJECTION-FAILURE",
                           "cjk_rate": sum(cjk_window) / len(cjk_window),
                           "n_reads": len(rows)},
                          open(out_dir / "cycle_stats.json", "w"), indent=2)
                return 3
            term = p["code_l1b"][int(sp[0]):int(sp[1])].strip()
            rows.append({
                "snippet_id": p["snippet_id"], "span_i": si, "term": term,
                "pos_last": last, "n_pos": len(pos), "read": text, "cjk_frac": cjk,
                "h": h.tolist(), "rec": rec.tolist(),
                "mentions_decoy": mentions(text, p["terms_decoy"]),
                "mentions_true": mentions(text, p["terms_true"]),
            })
        print(f"[W0] {pi+1}/{len(pairs)} {p['snippet_id']}  spans_done={len(rows)}", flush=True)

    if not rows:
        json.dump({"verdict": "W0-NO-DATA", "skipped": skipped},
                  open(out_dir / "cycle_stats.json", "w"), indent=2)
        print("[W0] REFUSED: no usable spans.", flush=True)
        return 2

    # ── cosines ──────────────────────────────────────────────────────────────
    rng = np.random.default_rng(SEED)
    H = np.array([r["h"] for r in rows], dtype=np.float32)
    R = np.array([r["rec"] for r in rows], dtype=np.float32)
    sids = [r["snippet_id"] for r in rows]

    matched, within, cross = {}, {}, {}
    for i in range(len(rows)):
        s = sids[i]
        matched.setdefault(s, []).append(cosv(R[i], H[i]))
        same = [j for j in range(len(rows)) if sids[j] == s and j != i]
        for j in same:
            within.setdefault(s, []).append(cosv(R[i], H[j]))
        other = [j for j in range(len(rows)) if sids[j] != s]
        if other:
            for j in rng.choice(other, size=min(N_CROSS, len(other)), replace=False):
                cross.setdefault(s, []).append(cosv(R[i], H[int(j)]))

    def pack(d):
        return [np.array(v, dtype=float) for v in d.values() if len(v)]

    m_obs, m_lo, m_hi = cluster_boot(pack(matched), np.random.default_rng(SEED))
    w_obs, w_lo, w_hi = cluster_boot(pack(within), np.random.default_rng(SEED + 1)) \
        if pack(within) else (float("nan"),) * 3
    c_obs, c_lo, c_hi = cluster_boot(pack(cross), np.random.default_rng(SEED + 2)) \
        if pack(cross) else (float("nan"),) * 3

    # Frozen rule: separation = NON-OVERLAPPING 95% CIs, matched above. This is the R2
    # convention, adopted after R's `R-1` was found to have used a permissive "lower by more
    # than a floor" clause that a matched random control also cleared.
    w0a = bool(np.isfinite(c_lo) and m_lo > c_hi)
    w0b = bool(np.isfinite(w_lo) and m_lo > w_hi)
    verdict = "NLA-LIVE" if (w0a and w0b) else ("NLA-COARSE" if w0a else "NLA-VOID")

    n_items = len(set(sids))
    dec = np.array([r["mentions_decoy"] for r in rows], dtype=float)
    tru = np.array([r["mentions_true"] for r in rows], dtype=float)
    by_item = lambda arr: [arr[[k for k in range(len(rows)) if sids[k] == s]]  # noqa: E731
                           for s in sorted(set(sids))]
    d_obs, d_lo, d_hi = cluster_boot(by_item(dec), np.random.default_rng(SEED + 3))
    t_obs, t_lo, t_hi = cluster_boot(by_item(tru), np.random.default_rng(SEED + 4))

    stats = {
        "experiment": "W0_cycle_consistency",
        "host": args.model, "model": model_name, "layer": layer,
        "av": str(av_dir), "ar": str(ar_dir),
        "seed": SEED, "n_boot": N_BOOT,
        "n_items": n_items, "n_spans": len(rows), "skipped": skipped,
        "cjk_mean": float(np.mean([r["cjk_frac"] for r in rows])),
        "cos": {
            "matched":        {"mean": m_obs, "ci95": [m_lo, m_hi]},
            "within_item":    {"mean": w_obs, "ci95": [w_lo, w_hi]},
            "cross_item":     {"mean": c_obs, "ci95": [c_lo, c_hi]},
        },
        "H_W0a_cross_item_separation": w0a,
        "H_W0b_within_item_separation": w0b,
        "H_W3_mention_rate": {
            "decoy": {"mean": d_obs, "ci95": [d_lo, d_hi]},
            "true":  {"mean": t_obs, "ci95": [t_lo, t_hi]},
        },
        "verdict": verdict,
        "licenses_stage1": verdict == "NLA-LIVE",
        "elapsed_h": (time.time() - t0) / 3600,
    }
    json.dump(stats, open(out_dir / "cycle_stats.json", "w"), indent=2)
    # Vectors are dropped from the persisted rows: 3840 floats x n_spans is large and every
    # downstream question is answered by the cosines. Reads are kept - they are the H-W3
    # evidence and stage 1's edit targets.
    with open(out_dir / "cycle_rows.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps({k: v for k, v in r.items() if k not in ("h", "rec")}) + "\n")

    print(json.dumps({k: v for k, v in stats.items() if k != "skipped"}, indent=2), flush=True)

    if args.smoke:
        assert stats["cjk_mean"] < 0.2, f"smoke: CJK {stats['cjk_mean']:.3f} — injection broken"
        assert m_obs > 0.0, f"smoke: matched cos {m_obs:.3f} <= 0 — round trip is not working"
        print("[W0] smoke OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
