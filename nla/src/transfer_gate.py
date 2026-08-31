"""B1 — does the released NLA read a *different* base model's residual stream?

The released pair `kitft/nla-qwen2.5-7b-L20-{av,ar}` was trained on
`Qwen/Qwen2.5-7B-Instruct` layer-20 activations. `Qwen2.5-Coder-7B-Instruct` is the same
architecture (`qwen2`, d_model=3584, 28 layers, same tokenizer), so the checkpoint *loads*
against it. Whether it *reads* it is empirical, forward-pass-only, and the NLA release says
nothing about cross-model transfer — which is what makes this a contribution and not just a
gate. It is also the branch point for the whole paper: pass and the subject model is the Coder
(matching CodeSteer's primary model and the 244 banked replication runs), fail and we fall back
to Qwen2.5-7B-Instruct and regenerate the CodeSteer conditions on it.

**The comparison, not the absolute, is what decides.** A round-trip cosine of 0.7 on code means
nothing on its own — code is a different distribution from the NLA's training text. So this
runner is invoked once per model over *identical inputs* and the two runs are compared by
`--compare`. The criterion is how far the subject's median cosine falls below the control's,
with the threshold frozen in the config before the run. `fve_nrm` is reported for comparability
with the reference's published band but does **not** gate — see its docstring.

**The pairing is exact, not approximate.** Qwen2.5-7B-Instruct and Qwen2.5-Coder-7B-Instruct
ship the *same* `tokenizer.json` blob and a byte-identical chat template, and agree on
hidden_size / layers / heads / kv-heads / vocab. So for any input string the two models
produce the identical token sequence and identical position indices: the only variable in this
comparison is the weights. That is what makes a deterministic position rule (below) a genuinely
paired design rather than an approximate one.

**Every headline number carries a null attached to *that* number.** The programme's most
expensive lesson (N10's saturated capability statistic, N11's echo-free residue sitting inside
its own foreign-read null) is that a plausible own-item rate means nothing without the rate you
get from a *different* item's activation. So each explanation is also scored against a foreign
activation — a different item's vector, at matched locus — and the gate is own-vs-foreign, not
own-vs-zero. This is nearly free: the reconstruction is computed once and compared to two golds.

On the metric identity, verified against the reference example
(`vendor/nla-repo/examples/qwen7b_layer20_step4200.txt`, 101 tokens): `mse_nrm = 2(1 - cos)` to
within its own 3-decimal rounding (max deviation 0.001), and the reference's `fve_nrm` is
`1 - mse/denom` with `denom` constant at ~0.734 across all tokens — a corpus-level variance
baseline, which is the shape `fve_nrm()` below reproduces on our own split.

Usage (one model per invocation — the subject model and the AV server share a card):

    # with an AV server already up (see scripts/transfer_gate.sh)
    python -m src.transfer_gate run --model Qwen/Qwen2.5-Coder-7B-Instruct --tag coder
    python -m src.transfer_gate run --model Qwen/Qwen2.5-7B-Instruct       --tag instruct
    python -m src.transfer_gate compare --a coder --b instruct

Why a new runner rather than a `--model` flag threaded through the existing ones: nine runners
carry `TARGET_MODEL` as a module constant and three N13 modules import it *from*
`overnight_capture`, so threading a flag through would touch every banked-corpus code path for
the sake of one forward-pass-only experiment. This module reuses the pieces that are already
model-agnostic — `extract.ActivationExtractor` (its `_layers()` probe handles any of the three
common layer paths), `capture_core.align_reply` / `token_ok` (tokenizer passed in), and the
CJK-abort constants — and hardcodes nothing about the subject.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_NLA_ROOT = _HERE.parent
_PROJ = _NLA_ROOT.parent
sys.path.insert(0, str(_NLA_ROOT / "vendor" / "nla-repo"))
sys.path.insert(0, str(_HERE))

from capture_core import (  # noqa: E402
    CJK_FRAC_ABORT, CJK_MIN_SAMPLES, CJK_RATE_ABORT, CJK_WINDOW, CJK_RE,
    JsonlSink, WallGuard, align_reply, token_ok,
)
from deception_capture import OBF_ROOT, PREAMBLE, SRC_ROOT  # noqa: E402
from extract import MIN_POSITION, ActivationExtractor, sha256_file  # noqa: E402

LAYER_INDEX = 20
SEED = 20260724
MAX_NEW_READ = 200
OUT_DIR = _PROJ / "data" / "nla" / "b1"

# The NLA host, and the control arm of every comparison.
CONTROL_MODEL = "Qwen/Qwen2.5-7B-Instruct"
# Qwen L20 activation norms sit in this band; `injection_scale=150` was tuned for it, and a
# subject whose norms sit far outside it is the classic silent injection failure.
NORM_BAND = (100.0, 170.0)

# ─── gate thresholds (plan v0.3, docs/PLAN_believe_the_lie.md "Gate E1") ────────────────
# v0.3's gate led with fve_nrm; it is demoted to a reported diagnostic here for the reasons in
# `fve_nrm`'s docstring, and the cosine drop against the control takes its place as the
# transfer criterion. Recorded as a deviation in the pre-registration rather than silently.
GATE_COS_MEDIAN = 0.70       # established code-text round-trip band is 0.70-0.96
GATE_COS_DELTA = 0.05        # max median-cosine drop below the control on identical inputs
GATE_NORM_RATIO = 2.0        # median norm within ~2x of the control band
GATE_CJK_FRAC = 0.01         # mostly-CJK reads < 1%
# The null attached to the primary: own-activation cosine must clear the foreign-activation
# cosine with a 95% CI that excludes 0. Reads carry SOME shared structure (they are all
# "activation from a Java program"), so the foreign rate is not 0 and a raw own-rate is not
# evidence. N10's item-level statistic died exactly here: own 0.874 < foreign 0.903.
GATE_OWN_MINUS_FOREIGN = 0.0
N_BOOT = 2000


def load_snippets(dataset: str, n: int, rng: random.Random) -> list[dict]:
    """Original (C0) Java sources, deterministically sampled.

    Reuses the same corpus and manifest the B3/B4 runs read, so a transfer number here is
    about the *model*, not about a different pile of code.
    """
    root = OBF_ROOT / dataset
    manifest = json.load(open(root / "build_manifest.json"))
    src_dir = SRC_ROOT / ("Humaneval" if dataset == "humaneval" else "Cruxeval") / "java"

    out = []
    for snippet in sorted({k.split("/")[0] for k in manifest["entries"]}):
        p = src_dir / f"{snippet}.java"
        if p.exists():
            out.append({"snippet": snippet, "dataset": dataset, "code": p.read_text()})
    rng.shuffle(out)
    return out[:n]


def pick_positions(align, code_start: int, code_end: int, k: int) -> list[int]:
    """`k` evenly spaced token positions inside the code region.

    Even spacing rather than random: the two models must be read at the *same construct*, and
    with identical inputs and the same tokenizer the token grids coincide, so a deterministic
    rule gives a genuinely paired comparison. Positions below `MIN_POSITION` decode to noise
    (the reference's own datagen floor) and single-char / pure-digit pieces verbalize as
    numerology — both are excluded rather than silently averaged in.
    """
    usable = [
        i for i, (s, e) in enumerate(align.offsets)
        if i >= MIN_POSITION and code_start <= s < code_end and token_ok(align.full, s, e)
    ]
    if len(usable) <= k:
        return usable
    idx = np.linspace(0, len(usable) - 1, k).round().astype(int)
    return [usable[i] for i in sorted(set(idx.tolist()))]


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    """Plain cosine. The critic normalizes both vectors to `mse_scale` before scoring, which
    leaves the cosine unchanged, so this reproduces `NLACritic.score`'s second return value."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / max(na * nb, 1e-12))


def foreign_null(
    recs: np.ndarray, golds: np.ndarray, snippets: list[str], rng: random.Random
) -> np.ndarray:
    """Cosine of each reconstruction against a *different snippet's* activation.

    Foreign at the snippet level, not the read level: two reads from the same program share
    its subject matter, so a within-snippet 'foreign' vector would smuggle the very signal
    the null is supposed to strip.
    """
    by_snip: dict[str, list[int]] = {}
    for i, s in enumerate(snippets):
        by_snip.setdefault(s, []).append(i)
    others = sorted(by_snip)
    out = np.empty(len(recs), dtype=float)
    for i, s in enumerate(snippets):
        pool = [o for o in others if o != s]
        j = rng.choice(by_snip[rng.choice(pool)]) if pool else i
        out[i] = _cos(recs[i], golds[j])
    return out


def cluster_bootstrap_ci(
    own: np.ndarray, foreign: np.ndarray, snippets: list[str], rng: random.Random
) -> tuple[float, float, float]:
    """(mean difference, lo, hi) resampling SNIPPETS, not reads.

    Reads within a snippet are not independent — the same lesson `n13_torn.py` encodes and
    that the 2026-08-06 faithfulness result violated (read-level bootstrap, no case
    clustering, which is what made it a length artifact).
    """
    d = own - foreign
    groups: dict[str, list[int]] = {}
    for i, s in enumerate(snippets):
        groups.setdefault(s, []).append(i)
    keys = sorted(groups)
    boots = []
    for _ in range(N_BOOT):
        picked = [groups[keys[rng.randrange(len(keys))]] for _ in keys]
        idx = [i for g in picked for i in g]
        boots.append(float(d[idx].mean()))
    boots.sort()
    return (
        float(d.mean()),
        boots[int(0.025 * len(boots))],
        boots[int(0.975 * len(boots)) - 1],
    )


MIN_READS_FOR_FVE = 100


def fve_nrm(cosines: np.ndarray, golds: np.ndarray, mse_scale: float) -> float:
    """Fraction of variance explained, in the critic's normalized space, per split.

    Reported for comparability with the reference's published 0.6-0.8 band. **It is not a gate
    criterion**, for two reasons found while smoke-testing this module:

    1. `MSE = 2(1 - cos)` exactly, so with a denominator held fixed across arms, FVE is a
       monotone reparameterization of mean cosine and carries no independent information.
    2. With a *per-split* denominator it does carry extra information — but that information
       is how dispersed this model's activation cloud is, which is a property of the model,
       not of how well the NLA reads it. Gating on it would penalize a subject for having
       tighter activations.

    It is also unstable at small n: the denominator is the variance of the normalized golds
    about their own mean, so on a handful of reads it collapses and FVE goes negative (the
    3-snippet smoke returned -0.91 on the subject and +0.17 on the control, from the same
    pipeline that produced perfectly sane cosines). Hence the n guard.
    """
    if len(cosines) < MIN_READS_FOR_FVE:
        return float("nan")
    g = golds / np.maximum(np.linalg.norm(golds, axis=1, keepdims=True), 1e-12) * mse_scale
    hbar = g.mean(axis=0)
    denom = float(((g - hbar) ** 2).mean())
    loss = float(np.mean(2.0 * (1.0 - cosines)))
    return 1.0 - loss / denom if denom > 0 else float("nan")


def run(args: argparse.Namespace) -> int:
    from nla_inference import NLAClient, NLACritic  # noqa: E402

    global OUT_DIR
    if getattr(args, "out_dir", None):
        OUT_DIR = Path(args.out_dir)

    rng = random.Random(args.seed)
    snippets = load_snippets(args.dataset, args.n_snippets, rng)
    print(f"[b1] {len(snippets)} snippets from {args.dataset}, model={args.model}")

    ex = ActivationExtractor(args.model, LAYER_INDEX, device=args.device)

    # The one assertion that turns a silent garbage run into a loud failure: the AV/AR were
    # trained at a fixed width, and a subject of a different d_model would fail deep inside
    # _build_embeds with an unrelated-looking shape error.
    av_dir = _NLA_ROOT / "data" / "checkpoints" / "av"
    ar_dir = _NLA_ROOT / "data" / "checkpoints" / "ar"
    import yaml
    av_meta = yaml.safe_load((av_dir / "nla_meta.yaml").read_text())
    # schema_version 2 puts d_model at the top level; tolerate the nested layouts rather than
    # assuming one, since the sidecar is the contract and its shape is not ours to fix.
    d_nla = (av_meta.get("d_model")
             or (av_meta.get("model") or {}).get("d_model")
             or (av_meta.get("extraction") or {}).get("d_model"))
    assert d_nla, f"no d_model in {av_dir}/nla_meta.yaml — not an NLA sidecar?"
    d_nla = int(d_nla)
    assert ex.d_model == d_nla, (
        f"d_model mismatch: subject {args.model} has {ex.d_model}, NLA expects {d_nla}. "
        f"This checkpoint cannot read this model at all."
    )

    av = NLAClient(av_dir, sglang_url=args.sglang_url)
    ar = NLACritic(ar_dir, device=args.ar_device)

    # Reads are done inline rather than via capture_core.ReadEngine for one reason: the
    # engine calls NLACritic.score(), which reconstructs internally and returns only the
    # scalar. The foreign null needs the reconstructed VECTOR, and computing it once yields
    # both the own-cosine (identical to what score() returns — normalizing to mse_scale does
    # not change a cosine) and the foreign one. Reusing the engine would mean a second AR
    # forward per read for a number we already hold. The CJK abort semantics are preserved
    # exactly by importing the engine's own constants.
    cjk_recent: list[bool] = []

    def cjk_alarm() -> str | None:
        window = cjk_recent[-CJK_WINDOW:]
        if len(window) >= CJK_MIN_SAMPLES:
            rate = sum(window) / len(window)
            if rate > CJK_RATE_ABORT:
                return f"CJK rate {sum(window)}/{len(window)} — injection failure"
        return None

    # Append-and-flush per read, resumable by key. The first version accumulated everything in
    # memory and wrote once at the end, which is fine for a 3-snippet smoke and wrong for the
    # real run: at the measured 8.8 s/read (3.8 s AV on GPU + ~5 s AR on CPU) two arms take
    # ~4.9 h, and a server death at hour four would have lost all of it. Every other runner in
    # this harness uses JsonlSink for exactly this reason; this one was the outlier.
    # The reconstruction is stored alongside the read so a resumed run can rebuild the foreign
    # null without re-verbalizing anything.
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sink = JsonlSink(OUT_DIR / f"reads_{args.tag}.jsonl", key_field="read_key")
    done_keys = sink.done_keys()
    banked = [json.loads(l) for l in open(OUT_DIR / f"reads_{args.tag}.jsonl")
              if l.strip()] if (OUT_DIR / f"reads_{args.tag}.jsonl").exists() else []
    banked = [r for r in banked if "error" not in r]
    if banked:
        print(f"[b1] resuming: {len(banked)} reads already banked", flush=True)

    rows: list[dict] = [{k: v for k, v in r.items() if k != "rec"} for r in banked]
    golds: list[np.ndarray] = [np.array(r["gold"], dtype=np.float32) for r in banked]
    recs: list[np.ndarray] = [np.array(r["rec"], dtype=np.float32) for r in banked]
    wall = WallGuard(args.max_hours)
    for si, s in enumerate(snippets):
        if wall.expired():
            print(f"[b1] wall budget {args.max_hours} h reached — stopping cleanly", flush=True)
            break
        user = PREAMBLE + s["code"]
        align = align_reply(ex.tokenizer, user, "")
        code_start = align.full.index(s["code"][:40])
        # `code_end` must be where the CODE ends, not where the string ends. Passing
        # len(align.full) let the evenly-spaced grid run past the code into the chat-template
        # tail (`<|im_end|>\n<|im_start|>assistant\n`), so slot 4 of 5 landed on the token
        # `assistant` in 100% of snippets — 20% of every read budget spent on a non-code token,
        # and a "cosine on code reads" median diluted by a different token class. Found
        # 2026-08-16 mid-run; the affected run kept the defect deliberately (changing positions
        # mid-flight would have broken the exact pairing between arms) and filtered those reads
        # at analysis time instead.
        code_end = code_start + len(s["code"])
        positions = pick_positions(align, code_start, code_end, args.reads_per_snippet)
        if not positions:
            print(f"[b1] {s['snippet']}: no usable positions, skipped")
            continue

        todo = [p for p in positions if f"{s['snippet']}|{p}" not in done_keys]
        if not todo:
            continue
        res = ex.extract(align.full, positions=todo, text_id=s["snippet"])
        for i, pos in enumerate(res.positions):
            v = res.activations[i]
            text = av.generate(v, temperature=0.0, max_new_tokens=args.max_new_read)
            rec = ar.reconstruct(text).numpy()
            frac = len(CJK_RE.findall(text)) / max(len(text), 1)
            cjk_recent.append(frac > CJK_FRAC_ABORT)
            recs.append(rec)
            sp, epos = align.offsets[pos]
            row = {
                "read_key": f"{s['snippet']}|{pos}",
                "model": args.model,
                "snippet": s["snippet"],
                "dataset": s["dataset"],
                "position": int(pos),
                "token": align.full[sp:epos],
                "act_norm": float(np.linalg.norm(v)),
                "rt_cos": _cos(rec, v),
                "cjk_frac": frac,
                "read": text,
            }
            rows.append(dict(row))
            golds.append(v)
            # Vectors ride along so a resume can rebuild the null without re-verbalizing.
            sink.append({**row, "gold": v.tolist(), "rec": rec.tolist()})

        alarm = cjk_alarm()
        if alarm:
            print(f"[b1] ABORT — {alarm}", flush=True)
            return 3
        if (si + 1) % 10 == 0:
            cs = np.array([x["rt_cos"] for x in rows])
            print(f"[b1] {si + 1}/{len(snippets)} snippets · {len(rows)} reads · "
                  f"median rt_cos {np.median(cs):.3f} · {wall.elapsed_h():.2f} h",
                  flush=True)

    cos = np.array([r["rt_cos"] for r in rows])
    norms = np.array([r["act_norm"] for r in rows])
    cjk = float(np.mean([r["cjk_frac"] > 0.30 for r in rows])) if rows else float("nan")

    snips = [r["snippet"] for r in rows]
    null_rng = random.Random(args.seed + 1)
    foreign = foreign_null(np.array(recs), np.array(golds), snips, null_rng)
    for r, f in zip(rows, foreign):
        r["foreign_cos"] = float(f)
    d_mean, d_lo, d_hi = cluster_bootstrap_ci(cos, foreign, snips, random.Random(args.seed + 2))

    summary = {
        "model": args.model,
        "tag": args.tag,
        "layer_index": LAYER_INDEX,
        "dataset": args.dataset,
        "seed": args.seed,
        "n_snippets": len({r["snippet"] for r in rows}),
        "n_reads": len(rows),
        "cos_median": float(np.median(cos)) if rows else float("nan"),
        "cos_mean": float(cos.mean()) if rows else float("nan"),
        "cos_q25": float(np.quantile(cos, 0.25)) if rows else float("nan"),
        "cos_q75": float(np.quantile(cos, 0.75)) if rows else float("nan"),
        "fve_nrm": fve_nrm(cos, np.array(golds), ar.mse_scale) if rows else float("nan"),
        "foreign_cos_median": float(np.median(foreign)) if rows else float("nan"),
        "own_minus_foreign": d_mean,
        "own_minus_foreign_lo": d_lo,
        "own_minus_foreign_hi": d_hi,
        "act_norm_median": float(np.median(norms)) if rows else float("nan"),
        "act_norm_q05": float(np.quantile(norms, 0.05)) if rows else float("nan"),
        "act_norm_q95": float(np.quantile(norms, 0.95)) if rows else float("nan"),
        "cjk_read_frac": cjk,
        "mse_scale": float(ar.mse_scale),
        "script_sha256": sha256_file(__file__),
        "ts": datetime.now(timezone.utc).isoformat(),
    }

    sink.close()
    (OUT_DIR / f"summary_{args.tag}.json").write_text(json.dumps(summary, indent=1))
    ex.close()

    print(json.dumps({k: v for k, v in summary.items() if k != "script_sha256"}, indent=1))
    print(f"[b1] wrote {OUT_DIR}/summary_{args.tag}.json")
    return 0


def compare(args: argparse.Namespace) -> int:
    """The gate table. `--a` is the new subject, `--b` the control (the NLA's own host)."""
    global OUT_DIR
    if getattr(args, "out_dir", None):
        OUT_DIR = Path(args.out_dir)
    a = json.loads((OUT_DIR / f"summary_{args.a}.json").read_text())
    b = json.loads((OUT_DIR / f"summary_{args.b}.json").read_text())

    checks = []
    checks.append((
        "median rt_cos >= 0.70",
        a["cos_median"] >= GATE_COS_MEDIAN,
        f"{a['cos_median']:.3f} (control {b['cos_median']:.3f})",
    ))
    # The comparison that actually decides transfer: how far the subject falls below the NLA's
    # own host on identical inputs. delta is frozen in the config before the run.
    drop = b["cos_median"] - a["cos_median"]
    checks.append((
        f"within {GATE_COS_DELTA:.2f} of control",
        drop <= GATE_COS_DELTA,
        f"control - subject = {drop:+.3f}",
    ))
    lo, hi = NORM_BAND
    in_band = (lo / GATE_NORM_RATIO) <= a["act_norm_median"] <= (hi * GATE_NORM_RATIO)
    checks.append((
        "act_norm median within 2x band",
        in_band,
        f"{a['act_norm_median']:.1f} (control {b['act_norm_median']:.1f}, band {lo}-{hi})",
    ))
    checks.append((
        "mostly-CJK reads < 1%",
        a["cjk_read_frac"] < GATE_CJK_FRAC,
        f"{a['cjk_read_frac']:.4f}",
    ))
    # The null attached to the primary. A high own-cosine that does not clear its own
    # foreign rate is the N10 failure mode and means the reads carry nothing item-specific.
    checks.append((
        "own > foreign (95% CI excl. 0)",
        a["own_minus_foreign_lo"] > GATE_OWN_MINUS_FOREIGN,
        f"own {a['cos_median']:.3f} vs foreign {a['foreign_cos_median']:.3f}, "
        f"Δ {a['own_minus_foreign']:+.3f} [{a['own_minus_foreign_lo']:+.3f}, "
        f"{a['own_minus_foreign_hi']:+.3f}]",
    ))

    print(f"\nB1 transfer gate — subject {a['model']}  vs control {b['model']}")
    print(f"{a['n_reads']} vs {b['n_reads']} reads over {a['n_snippets']} identical snippets\n")
    print(f"{'check':<32} {'verdict':<8} detail")
    for name, ok, detail in checks:
        print(f"{name:<32} {'PASS' if ok else 'FAIL':<8} {detail}")

    passed = all(ok for _, ok, _ in checks)
    print("\nGATE " + ("PASS — subject model is " + a["model"] if passed else "FAIL"))
    if not passed:
        print("Decision rule (plan v0.3): norm band off but content sane -> re-derive "
              "injection_scale from this model's norm distribution and re-run, reporting "
              "transfer quality as a first-class limitation. Content degraded -> fall back "
              "to Qwen2.5-7B-Instruct and regenerate the CodeSteer conditions on it.")

    out = OUT_DIR / f"gate_{args.a}_vs_{args.b}.json"
    out.write_text(json.dumps({
        "subject": a, "control": b, "passed": passed,
        "checks": [{"name": n, "pass": bool(o), "detail": d} for n, o, d in checks],
    }, indent=1))
    print(f"wrote {out}")
    return 0 if passed else 1


def apply_config(args: argparse.Namespace) -> argparse.Namespace:
    """Overlay a YAML config, with explicit CLI flags still winning.

    Only fills values the user did not pass, so a config can set the defaults for a run
    without silently overriding a flag typed at launch. The gate thresholds are applied to
    the module globals so `compare` uses the same frozen numbers the run was planned with.
    """
    if not getattr(args, "config", None):
        return args
    import yaml
    cfg = yaml.safe_load(Path(args.config).read_text()) or {}
    passed = {a.split("=")[0].lstrip("-").replace("-", "_") for a in sys.argv[1:]}
    mapping = {
        "model": "subject_model", "dataset": "dataset", "n_snippets": "n_snippets",
        "reads_per_snippet": "reads_per_snippet", "max_new_read": "max_new_read",
        "seed": "seed", "sglang_url": "sglang_url",
    }
    for dest, key in mapping.items():
        if key in cfg and dest not in passed and hasattr(args, dest):
            setattr(args, dest, cfg[key])
    g = cfg.get("gates") or {}
    globals().update({
        "GATE_COS_MEDIAN": g.get("cos_median", GATE_COS_MEDIAN),
        "GATE_COS_DELTA": g.get("cos_delta_vs_control", GATE_COS_DELTA),
        "GATE_NORM_RATIO": g.get("norm_ratio", GATE_NORM_RATIO),
        "GATE_CJK_FRAC": g.get("cjk_frac", GATE_CJK_FRAC),
        "GATE_OWN_MINUS_FOREIGN": g.get("own_minus_foreign", GATE_OWN_MINUS_FOREIGN),
    })
    if "layer_index" in cfg:
        globals()["LAYER_INDEX"] = int(cfg["layer_index"])
    args.config_used = str(args.config)
    return args


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=None, help="YAML config; CLI flags override it")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="read one subject model")
    r.add_argument("--config", default=None, help="YAML config; CLI flags override it")
    r.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    r.add_argument("--tag", required=True, help="short name for the output files")
    r.add_argument("--dataset", default="humaneval", choices=["humaneval", "cruxeval"])
    r.add_argument("--n-snippets", type=int, default=200)
    r.add_argument("--reads-per-snippet", type=int, default=5)
    r.add_argument("--device", default="cuda")
    r.add_argument("--ar-device", default="cuda",
                   help="'cpu' when the AV server and subject model share a card — AV 19.2 GB "
                        "+ subject 17 GB + AR 11 GB does not fit in 48 GB (deception_1gpu.sh "
                        "hit exactly this).")
    r.add_argument("--sglang-url", default="http://localhost:30011")
    r.add_argument("--max-new-read", type=int, default=MAX_NEW_READ)
    r.add_argument("--seed", type=int, default=SEED)
    r.add_argument("--out-dir", default=None,
                   help="override the output directory (used to smoke-test resume without "
                        "touching a real run's jsonl)")
    r.add_argument("--max-hours", type=float, default=6.0,
                   help="wall budget per arm; stops cleanly and the next run resumes")
    r.set_defaults(fn=run)

    c = sub.add_parser("compare", help="emit the gate table")
    c.add_argument("--config", default=None, help="YAML config; CLI flags override it")
    c.add_argument("--a", required=True, help="tag of the new subject")
    c.add_argument("--b", required=True, help="tag of the control (the NLA's own host)")
    c.add_argument("--out-dir", default=None)
    c.set_defaults(fn=compare)

    a = apply_config(p.parse_args())
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
