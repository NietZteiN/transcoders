"""N13 Stage 3 — read instability: does a vector HAVE one sentence, or several?

THE INSTRUMENT. `rt_cos` asks whether *one* sentence carried the vector. Instability asks whether
the vector *has* one sentence. Verbalize the SAME activation K times at temperature > 0: a vector
holding one reading should come back saying the same thing every time; a vector holding two
incompatible readings at once should not. That construction is the only one in this programme
whose shape matches the phenomenon being hunted — a model that is genuinely torn between
`a + b^2` and `(a+b)^2`.

Agreement is measured in **AR space**, not string space. `NLACritic.reconstruct` maps each
reading back to a vector, and the mean pairwise cosine between those K vectors asks "do these
sentences MEAN the same thing" in the space the activations live in. Two paraphrases of one
reading score high; two different readings score low, however similar their wording. A judge-free
lexical Jaccard runs alongside as a fallback that depends on no learned component — after the N7
LLM judge failed at kappa = 0.049, no measure here rests on a model's opinion.

THREE STRATA, AND THE DISTINCTION THAT MAKES THE RESULT MEAN ANYTHING:

  fn_orig      function names in clean code. The most describable class in the corpus
               (rt_cos 0.934). Unambiguous content -> should be STABLE. **This is the
               temperature-only floor.** If it is as unstable as everything else, the measure is
               reading decoder stochasticity and Stage 4 does not run.
  adversarial  decoy identifiers in L1b. Engineered ambiguity: a name asserting one algorithm
               over code implementing another. **The torn candidate.**
  l1_neutral   gibberish renames (`var_bd90`). Reads WORST in the corpus (0.847).

The l1_neutral stratum is the control the plan did not name, and it guards the interpretation
rather than the measurement. Instability can arise two ways: from **superposition** (two readings
present) or from **emptiness** (nothing present, so the decoder free-runs). Those are opposite
internal states that would produce the same instability number. If adversarial and l1_neutral are
equally unstable, this is an emptiness meter and the "torn" reading is unsupported; the claim
needs adversarial (rich, conflicting) to separate from l1_neutral (poor, vacant).

**TWO CONFOUNDS IN THE FLOOR, MEASURED FROM THE BANKED CORPUS AND STATED UP FRONT:**

  1. `fn_orig` is not L0-only. Its 58 reads span L0(12) L1(29) L2(13) L3(4) — the original
     function name survives renaming and flattening — while `adversarial` exists only at
     L1b/L3. So class is confounded with tier, and a floor-vs-torn gap cannot be cleanly
     attributed to ambiguity.
  2. The token TEXT differs in kind. `fn_orig` tokens are whole meaningful words (` fib`,
     ` generate`, ` make`, ` separate`); `adversarial` and `l1_neutral` tokens are bracket-glued
     identifier fragments (` (_`, `(a`, `[f`, `_list`). Given that rt_cos correlates +0.277 with
     read length, a floor-vs-torn gap is partly a token-informativeness gap.

Both confounds point the same way: **the sharpest contrast here is adversarial vs l1_neutral**,
not floor vs torn. Those two are matched in token form (identifier fragments) and both carry
renamed identifiers, differing in exactly one thing — whether the new name asserts a confident
wrong meaning. `fn_orig` is retained as the absolute stability ceiling and as the temperature
floor the plan requires, but the interpretive weight sits on the matched pair. Token text and
length are recorded per position so Stage 4 can control for them.

Reads are STRICTLY SEQUENTIAL. Batching the AV path was measured to break its determinism —
6.3x faster, only 15% byte-identical — so nothing here is batched, unlike Stage 1.

TWO PHASES, for a reason that is purely about memory. Phase A extracts every needed activation
with the subject model and then frees it; phase B loads the AR critic onto the freed GPU. AR is
the slowest component (a 21-layer 7B forward per reading) and running it on CPU clocked the stage
at 4.3 reads/min = 7.7 h. It cannot share the card with both the 19.2 GB AV server and the 17 GB
subject model, but it fits comfortably once the subject model is gone.

Env `nla-mi`, one GPU (AV server + AR critic; subject model freed after phase A).
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_NLA_ROOT = _HERE.parent
_PROJ = _NLA_ROOT.parent
sys.path.insert(0, str(_NLA_ROOT / "vendor" / "nla-repo"))
sys.path.insert(0, str(_HERE))

from capture_core import MAX_NEW_READ, align_reply  # noqa: E402
from overnight_capture import (  # noqa: E402
    LAYER_INDEX, SEED, TARGET_MODEL, build_tasks, build_user, task_key,
)

ENRICHED = _PROJ / "data/nla/overnight/2026-08-04/enriched.json"
CAPTURES = _PROJ / "data/nla/overnight/2026-08-04/captures.jsonl"

# floor first: the stratum that decides whether Stage 4 runs at all
FLOOR_CLS = "fn_orig"
TORN_CLS = "adversarial"
VACANT_CLS = "l1_neutral"
# quotas; fn_orig has only 58 in the whole corpus so it is taken exhaustively
QUOTA = {FLOOR_CLS: 58, TORN_CLS: 110, VACANT_CLS: 90,
         "orig": 60, "cot": 50, "answer": 32}

STOP = set("the a an is are was were be been of to in on at for and or but with as by from "
           "this that these those it its which what how then so if we you i they there here "
           "code function value number list string variable name step".split())


def content_words(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]{3,}", s.lower()) if w not in STOP}


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if (a or b) else 1.0


def pairwise_stats(vecs: list[np.ndarray], texts: list[str]) -> dict:
    """AR-space agreement among K readings of ONE vector, plus a judge-free lexical echo."""
    cos, jac = [], []
    for i, j in combinations(range(len(vecs)), 2):
        a, b = vecs[i], vecs[j]
        cos.append(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)))
        jac.append(jaccard(content_words(texts[i]), content_words(texts[j])))
    n_exact = len({t.strip() for t in texts})
    return {
        "ar_cos_mean": round(float(np.mean(cos)), 4) if cos else None,
        "ar_cos_min": round(float(np.min(cos)), 4) if cos else None,
        "ar_cos_sd": round(float(np.std(cos)), 4) if cos else None,
        "jaccard_mean": round(float(np.mean(jac)), 4) if jac else None,
        "n_distinct_texts": n_exact,
    }


def build_frame(enriched: list[dict], k_entropy: dict[str, float] | None,
                quota: dict[str, int], seed: int) -> list[dict]:
    """Stratified position sample: class quotas, spread over tier and Stage-1 entropy."""
    rng = random.Random(seed)
    pool: dict[str, list[dict]] = defaultdict(list)
    for r in enriched:
        for x in r["reads"]:
            cls = x.get("cls")
            if cls not in quota:
                continue
            pool[cls].append({
                "case": r["task_key"], "position": int(x["position"]), "cls": cls,
                "tier": r.get("tier"), "kind": r.get("kind"),
                "rt_cos_banked": x.get("rt_cos"), "read_banked": x.get("read"),
                "in_reply": (x.get("anchor") or {}).get("in_reply"),
                "correct": r.get("correct"),
                "item_entropy": (k_entropy or {}).get(r["task_key"]),
            })
    frame = []
    for cls, want in quota.items():
        cand = pool.get(cls, [])
        if len(cand) <= want:
            frame.extend(cand)
            continue
        # spread across tier x entropy tercile so a class is not sampled from one corner
        buckets: dict[tuple, list[dict]] = defaultdict(list)
        for c in cand:
            e = c["item_entropy"]
            tb = "na" if e is None else ("lo" if e < 0.34 else "mid" if e < 0.67 else "hi")
            buckets[(str(c["tier"]), tb)].append(c)   # tier is None on slice items
        keys = sorted(buckets)
        for kk in keys:
            rng.shuffle(buckets[kk])
        # round-robin over buckets so every tier x entropy cell contributes before any repeats
        picked: list[dict] = []
        i = 0
        while len(picked) < want and any(buckets[kk] for kk in keys):
            kk = keys[i % len(keys)]
            i += 1
            if buckets[kk]:
                picked.append(buckets[kk].pop())
        frame.extend(picked)
    # Shuffle so that a partial run — wall-guard cutoff or --limit — is still balanced across
    # classes. The frame is built class-by-class, so unshuffled it would yield 58 fn_orig and
    # nothing else, and the floor gate needs every class present to mean anything.
    rng.shuffle(frame)
    return frame


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--enriched", default=str(ENRICHED))
    ap.add_argument("--captures", default=str(CAPTURES))
    ap.add_argument("--entropy", default=str(_PROJ / "data/nla/n13/answer_entropy.jsonl"))
    ap.add_argument("--out", default=str(_PROJ / "data/nla/n13/read_instability.jsonl"))
    ap.add_argument("--sglang-url", default="http://localhost:30003")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ar-device", default="cuda",
                    help="AR fits on GPU because the subject model is freed after phase A")
    ap.add_argument("--max-hours", type=float, default=6.0)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    import torch
    from extract import ActivationExtractor
    from nla_inference import NLAClient, NLACritic
    from capture_core import JsonlSink, WallGuard

    torch.manual_seed(SEED)

    enriched = json.load(open(args.enriched))
    caps = {r["task_key"]: r for r in map(json.loads, open(args.captures)) if "task_key" in r}
    tasks = {task_key(t): t for t in build_tasks()[0]}

    ent = {}
    ep = Path(args.entropy)
    if ep.exists():
        for l in open(ep):
            r = json.loads(l)
            if "error" not in r:
                ent[r["task_key"]] = r["entropy"]
    print(f"[n13s3] Stage-1 entropy available for {len(ent)} items", flush=True)

    frame = build_frame(enriched, ent, QUOTA, SEED)
    if args.limit:
        frame = frame[: args.limit]
    by_cls: dict[str, int] = defaultdict(int)
    for f in frame:
        by_cls[f["cls"]] += 1
    print(f"[n13s3] frame = {len(frame)} positions · {dict(by_cls)}", flush=True)

    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    sink = JsonlSink(out_p, key_field="uid")
    done = sink.done_keys()
    todo = [f for f in frame if f"{f['case']}@{f['position']}" not in done]
    print(f"[n13s3] {len(todo)} to go ({len(done)} done)", flush=True)
    if not todo:
        return 0

    wall = WallGuard(args.max_hours)

    # ── phase A: extract every needed vector, then FREE the subject model ──────────────
    # The AR critic is the bottleneck (a 21-layer 7B forward per reading) and on CPU it ran the
    # whole stage at 4.3 reads/min -> 7.7 h. It only fits on GPU beside the 19.2 GB AV server if
    # the 17 GB subject model is gone, so all activations are extracted up front and cached.
    # 400 positions x 3584 floats is ~5.7 MB, so holding them costs nothing.
    grouped: dict[str, list[dict]] = defaultdict(list)
    for f in todo:
        grouped[f["case"]].append(f)

    ex = ActivationExtractor(TARGET_MODEL, LAYER_INDEX, device=args.device)
    cache: dict[str, np.ndarray] = {}
    tokmeta: dict[str, tuple[str, int, bool]] = {}
    for ci, (case, items) in enumerate(sorted(grouped.items())):
        t = tasks.get(case)
        if t is None or case not in caps:
            continue
        user = build_user(t)
        reply = caps[case].get("model_reply") or ""
        positions = sorted({f["position"] for f in items})
        try:
            res = ex.extract_chat(user, reply, positions=positions, text_id=case)
        except Exception as e:
            print(f"[n13s3] ERROR extract {case}: {e!r}", flush=True)
            continue
        a = align_reply(ex.tokenizer, user, reply)
        for p, v in zip(res.positions, res.activations):
            cache[f"{case}@{p}"] = np.asarray(v, dtype=np.float32)
            s, e_ = a.offsets[p] if p < len(a.offsets) else (0, 0)
            txt = a.full[s:e_]
            tokmeta[f"{case}@{p}"] = (txt[:24], len(txt.strip()), txt.strip().isalpha())
        if (ci + 1) % 50 == 0:
            print(f"[n13s3] extracted {ci+1}/{len(grouped)} cases", flush=True)
    ex.close()          # drop the forward hook (it holds a ref to the layer, and _captured
    del ex              # holds a GPU tensor) before releasing the model
    torch.cuda.empty_cache()
    free_gb = torch.cuda.memory_allocated() / 1e9
    print(f"[n13s3] cached {len(cache)} vectors; subject model freed "
          f"(torch still holds {free_gb:.1f} GB)", flush=True)

    # ── phase B: verbalize each cached vector K times ─────────────────────────────────
    av = NLAClient(_NLA_ROOT / "data" / "checkpoints" / "av", sglang_url=args.sglang_url)
    ar = NLACritic(_NLA_ROOT / "data" / "checkpoints" / "ar", device=args.ar_device)

    # If the AV server dies mid-run every remaining position raises, and without this the script
    # would grind through the whole frame writing error rows and burn the wall budget producing
    # nothing. Aborting on a run of consecutive failures turns a silent 6-hour waste into a
    # diagnosis, and the sink is resumable so a restart loses no completed work.
    MAX_CONSECUTIVE_FAILURES = 10
    consec_fail = 0

    t0, n_pos, n_reads = time.time(), 0, 0
    for ci, (case, items) in enumerate(sorted(grouped.items())):
        if wall.expired():
            print("[n13s3] wall budget reached — stopping cleanly", flush=True)
            break
        if consec_fail >= MAX_CONSECUTIVE_FAILURES:
            print(f"[n13s3] ABORT: {consec_fail} consecutive read failures — the AV server is "
                  f"probably down. See the server log; rerun to resume.", file=sys.stderr)
            break
        for f in items:
            uid = f"{f['case']}@{f['position']}"
            v = cache.get(uid)
            if v is None:
                continue
            texts, vecs, rts = [], [], []
            try:
                for _ in range(args.k):
                    # sequential by design — batching breaks AV determinism
                    txt = av.generate(v, temperature=args.temperature,
                                      max_new_tokens=MAX_NEW_READ)
                    texts.append(txt)
                    # ONE reconstruct per reading: NLACritic.score() re-runs reconstruct
                    # internally, so calling both doubled the cost of the slowest component.
                    # score()'s cosine is invariant to its L2 rescaling, so cos(pred, v) here
                    # is the same rt_cos the banked pipeline recorded.
                    pred = ar.reconstruct(txt).numpy()
                    vecs.append(pred)
                    rts.append(float(pred @ v / (np.linalg.norm(pred) * np.linalg.norm(v) + 1e-9)))
                    n_reads += 1
            except Exception as e:
                sink.append({"uid": uid, "error": repr(e)[:200]})
                consec_fail += 1
                print(f"[n13s3] ERROR read {uid} (consecutive {consec_fail}): {e!r}", flush=True)
                if consec_fail >= MAX_CONSECUTIVE_FAILURES:
                    break
                continue
            consec_fail = 0
            tk, tlen, talpha = tokmeta.get(uid, ("", 0, False))
            sink.append({
                "uid": uid, **{k: f[k] for k in
                    ("case", "position", "cls", "tier", "kind", "rt_cos_banked",
                     "in_reply", "correct", "item_entropy")},
                # token text/length recorded so Stage 4 can control for the informativeness
                # confound: fn_orig tokens are whole words, adversarial ones bracket fragments
                "token": tk, "token_len": tlen, "token_alpha": talpha,
                "k": args.k, "temperature": args.temperature,
                "act_norm": round(float(np.linalg.norm(v)), 4),
                "rt_cos_sampled_mean": round(float(np.mean(rts)), 4),
                "rt_cos_sampled_sd": round(float(np.std(rts)), 4),
                "read_lens": [len(x) for x in texts],
                "reads": texts,
                **pairwise_stats(vecs, texts),
            })
            n_pos += 1

        if (ci + 1) % 10 == 0 or ci + 1 == len(grouped):
            el = time.time() - t0
            rate = n_pos / max(el, 1e-9)
            print(f"[n13s3] case {ci+1}/{len(grouped)} · {n_pos}/{len(todo)} positions · "
                  f"{n_reads} reads · {n_reads/max(el,1e-9)*60:.1f} reads/min "
                  f"· ETA {(len(todo)-n_pos)/max(rate,1e-9)/60:.0f} min", flush=True)

    sink.close()
    rows = [json.loads(l) for l in open(out_p) if l.strip()]
    ok = [r for r in rows if "error" not in r]

    # ── the floor gate ────────────────────────────────────────────────────
    def mean_of(cls: str, key: str = "ar_cos_mean"):
        vals = [r[key] for r in ok if r.get("cls") == cls and r.get(key) is not None]
        return (round(float(np.mean(vals)), 4), len(vals)) if vals else (None, 0)

    floor, n_floor = mean_of(FLOOR_CLS)
    torn, n_torn = mean_of(TORN_CLS)
    vacant, n_vacant = mean_of(VACANT_CLS)
    gate_pass = (floor is not None and torn is not None and floor > torn)
    manifest = {
        "experiment": "n13_stage3_read_instability", "seed": SEED, "argv": sys.argv,
        "model": TARGET_MODEL, "layer": LAYER_INDEX, "k": args.k,
        "temperature": args.temperature, "max_new_read": MAX_NEW_READ,
        "n_positions": len(ok), "n_errors": len(rows) - len(ok), "n_reads": n_reads,
        "ar_cos_by_class": {c: mean_of(c)[0] for c in QUOTA},
        "n_by_class": {c: mean_of(c)[1] for c in QUOTA},
        "floor_gate": {
            "floor_class": FLOOR_CLS, "floor_ar_cos": floor, "n": n_floor,
            "torn_class": TORN_CLS, "torn_ar_cos": torn, "n_torn": n_torn,
            "vacant_class": VACANT_CLS, "vacant_ar_cos": vacant, "n_vacant": n_vacant,
            "pass": bool(gate_pass),
            "note": ("Floor must be MORE stable (higher AR-space agreement) than the torn "
                     "candidate, or the measure is reading decoder stochasticity and Stage 4 "
                     "does not run. Separately: if torn ~ vacant, instability is an emptiness "
                     "meter, not a superposition meter."),
        },
        "elapsed_hours": round(wall.elapsed_h(), 3),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
    }
    (out_p.parent / "read_instability_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest["floor_gate"], indent=2), flush=True)
    print(f"[n13s3] done · {len(ok)} positions · {n_reads} reads", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
