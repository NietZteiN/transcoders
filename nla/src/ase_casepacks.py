"""H-R1a stage 1 — build execution-validated T/F case packs for HumanEval-X Java using the ASE-2026
artifact's OWN code, so the replication cannot drift from their protocol.

Pre-registered in log/nla-harness/2026-09-14_ase-replication-prereg.md.

WHY THEIR CODE AND NOT A REIMPLEMENTATION. The whole point is to test whether their -36.3-point
identifier-renaming drop reproduces on a model we are allowed to run, against my +0.0117 on
Python/JS output prediction. If I rebuilt the case construction myself, a null would be
uninterpretable -- it could be my case packs. So `counterfactual_eval.build_case_pack` (which
execution-validates every case with a real javac+java run) and `build_counterfactual_instruction`
are imported from the artifact unmodified. Only the model runner is ours.

CPU ONLY. Every case is validated by executing Java, so this stage needs a JDK (17.0.17 present)
and no GPU. Packs are cached, so a re-run is free and stage 2 never rebuilds them.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_PROJ = Path(__file__).resolve().parents[2]
TAG = "[PACK]"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifact", default="/scratch/juno/jvl210002/ase2026/LLM-Attention-Fixation_submission")
    ap.add_argument("--dataset", default="humaneval", choices=("humaneval", "cruxeval"))
    ap.add_argument("--lang-dir", default="Source/Humaneval/java")
    ap.add_argument("--cache-dir", default="/scratch/juno/jvl210002/ase2026/cache_cases")
    ap.add_argument("--out", default="/scratch/juno/jvl210002/ase2026/packs_humaneval_java.jsonl")
    ap.add_argument("--min-cases", type=int, default=8)
    ap.add_argument("--target-cases", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-hours", type=float, default=3.0)
    args = ap.parse_args()

    R = Path(args.artifact)
    if not R.exists():
        print(f"{TAG} REFUSED: artifact not found at {R}"); return 2
    sys.path.insert(0, str(R))
    import counterfactual_eval as ce            # noqa: E402  (their code, unmodified)

    # --lang-dir may be absolute (the RENAMED corpus lives outside the artifact tree)
    ld = Path(args.lang_dir)
    src_dir = ld if ld.is_absolute() else (R / ld)
    src = sorted(src_dir.glob("*.java"))
    if args.limit:
        src = src[:args.limit]
    print(f"{TAG} {len(src)} snippets from {src_dir} · dataset={args.dataset}", flush=True)

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    if out.exists():                              # resumable: a wall-clock stop costs nothing
        done = {json.loads(l)["snippet"] for l in open(out) if l.strip()}
        print(f"{TAG} resuming; {len(done)} packs already written", flush=True)

    t0 = time.time()
    n_ok = n_fail = 0
    sink = open(out, "a")
    for i, p in enumerate(src, 1):
        cls = p.stem
        if cls in done:
            continue
        try:
            pack = ce.build_case_pack(java_code=p.read_text(), class_name=cls, dataset=args.dataset,
                                      snippet=cls, min_cases=args.min_cases,
                                      target_cases=args.target_cases, cache_dir=args.cache_dir)
        except Exception as e:                    # a snippet that will not build is recorded, never silently dropped
            n_fail += 1
            sink.write(json.dumps({"snippet": cls, "error": f"{type(e).__name__}: {e}"}) + "\n"); sink.flush()
            print(f"{TAG} {i}/{len(src)} {cls} FAILED {type(e).__name__}: {e}", flush=True)
            continue
        cases = pack.get("cases", [])
        n_true = sum(1 for c in cases if c.get("expected_bool"))
        rec = {"snippet": cls, "n_cases": len(cases), "n_true": n_true,
               "n_false": len(cases) - n_true, "java_path": str(p), "pack": pack}
        sink.write(json.dumps(rec) + "\n"); sink.flush()
        n_ok += 1
        print(f"{TAG} {i}/{len(src)} {cls} cases={len(cases)} (T {n_true}/F {len(cases)-n_true}) · "
              f"{(time.time()-t0)/60:.1f} min", flush=True)
        if time.time() - t0 > args.max_hours * 3600:
            print(f"{TAG} wall-clock stop after {i}; re-run to resume", flush=True)
            break
    sink.close()

    rows = [json.loads(l) for l in open(out) if l.strip()]
    good = [r for r in rows if "pack" in r]
    tot = sum(r["n_cases"] for r in good)
    print(f"{TAG} DONE built {n_ok} this pass · {len(good)} packs total · {n_fail} failures this pass")
    if good:
        print(f"{TAG} {tot} validated cases · {tot/len(good):.2f} per snippet "
              f"(their HumanEval-X figure: 11.72) · T/F balance "
              f"{sum(r['n_true'] for r in good)}/{sum(r['n_false'] for r in good)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
