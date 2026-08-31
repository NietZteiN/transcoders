"""Producer for enriched.json — the file the results-browser artifact reads.

Until now this file had NO producer: it was built ad hoc, and its `question` field was
malformed for slice_prediction cases (it contained trailing code plus the convention text
instead of the question). Prompts are now rebuilt from `overnight_capture.build_user()`, the
single source of truth. Do NOT use `answer_reads.build_user` — it is a stale copy whose slice
wording differs and which omits SLICE_CONVENTION.

Adds, per case:
  code, question           reconstructed from the task definition
  reads[].anchor           char spans + parent word + context (for the token↔read linking)
  scores{}                 the stable sort contract the artifact uses:
                             faith_mean / faith_n     available now
                             align_* / source         filled when the judge run lands
  first_error              joined from the N4 oracle when available

Run (CPU only):
    python -m src.build_enriched --out data/nla/artifact/<date>/enriched.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from capture_core import align_reply, load_captures          # noqa: E402
from overnight_capture import build_tasks, build_user, task_key  # noqa: E402

PROJ = _HERE.parent.parent
WORD = re.compile(r"[A-Za-z0-9_$]")


def parent_word(full: str, s: int, e: int) -> str:
    """The whole identifier/number a token is a piece of ('` (_`' -> '_lastNSecs')."""
    idxs = [i for i in range(s, min(e, len(full))) if WORD.match(full[i])]
    if not idxs:
        return ""
    a = b = idxs[-1]
    while a > 0 and WORD.match(full[a - 1]):
        a -= 1
    while b + 1 < len(full) and WORD.match(full[b + 1]):
        b += 1
    return full[a:b + 1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--captures", default=str(PROJ / "data/nla/overnight/2026-08-04/captures.jsonl"))
    ap.add_argument("--first-errors", default=str(PROJ / "data/nla/n4/2026-08-06/first_errors.jsonl"))
    ap.add_argument("--alignment", default=None, help="N7 alignment parquet/jsonl, when it exists")
    ap.add_argument("--out", default=str(PROJ / "data/nla/overnight/2026-08-04/enriched.json"))
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

    caps = load_captures(args.captures)
    tasks = {task_key(t): t for t in build_tasks()[0]}

    ferr: dict[str, dict] = {}
    if Path(args.first_errors).exists():
        for line in open(args.first_errors):
            r = json.loads(line)
            if r.get("first_error"):
                ferr[r["task_key"]] = r["first_error"]

    align: dict[str, dict] = {}
    if args.alignment and Path(args.alignment).exists():
        for line in open(args.alignment):
            r = json.loads(line)
            align[r["task_key"]] = r

    out = []
    for k, cap in caps.items():
        t = tasks.get(k)
        if t is None:
            continue
        user = build_user(t)                     # SINGLE SOURCE OF TRUTH for the prompt
        code = t.get("code", "")
        question = user.split(code, 1)[-1].strip() if code and code in user else user
        reply = cap.get("model_reply") or ""
        a = align_reply(tok, user, reply)

        reads = []
        faith = []
        for rd in cap.get("reads", []):
            p = rd["position"]
            anchor = None
            if 0 <= p < len(a.offsets):
                s, e = a.offsets[p]
                if e > s:
                    anchor = {"s": s, "e": e, "in_reply": s >= len(a.templ),
                              "rs": max(0, s - len(a.templ)), "re": max(0, e - len(a.templ)),
                              "tok": a.full[s:e], "word": parent_word(a.full, s, e),
                              "before": a.full[max(0, s - 46):s].replace("\n", " "),
                              "after": a.full[e:e + 46].replace("\n", " ")}
            faith.append(rd["rt_cos"])
            al = (align.get(k, {}).get("reads", {}) or {}).get(str(p), {})
            reads.append({**rd, "anchor": anchor,
                          "scores": {"faith": rd["rt_cos"],
                                     "align": al.get("align"), "align_cat": al.get("category")}})

        al_case = align.get(k, {})
        out.append({
            "task_key": k, "kind": cap["kind"], "dataset": cap["dataset"], "tier": cap["tier"],
            "snippet_id": cap["snippet_id"], "language": cap["language"],
            "call": cap.get("call"), "target": cap.get("target"),
            "model_answer": cap.get("model_answer"), "truth": cap.get("truth"),
            "correct": cap.get("correct"), "truncated": cap.get("truncated"),
            "reply_tokens": cap.get("reply_tokens"), "model_reply": reply,
            "code": code, "question": question,
            "first_error": ferr.get(k),
            "scores": {
                "faith_mean": round(sum(faith) / len(faith), 4) if faith else None,
                "faith_n": len(faith),
                "align_mean": al_case.get("align_mean"),
                "align_spec": al_case.get("align_spec"),
                "align_contra": al_case.get("align_contra"),
                "align_z": al_case.get("align_z"),
                "align_n": al_case.get("align_n", 0),
                "source": "judge_v1" if al_case else "faithfulness_v1",
            },
            "reads": reads,
        })

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, separators=(",", ":")))
    n_anch = sum(1 for c in out for r in c["reads"] if r["anchor"])
    n_fe = sum(1 for c in out if c["first_error"])
    print(f"[enriched] {len(out)} cases · {sum(len(c['reads']) for c in out)} reads · "
          f"{n_anch} anchored · {n_fe} with first_error · align={'yes' if align else 'no'}")
    print(f"           wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
