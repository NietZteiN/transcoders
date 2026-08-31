"""N7b — score CoT<->NLA agreement with a blinded, cross-family judge.

BLINDING: this module opens `pack.jsonl` and NOTHING else. It never imports the corpus, the grades,
or the first-error file. The `_assert_blind` check fails the run if the pack carries any column
beyond {item_id, read, window} — so unblinding requires editing *this* file AND `sanitize_windows.py`,
which is the point.

JUDGE: `meta-llama/Llama-3.1-8B-Instruct` — a DIFFERENT family from the Qwen2.5-7B subject and from
the Qwen-based NLA, so the judge cannot be scoring its own idiolect. `--judge` swaps in
`microsoft/Phi-3.5-mini-instruct` for the kappa cross-check.

NO CONSTRAINED DECODING. The 2026-08-06 N4b run proved that sglang's constrained-JSON path
SEGFAULTS the scheduler (exit -11) on this build. Verdicts are parsed leniently instead, and
anything unparseable is recorded as `null` and counted rather than silently dropped.

Prompt design matters more than the model here, so it is explicit about three things the judge would
otherwise get wrong:
  * correctness of the reasoning is IRRELEVANT — a confidently wrong step described accurately is a 3;
  * the read is auto-generated from an internal activation, so its *specifics* (language, library
    names) are known to confabulate while its *themes* are reliable — score the theme;
  * a read that merely shares the domain ("some Python code") is a 1, not a 3. Without this the
    scale collapses upward and the shuffled null stops being a floor.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import requests

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

PROJ = _HERE.parent.parent
ALLOWLIST = {"item_id", "read", "window"}
SEED = 20260724

SYSTEM = """You rate how well a machine-generated DESCRIPTION matches a STEP of reasoning text.

The description was produced automatically by decoding a snapshot of a language model's internal
state while it was writing that reasoning. Your job is to judge SEMANTIC MATCH only.

Three rules that override your instincts:
1. IGNORE CORRECTNESS. The reasoning may be wrong, and the description may describe wrong
   reasoning. A confidently wrong step, described accurately, is a perfect match.
2. JUDGE THEMES, NOT SPECIFICS. These descriptions reliably capture what KIND of operation is
   happening, but routinely confabulate specifics -- they may name the wrong programming language,
   invent library names, or misstate numbers. Do not penalise those. Penalise a wrong THEME.
3. SHARED DOMAIN IS NOT A MATCH. "This is some code doing a computation" is true of nearly every
   step here. That is a 1, not a 3.

Scale:
  0 = unrelated. The description is about a different kind of activity entirely.
  1 = same broad domain only (both are "code reasoning") with no shared specific content.
  2 = same kind of operation (e.g. both are about iterating a loop and accumulating, or both about
      checking a base case), but not clearly THIS step.
  3 = describes THIS step: the same operation on the same sort of quantity at the same stage.

Reply with JSON only:
{"score": <0-3>, "category": "<a few words naming the operation you matched on>", "evidence": "<one short sentence>"}"""

USER = """STEP (reasoning text):
\"\"\"{window}\"\"\"

DESCRIPTION (decoded from the internal state at this step):
\"\"\"{read}\"\"\"

JSON:"""

_JSON = re.compile(r"\{.*?\}", re.S)
_SCORE = re.compile(r'"?score"?\s*[:=]\s*([0-3])')


def parse_verdict(text: str) -> dict | None:
    """Lenient parse. Constrained decoding is unavailable (segfaults), so recover what we can."""
    m = _JSON.search(text or "")
    if m:
        try:
            d = json.loads(m.group(0))
            if isinstance(d.get("score"), (int, float)) and 0 <= d["score"] <= 3:
                return {"score": int(d["score"]), "category": str(d.get("category", ""))[:80],
                        "evidence": str(d.get("evidence", ""))[:200]}
        except Exception:
            pass
    m = _SCORE.search(text or "")           # fall back to the number alone
    if m:
        return {"score": int(m.group(1)), "category": "", "evidence": ""}
    return None


def _assert_blind(rows: list[dict]) -> None:
    for r in rows[:200]:
        extra = set(r) - ALLOWLIST
        if extra:
            raise SystemExit(f"BLINDING VIOLATION: pack row carries {extra}; refusing to judge.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default=str(PROJ / "data/nla/n7/2026-08-07/pack.jsonl"))
    ap.add_argument("--out", default=None, help="default <pack dir>/verdicts_<tag>.jsonl")
    ap.add_argument("--url", default="http://localhost:30010")
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--tag", default="llama")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--concurrency", type=int, default=8)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.pack)]
    _assert_blind(rows)
    if args.limit:
        rows = rows[: args.limit]
    out_p = Path(args.out) if args.out else Path(args.pack).parent / f"verdicts_{args.tag}.jsonl"

    done = set()
    if out_p.exists():
        for l in open(out_p):
            try:
                done.add(json.loads(l)["item_id"])
            except Exception:
                pass
    todo = [r for r in rows if r["item_id"] not in done]
    print(f"[N7b] {len(rows)} items · {len(done)} already judged · {len(todo)} to go", flush=True)
    if not todo:
        return 0

    sess = requests.Session()

    def judge(r: dict) -> dict:
        body = {"model": args.model, "temperature": 0.0, "max_tokens": 160, "seed": SEED,
                "messages": [{"role": "system", "content": SYSTEM},
                             {"role": "user", "content": USER.format(window=r["window"],
                                                                     read=r["read"])}]}
        for attempt in range(3):
            try:
                resp = sess.post(f"{args.url}/v1/chat/completions", json=body, timeout=120)
                txt = resp.json()["choices"][0]["message"]["content"]
                v = parse_verdict(txt)
                return {"item_id": r["item_id"], **(v or {"score": None}),
                        "raw": None if v else (txt or "")[:200]}
            except Exception as e:
                if attempt == 2:
                    return {"item_id": r["item_id"], "score": None, "error": repr(e)[:160]}
                time.sleep(2 * (attempt + 1))
        return {"item_id": r["item_id"], "score": None}

    t0, n_bad = time.time(), 0
    with open(out_p, "a") as f, ThreadPoolExecutor(args.concurrency) as pool:
        for i, v in enumerate(pool.map(judge, todo)):
            n_bad += v.get("score") is None
            f.write(json.dumps(v) + "\n")
            f.flush()
            if (i + 1) % 250 == 0:
                rate = (i + 1) / (time.time() - t0)
                print(f"[N7b] {i+1}/{len(todo)} · {rate:.1f}/s · unparsed {n_bad} "
                      f"· ETA {(len(todo)-i-1)/max(rate,1e-9)/60:.0f} min", flush=True)

    print(f"[N7b] done · {len(todo)} judged · {n_bad} unparsed ({n_bad/max(len(todo),1):.1%})", flush=True)
    (out_p.parent / f"judge_manifest_{args.tag}.json").write_text(json.dumps({
        "experiment": "n7b_judge_align", "model": args.model, "seed": SEED, "argv": sys.argv,
        "temperature": 0.0, "concurrency": args.concurrency,
        "constrained_decoding": False, "reason": "sglang constrained JSON segfaults this build (N4b)",
        "n_items": len(rows), "n_unparsed": n_bad,
        "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
