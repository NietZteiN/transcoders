"""Render steering examples: the text edit, and what it did to the model's reasoning.

WHY THIS EXISTS. The B4 steering run stored answers and reply *lengths* but not the replies, so
"what did steering actually do to the text" was unanswerable from the banked results. This renders
a small re-run (`steer_run.py --save-replies`) over the items that flipped.

WHAT A STEERING EDIT IS HERE. V1 builds its direction from a pair of English sentences:

    delta = AR("<true description>") - AR("<decoy description>")

and adds `alpha * ||h|| * delta/||delta||` to the residual stream at one position. So the "edit"
is literally the difference between two descriptions, and both are shown — that pair IS the
intervention, not an illustration of it.

alpha = 0 is the unsteered baseline. The write hook was verified byte-identical at alpha = 0, so
the alpha-0 and alpha-1 rows differ in exactly one thing.

HONESTY REQUIREMENT. Steering flips answers in both directions — on the full run V1 recovered 42
wrong answers and broke 18 right ones. Showing only recoveries would misrepresent an intervention
that failed its pre-registered gate. Damage cases are rendered with equal prominence and the
counts are stated.

Env: any. CPU, seconds.
"""
from __future__ import annotations

import argparse
import html
import json
import sys
from collections import defaultdict
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent.parent


def esc(s) -> str:
    return html.escape(str(s if s is not None else ""), quote=True)


def clip(s: str, n: int) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[:n].rsplit(" ", 1)[0] + "…"


def diff_words(a: str, b: str) -> tuple[str, str]:
    """Mark words that differ between the two glosses, so the edit is visible at a glance."""
    aw, bw = (a or "").split(), (b or "").split()
    sa, sb = set(w.strip(".,;:'\"()").lower() for w in aw), set(w.strip(".,;:'\"()").lower() for w in bw)
    def render(words, other):
        out = []
        for w in words:
            k = w.strip(".,;:'\"()").lower()
            out.append(f"<mark>{esc(w)}</mark>" if k and k not in other else esc(w))
        return " ".join(out)
    return render(aw, sb), render(bw, sa)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(PROJ / "data/nla/n12_examples/steer_results.jsonl"))
    ap.add_argument("--out", default=str(PROJ / "data/nla/n12_examples/section.html"))
    ap.add_argument("--condition", default="V1_gloss")
    ap.add_argument("--max-cases", type=int, default=8)
    ap.add_argument("--max-reply", type=int, default=1200)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.results) if l.strip()]
    rows = [r for r in rows if "error" not in r and r.get("reply")]
    by: dict[str, dict] = defaultdict(dict)
    for r in rows:
        if r["condition"] not in (args.condition, "V3_taskvec"):
            continue
        # alpha 0 is the unsteered run; it is identical across conditions by construction
        slot = "base" if float(r["alpha"]) == 0.0 else r["condition"]
        by[r["snippet_id"]].setdefault(slot, r)

    cases = []
    for sid, d in by.items():
        b, s = d.get("base"), d.get(args.condition)
        if not b or not s:
            continue
        if bool(b["correct"]) == bool(s["correct"]):
            continue                                   # only the flips are interesting here
        cases.append((sid, b, s))
    # recoveries first, then damage, but both shown
    cases.sort(key=lambda t: (t[1]["correct"], t[0]))
    rec = [c for c in cases if not c[1]["correct"]]
    dam = [c for c in cases if c[1]["correct"]]
    chosen = rec[: args.max_cases // 2 + 1] + dam[: args.max_cases // 2]
    print(f"[ex] {len(rec)} recoveries, {len(dam)} damages; rendering {len(chosen)}")

    blocks = []
    for sid, b, s in chosen:
        gd, gt = diff_words(s.get("gloss_decoy") or "", s.get("gloss_true") or "")
        good = bool(s["correct"])
        blocks.append(f'''
      <article class="sx {'sx-good' if good else 'sx-bad'}">
        <div class="sxhead">
          <span class="id">{esc(sid)}</span>
          <span class="pill {'p-ok' if good else 'p-bad'}">{'recovered' if good else 'broken'}</span>
          <span class="pill p-mute">truth {esc(clip(str(s.get('truth')), 40))}</span>
          <span class="sxa">&alpha;=1.0 &#183; one position</span>
        </div>
        <div class="sxedit">
          <p class="sxlab">the edit &mdash; the steering vector is the difference between these two sentences</p>
          <div class="sxg"><b>decoy</b><span>{gd}</span></div>
          <div class="sxg"><b>true</b><span>{gt}</span></div>
        </div>
        <div class="sxcols">
          <div class="sxcol">
            <p class="sxlab">unsteered &#183; answered <code>{esc(clip(str(b.get('answer')),30))}</code>
              <em class="{'ok' if b['correct'] else 'no'}">{'correct' if b['correct'] else 'wrong'}</em></p>
            <pre>{esc(clip(b['reply'], args.max_reply))}</pre>
          </div>
          <div class="sxcol">
            <p class="sxlab">steered &#183; answered <code>{esc(clip(str(s.get('answer')),30))}</code>
              <em class="{'ok' if s['correct'] else 'no'}">{'correct' if s['correct'] else 'wrong'}</em></p>
            <pre>{esc(clip(s['reply'], args.max_reply))}</pre>
          </div>
        </div>
      </article>''')

    Path(args.out).write_text("\n".join(blocks))
    print(f"[ex] wrote {args.out} ({sum(len(b) for b in blocks):,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
