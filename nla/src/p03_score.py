"""P0.3 scorer — per-cell P@1 from the per-run score.json tree.

Pre-registration: log/nla-harness/2026-08-27_p0-triage-prereg.md

The obfuscation runner writes one directory per (snippet, technique, run_tag, run), each holding a
`score.json` with `case_correct` / `case_total`. P0.3 tags every cell `p03_<name>`, so the tag is
the grouping key and no filename parsing is needed.

DENOMINATOR, STATED EXPLICITLY. P@1 here is `sum(case_correct) / sum(case_total)` over every run
in a cell. That is NOT necessarily the same denominator B5's `cells_scored.json` used (its 1,930
does not equal 164 snippets x 3 runs x ~14 cases), and the two are deliberately not compared: the
pre-registered decision rule only ever compares P0.3 cells against the P0.3 baseline, which is
re-run inside this same batch precisely so that no cross-run denominator has to be reconciled.

The smoke-test tag is excluded by name. It ran one snippet at 64 max-new-tokens and would drag a
cell average without ever looking wrong.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

_PROJ = Path(__file__).resolve().parent.parent.parent
_REPL = _PROJ.parent / "allocation_replication"
_RESULT_ROOT = (_REPL / "artifact" / "artifacts" / "obfuscation" / "result"
                / "Qwen_Qwen2.5-Coder-7B-Instruct")

EXCLUDE = {"p03_smoke"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(_RESULT_ROOT))
    ap.add_argument("--technique", default="adversarial_rename")
    ap.add_argument("--prefix", default="p03_")
    ap.add_argument("--out", default=str(_PROJ / "data/nla/p0/p03/cells_scored.json"))
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"[p03-score] result root missing: {root}")
        return 1

    correct: dict[str, int] = defaultdict(int)
    total: dict[str, int] = defaultdict(int)
    runs: dict[str, int] = defaultdict(int)
    snippets: dict[str, set[str]] = defaultdict(set)
    incomplete: dict[str, int] = defaultdict(int)

    for sc in root.glob(f"*/{args.technique}/{args.prefix}*/run_*/score.json"):
        tag = sc.parent.parent.name
        if tag in EXCLUDE:
            continue
        snippet = sc.parent.parent.parent.parent.name
        try:
            d = json.loads(sc.read_text())
        except Exception:
            incomplete[tag] += 1
            continue
        ct = d.get("case_total")
        cc = d.get("case_correct")
        if not isinstance(ct, int) or not isinstance(cc, int) or ct <= 0:
            # A run that produced no parseable cases is counted, not silently dropped: a cell
            # whose denominator quietly shrank is indistinguishable from one that did well.
            incomplete[tag] += 1
            continue
        correct[tag] += cc
        total[tag] += ct
        runs[tag] += 1
        snippets[tag].add(snippet)

    if not total:
        print(f"[p03-score] no scored runs under {root} matching {args.prefix}*")
        return 1

    out = {}
    for tag in sorted(total):
        name = re.sub(rf"^{re.escape(args.prefix)}", "", tag)
        out[name] = {
            "p_at_1": 100.0 * correct[tag] / total[tag],
            "cases": total[tag],
            "correct": correct[tag],
            "runs": runs[tag],
            "snippets": len(snippets[tag]),
            "runs_unscorable": incomplete[tag],
        }

    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1))

    print(f"[p03-score] {'cell':<14}{'P@1':>8}{'cases':>9}{'runs':>7}{'snips':>7}{'bad':>6}")
    for k, v in out.items():
        print(f"[p03-score] {k:<14}{v['p_at_1']:>8.2f}{v['cases']:>9}{v['runs']:>7}"
              f"{v['snippets']:>7}{v['runs_unscorable']:>6}")
    print(f"[p03-score] wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
