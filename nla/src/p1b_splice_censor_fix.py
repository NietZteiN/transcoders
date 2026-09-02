"""Splice the re-generated censored items back into the ladder.

Only items that were EVER censored were regenerated, at a 8192 budget, all ten draws each.
Greedy decoding stops at EOS and max_new_tokens only bounds the loop, so any generation that
terminated under the old 2048 cap is token-identical under the new one — which is why replacing
just these items is equivalent to regenerating all 6,000, and why no item ends up straddling two
budgets (whole items are swapped, never individual draws).

Originals are copied to *.pre_censor_fix before anything is written, and a provenance file
records exactly which items moved and what it did to their parse rate.
"""
from __future__ import annotations
import json, shutil, sys
from datetime import datetime, timezone
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
PROJ = _HERE.parent.parent
from p1b_ladder import read_draws  # noqa: E402

TIERS = ("L0", "L1", "L1b", "L2", "L3")
HOSTS = {"gemma12b": "ladder_gemma12b", "llama8b": "ladder_llama8b"}


def main() -> int:
    censored = json.loads((PROJ / "data/nla/p0/p1b/censored_items.json").read_text())
    prov = {"action": "splice_censor_fix", "new_budget": 8192, "old_budget": 2048,
            "rationale": "greedy stops at EOS; a larger cap cannot change a run that already "
                         "terminated, so only censored items are affected",
            "hosts": {}}

    for host, sub in HOSTS.items():
        prov["hosts"][host] = {}
        base = PROJ / "data/nla/p0/p1b" / sub
        fix = PROJ / "data/nla/p0/p1b" / f"censor_fix_{host}"
        for tier in TIERS:
            want = set(censored.get(f"{host}/{tier}", []))
            if not want:
                continue
            old_rows = read_draws(base / tier)
            new_rows = read_draws(fix / tier)
            if not new_rows:
                print(f"  {host}/{tier}: NO FIX ROWS — skipped", flush=True)
                continue

            kept = [r for r in old_rows if r["snippet_id"] not in want]
            replaced = [r for r in old_rows if r["snippet_id"] in want]
            merged = kept + [r for r in new_rows if r["snippet_id"] in want]

            # back up every shard file once, then collapse to a single draws.jsonl
            for f in sorted((base / tier).glob("draws*.jsonl")):
                bak = f.with_suffix(f.suffix + ".pre_censor_fix")
                if not bak.exists():
                    shutil.copy2(f, bak)
            for f in sorted((base / tier).glob("draws*.jsonl")):
                f.unlink()
            with open(base / tier / "draws.jsonl", "w") as fh:
                for r in sorted(merged, key=lambda r: (r["draw"], r["snippet_id"])):
                    fh.write(json.dumps(r) + "\n")

            def pr(rows): return sum(1 for r in rows if r.get("parsed", True)) / max(len(rows), 1)
            prov["hosts"][host][tier] = {
                "items_replaced": sorted(want), "rows_out": len(replaced),
                "rows_in": len(merged) - len(kept), "total_rows": len(merged),
                "parse_before": round(pr(old_rows), 4), "parse_after": round(pr(merged), 4),
            }
            b = prov["hosts"][host][tier]
            print(f"  {host:<9}{tier:<5}replaced {len(want):>2} items "
                  f"({b['rows_out']}->{b['rows_in']} rows) · total {b['total_rows']} · "
                  f"parse {b['parse_before']:.4f} -> {b['parse_after']:.4f}", flush=True)

    prov["finished_utc"] = datetime.now(timezone.utc).isoformat()
    out = PROJ / "data/nla/p0/p1b/censor_fix_provenance.json"
    out.write_text(json.dumps(prov, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
