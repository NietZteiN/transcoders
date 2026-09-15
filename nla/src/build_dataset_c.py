"""Build `dataset_c` — the 6x larger stimulus pool the accuracy arm needs — from the banked
HumanEval-X L1b alignment parquets.

WHY THIS EXISTS. Every accuracy result in this programme ran on 60 items with 7 flippable, and the
binding constraint on every one of them was the corpus, not the method: the decoy costs 2 items
(L0 34/60 vs L1b 32/60), so no intervention can show more than +0.033 and a perfect rescue of the
flippable subset is still McNemar p = 0.092. I had been treating that as a fact about the workspace.
It is not: `data/stimuli/alignment/humaneval_x_{python,js}_L1b_mapping.parquet` hold **164 problems
per language**, each already carrying all five tiers (`code_L0`, `obf_code_L1`, `code_L1b`,
`obf_code_L2`, `obf_code_L3`), the adversarial rename maps, and execution-derived I/O pairs. The
60-item set is a *sample* of that. This script converts the parquets into the `dataset_a`/`dataset_b`
row schema so the existing loaders can read them.

WHAT IT DOES NOT DO. It does not touch `load_pairs`' default dataset list. That function is imported
by ~10 scripts and silently widening it would change the 60-item set under every banked comparison in
the programme -- every `dG_S_*`, every gate, every identity tolerance. Inclusion is opt-in via
`load_pairs(datasets=...)`, and the banked default stays `("dataset_a", "dataset_b")`.

VALIDATION GATE (internal, and it passes 164/164 per language). The first version of this script
gated on matching the banked `dataset_a` stimulus for the ~20 overlapping HumanEval-X problems. That
gate FAILED 39 of 58 comparisons, and the failure was informative: the banked items are a *different
draw*. `Python/31`'s banked L0 is `def myFunct(n)` where the parquet has `def is_prime(n)`, and
`Python/0`'s banked L1b renames to `write_top/application/dayofweek` where the parquet renames to
`super_parent/failed_to_create_image_path_list`. Adversarial renaming is sampled, so both are valid
L1b stimuli of the same program -- the banked 60 are simply not reproducible from these parquets.

The gate that actually validates the conversion is INTERNAL: applying `rename_map` to `code_L0`
(word-boundary substitution, longest key first) must reproduce `code_L1b` exactly, and no original
identifier may survive in L1b nor any decoy name appear in L0. That holds for **164/164 rows in both
languages with zero leaks**, which certifies the tier pairing and the map together without reference
to any earlier sample.

QUARANTINE, NOT OVERRIDE. Two of 259 items fail the gate for real reasons and are DROPPED rather than
waved through: `JavaScript/137` maps an identifier onto **`let`** -- a JS keyword, so the "decoy name
in L0" hit is a genuine collision that would make the renamed code mean something different -- and
`JavaScript/95` carries `hi` on both sides of its map. A per-item quarantine keeps the gate meaningful
at a cost of 0.8 % of the corpus; `--allow-leaks` exists only to inspect them deliberately.

OVERLAP IS REPORTED, NOT GATED. ~20 of these problems underlie items in the banked 60. They are
DIFFERENT stimuli here, but they share the same underlying program, so results on them are not
independent of the banked results. `meta.task_id` and the manifest's `overlap_task_ids` record which,
so an analysis can cluster or exclude them; ids are namespaced `hexc/...` so nothing can collide.

TIER -> COLUMN, and the one asymmetry that matters: only **L1b** renames the FUNCTION as well as the
variables (`obf_code_L1` keeps `def has_close_elements(var_bd90...)`). `build_call` maps `fn_name`
through `rename_map` only when `meta.pairing_ok` is set, so pairing_ok is True for L1b alone --
setting it on L1/L2/L3 would look harmless and would silently produce calls to a function that does
not exist in that tier's code.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).resolve().parent
_PROJ = _HERE.parents[1]
sys.path.insert(0, str(_HERE))

TAG = "[DSC]"
PREFIX = "hexc"            # namespaces the ids: parquet task_ids ('Python/0') COLLIDE with dataset_a's
SOURCES = {"python": "humaneval_x_python_L1b_mapping.parquet",
           "javascript": "humaneval_x_js_L1b_mapping.parquet"}
TIER_COL = {"L0": "code_L0", "L1": "obf_code_L1", "L1b": "code_L1b",
            "L2": "obf_code_L2", "L3": "obf_code_L3"}


def spans_for(code: str, terms) -> list[list[int]]:
    """Character spans of every word-boundary occurrence of each term, longest term first.

    Longest-first matters: a decoy set can contain both `bb_carry` and `bb_carry_2`, and matching the
    shorter one first would annotate a prefix of the longer identifier.
    """
    out: list[list[int]] = []
    for t in sorted({str(x) for x in terms if x and len(str(x)) > 1}, key=len, reverse=True):
        for m in re.finditer(rf"(?<![\w$]){re.escape(t)}(?![\w$])", code):
            if not any(a <= m.start() < b for a, b in out):
                out.append([m.start(), m.end()])
    return sorted(out)


def io_pair(raw) -> tuple[str, str] | None:
    """(input-list literal, expected) from the first I/O pair, or None if unusable.

    `inputs_raw` is a list of per-argument SOURCE strings, so the arg list is their join in brackets;
    `build_call` runs `ast.literal_eval` on it, which is also the filter that drops JS rows carrying
    `true`/`false`/`null` literals. Those are skipped and counted, never silently coerced.
    """
    try:
        pairs = json.loads(raw) if isinstance(raw, str) else list(raw or [])
    except (TypeError, ValueError):
        return None
    for p in pairs:
        ins, exp = p.get("inputs_raw"), p.get("expected_raw")
        if ins is None or exp is None:
            continue
        lit = "[" + ", ".join(str(x) for x in list(ins)) + "]"
        try:
            import ast
            args = ast.literal_eval(lit)
        except (ValueError, SyntaxError):
            continue
        if isinstance(args, list):
            return lit, str(exp)
    return None


def rename_map_of(row) -> dict:
    m: dict = {}
    for col in ("L1b_mapping_var", "L1b_mapping_func"):
        v = row.get(col)
        if isinstance(v, str) and v.strip() and v.strip() not in ("None", "nan"):
            try:
                m.update(json.loads(v))
            except ValueError:
                pass
        elif isinstance(v, dict):
            m.update(v)
    return m


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(_PROJ / "data/stimuli/dataset_c/dataset_c.jsonl"))
    ap.add_argument("--allow-leaks", action="store_true",
                    help="write even if the internal rename gate fails (records the count)")
    args = ap.parse_args()

    banked: dict[str, dict[str, dict]] = {}
    for ds in ("dataset_a", "dataset_b"):
        p = _PROJ / "data/stimuli" / ds / f"{ds}.jsonl"
        if p.exists():
            for r in map(json.loads, open(p)):
                banked.setdefault(r["snippet_id"], {})[r["tier"]] = r

    def apply_map(code: str, rm: dict) -> str:
        for k in sorted(rm, key=len, reverse=True):      # longest first: a prefix must not shadow
            code = re.sub(rf"(?<![\w$]){re.escape(k)}(?![\w$])", rm[k], code)
        return code

    rows: list[dict] = []
    skipped: dict[str, int] = {"no_io_pair": 0, "no_code": 0, "no_rename_map": 0}
    gate = {"checked": 0, "rename_mismatch": 0, "key_leak": 0, "value_leak": 0}
    failures: list[str] = []
    overlap: list[str] = []
    quarantine: set[str] = set()          # tasks failing the rename gate are DROPPED, not overridden
    for lang, fname in SOURCES.items():
        df = pd.read_parquet(_PROJ / "data/stimuli/alignment" / fname)
        for _, r in df.iterrows():
            task = str(r["task_id"])
            rm = rename_map_of(r)
            if not rm:
                skipped["no_rename_map"] += 1; continue
            pair = io_pair(r.get("io_pairs_json"))
            if pair is None:
                skipped["no_io_pair"] += 1; continue
            inp, expected = pair
            codes = {t: (r.get(c) if isinstance(r.get(c), str) else None) for t, c in TIER_COL.items()}
            if not codes["L0"] or not codes["L1b"]:
                skipped["no_code"] += 1; continue

            # GATE: the rename map must carry L0 exactly onto L1b, with no identifier leaking either way
            gate["checked"] += 1
            if apply_map(codes["L0"], rm) != codes["L1b"]:
                gate["rename_mismatch"] += 1; failures.append(f"{task}:rename"); quarantine.add(task)
            for k, v in rm.items():
                if re.search(rf"(?<![\w$]){re.escape(k)}(?![\w$])", codes["L1b"]):
                    gate["key_leak"] += 1; failures.append(f"{task}:key_leak:{k}"); quarantine.add(task)
                if re.search(rf"(?<![\w$]){re.escape(str(v))}(?![\w$])", codes["L0"]):
                    gate["value_leak"] += 1; failures.append(f"{task}:value_leak:{v}"); quarantine.add(task)
            # OVERLAP with the banked 60 -- reported, never gated (different draw of the same program)
            for cand in (task, f"humaneval-x-{'python' if lang == 'python' else 'javascript'}/{task}"):
                if cand in banked:
                    overlap.append(f"{task}<->{cand}")

            sid = f"{PREFIX}/{task}"
            if task in quarantine:
                continue
            for tier, code in codes.items():
                if code is None:
                    continue
                renamed_fn = tier == "L1b"        # ONLY L1b renames the function -- see the docstring
                terms = list(rm.values()) if renamed_fn else list(rm.keys())
                rows.append({
                    "snippet_id": sid, "tier": tier, "language": lang, "code": code,
                    "expected_output": expected,
                    "identifier_spans": spans_for(code, terms),
                    "dispatcher_spans": [],
                    "meta": {"fn_name": str(r.get("fn_name") or ""), "input": inp,
                             "dataset_source": f"humaneval_x_{lang}_L1b_mapping.parquet",
                             "rename_map": rm, "pairing_ok": renamed_fn,
                             "built_by": "build_dataset_c.py", "task_id": task},
                })

    ids = sorted({r["snippet_id"] for r in rows})
    n_bad = gate["rename_mismatch"] + gate["key_leak"] + gate["value_leak"]
    print(f"{TAG} {len(ids)} snippets · {len(rows)} rows · skipped {skipped}", flush=True)
    print(f"{TAG} rename gate: {gate['checked']} items · mismatch {gate['rename_mismatch']} · "
          f"key leaks {gate['key_leak']} · decoy-in-L0 {gate['value_leak']}"
          + (f" -> {failures[:6]}" if failures else "  ALL PASS"), flush=True)
    print(f"{TAG} overlap with the banked 60 (reported, NOT pooled as independent): "
          f"{len(overlap)} tasks {overlap[:4]}", flush=True)
    print(f"{TAG} quarantined {len(quarantine)} item(s) that failed the gate: {sorted(quarantine)}",
          flush=True)
    if n_bad and args.allow_leaks:
        print(f"{TAG} --allow-leaks: quarantine NOT applied; this output is for inspection only")

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    man = {"experiment": "dataset_c_build", "n_snippets": len(ids), "n_rows": len(rows),
           "skipped": skipped, "rename_gate": gate, "gate_failures": failures,
           "quarantined": sorted(quarantine), "overlap_task_ids": overlap, "overlap_note":
               "different adversarial draw of the same underlying program as a banked-60 item; share "
               "the program, so never pool them as independent observations",
           "sources": SOURCES, "prefix": PREFIX,
           "built_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    (out.parent / "dataset_c_manifest.json").write_text(json.dumps(man, indent=1))
    print(f"{TAG} wrote {out} and the manifest")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
