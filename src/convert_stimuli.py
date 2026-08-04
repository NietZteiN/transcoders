"""Convert the canonical Papers 2-3 stimuli into the Snippet JSONL schema (src/data.py).

Inputs (symlinks under data/stimuli/, provenance in data/DATA_SOURCES.md):
  dataset_a_source.json / dataset_b_source.json — rows with task_id, obfuscation_level,
  language, code, fn_name, expected_output.

Output: data/stimuli/dataset_{a,b}/dataset_{a,b}.jsonl + data/stimuli/conversion_report.json

IDENTIFIER CLASSIFICATION — derived CROSS-TIER from the datasets themselves, not from the
alignment parquets: the parquets' L1b_mapping_* come from a DIFFERENT generation round
(verified 2026-08-04: A's fibfib->smoothArea vs parquet's fibfib->rns; ~0/10 js L1b rows
contain any parquet adversarial name), so the only mapping consistent with what humans and
models actually saw is the one recovered from the stimuli. Method: group rows by
(task_id, language); walk the ordered identifier-token sequences of two tiers' codes in
parallel (renaming preserves structure, so equal-length sequences pair positionally):
  L0 <-> L1   -> neutral-rename map     L0 <-> L1b  -> adversarial map (THE decoy<->true pairing E1 edits)
  L2 <-> L3   -> stacked-rename map (L3 = flattened L2 + renames; classify value by shape)
Falls back to set-difference (unpaired) when sequences don't align; pairing_ok recorded.
Every span carries a class in meta.span_info: orig | adversarial | fn_orig | fn_adversarial
| l1_neutral | self_derived. Dispatcher spans: python `while <var> <cmp>` state variable;
javascript object-dispatch table (`const <name> = {` with quoted helper keys).

Run:  python -m src.convert_stimuli [--validate]
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

from src.configs import PROJECT_ROOT
from src.data import Snippet

STIM = PROJECT_ROOT / "data" / "stimuli"

KEYWORDS = {
    "python": {"def","return","if","else","elif","for","while","in","not","and","or","is","None","True","False",
               "import","from","as","class","try","except","finally","with","lambda","yield","pass","break",
               "continue","global","nonlocal","assert","del","raise","print","range","len","int","str","float",
               "list","dict","set","tuple","bool","enumerate","zip","abs","min","max","sum","sorted","typing",
               "List","Dict","Optional","Tuple","Union","Any","self","append","sort","count","join","split",
               "reverse","reversed","isinstance","map","filter","round","input","open","ord","chr","type"},
    "javascript": {"const","let","var","function","return","if","else","for","while","of","in","new","typeof",
                   "instanceof","null","undefined","true","false","class","try","catch","finally","throw",
                   "break","continue","switch","case","default","do","this","console","log","Math","floor",
                   "ceil","abs","min","max","round","sqrt","pow","length","push","pop","shift","unshift","map",
                   "filter","forEach","reduce","split","join","slice","splice","concat","charAt","charCodeAt",
                   "fromCharCode","toLowerCase","toUpperCase","trim","String","Number","Array","Boolean",
                   "parseInt","parseFloat","JSON","Object","keys","values","entries","indexOf","includes",
                   "replace","repeat","substring","substr","toString","sort","reverse","some","every","find"},
}
IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_$]*")
L1_NEUTRAL_RE = re.compile(r"^(?:var|func|fn|v|f)_[0-9a-fA-F]{2,8}$")
PY_DISPATCH_RE = re.compile(r"\bwhile\s*\(?\s*([A-Za-z_]\w*)\s*(?:<|<=|!=|==)")
JS_DISPATCH_TABLE_RE = re.compile(r"\b(?:const|var|let)\s+([A-Za-z_$][\w$]*)\s*=\s*\{\s*['\"]")

TIERS = ["L0", "L1", "L1b", "L2", "L3"]


def word_spans(code: str, name: str) -> list[list[int]]:
    return [[m.start(), m.end()] for m in re.finditer(rf"(?<![A-Za-z0-9_$]){re.escape(name)}(?![A-Za-z0-9_$])", code)]


def ident_seq(code: str) -> list[str]:
    """Ordered sequence of ALL identifier-shaped tokens (keywords included — they pair with
    themselves in the parallel walk and anchor the alignment)."""
    return IDENT_RE.findall(code)


def derive_rename_map(code_from: str, code_to: str, language: str) -> tuple[dict[str, str], bool]:
    """Positional rename map between two tier codes. Returns (map, pairing_ok).

    Renaming preserves token structure, so equal-length identifier sequences pair 1:1.
    A pairing is kept only if it is a consistent bijection on the changed names; otherwise
    falls back to set-difference (names present in `to` but not `from`), unpaired.
    """
    kws = KEYWORDS[language]
    sf, st = ident_seq(code_from), ident_seq(code_to)
    if len(sf) == len(st):
        fwd: dict[str, str] = {}
        ok = True
        for a, b in zip(sf, st):
            if a == b:
                continue
            if a in kws or b in kws:
                ok = False
                break
            if fwd.get(a, b) != b:      # same orig must always map to the same new name
                ok = False
                break
            fwd[a] = b
        if ok and len(set(fwd.values())) == len(fwd):
            return fwd, True
    new_names = (set(t for t in st if t not in kws) - set(sf))
    return {f"?unpaired{i}": n for i, n in enumerate(sorted(new_names))}, False


def orig_identifiers(code: str, language: str) -> list[str]:
    kws = KEYWORDS[language]
    return [t for t, _ in Counter(t for t in ident_seq(code) if t not in kws).most_common()]


def detect_dispatcher(code: str, language: str) -> tuple[str | None, list[list[int]]]:
    rx = PY_DISPATCH_RE if language == "python" else JS_DISPATCH_TABLE_RE
    m = rx.search(code)
    if not m:                            # js sometimes flattens python-style; try both
        m = (JS_DISPATCH_TABLE_RE if language == "python" else PY_DISPATCH_RE).search(code)
    if m:
        return m.group(1), word_spans(code, m.group(1))
    return None, []


def build_group_info(tiers: dict[str, dict], language: str) -> dict:
    """Per-(task, language) derived knowledge: orig names + rename maps per tier."""
    info: dict = {"orig": [], "maps": {}, "pairing_ok": {}}
    if "L0" in tiers:
        info["orig"] = orig_identifiers(tiers["L0"]["code"], language)
    for src_tier, dst_tier in [("L0", "L1"), ("L0", "L1b"), ("L2", "L3")]:
        if src_tier in tiers and dst_tier in tiers:
            mp, ok = derive_rename_map(tiers[src_tier]["code"], tiers[dst_tier]["code"], language)
            info["maps"][dst_tier] = mp
            info["pairing_ok"][dst_tier] = ok
    return info


def classify_names(tier: str, info: dict, fn_name: str | None) -> list[tuple[str, str]]:
    """(name, cls) candidates for one tier, most-specific class first."""
    out: list[tuple[str, str]] = []
    mp = info["maps"].get(tier, {})
    if tier in ("L1", "L1b", "L3"):
        for _orig, new in mp.items():
            if tier == "L1" or L1_NEUTRAL_RE.match(new):
                cls = "l1_neutral"
            else:
                cls = "fn_adversarial" if _orig == fn_name else "adversarial"
            out.append((new, cls))
    for name in info["orig"]:
        out.append((name, "fn_orig" if name == fn_name else "orig"))
    return out


def convert_row(row: dict, info: dict) -> Snippet:
    code, tier = row["code"], row["obfuscation_level"]
    language = row["language"].lower()
    fn_name = row.get("fn_name")

    span_info: list[dict] = []
    seen: set[tuple[int, int]] = set()

    def add(name: str, cls: str) -> None:
        for sp in word_spans(code, name):
            if (sp[0], sp[1]) not in seen:
                seen.add((sp[0], sp[1]))
                span_info.append({"span": sp, "name": name, "cls": cls})

    for name, cls in classify_names(tier, info, fn_name):
        add(name, cls)
    if fn_name:
        add(fn_name, "fn_orig")
    if not span_info:                    # nothing derived (e.g. missing L0 sibling): fallback
        for name in orig_identifiers(code, language):
            add(name, "self_derived")

    state_var, dispatcher_spans = (None, [])
    if tier in ("L2", "L3"):
        state_var, dispatcher_spans = detect_dispatcher(code, language)

    return Snippet(
        snippet_id=row["task_id"], tier=tier, language=language, code=code,
        expected_output=str(row.get("expected_output")),
        identifier_spans=[si["span"] for si in span_info],
        dispatcher_spans=dispatcher_spans,
        meta={
            "fn_name": fn_name, "input": str(row.get("input")),
            "dataset_source": row.get("dataset_source"),
            "span_info": span_info,
            "rename_map": info["maps"].get(tier),           # decoy<->true pairing for E1/E2
            "pairing_ok": info["pairing_ok"].get(tier),
            "dispatcher_state_var": state_var,
        },
    )


def convert(dataset_key: str, report: dict) -> Path:
    rows = json.load(open(STIM / f"{dataset_key}_source.json"))
    groups: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
    for row in rows:
        groups[(row["task_id"], row["language"].lower())][row["obfuscation_level"]] = row

    out_dir = STIM / dataset_key
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{dataset_key}.jsonl"

    stats: dict[str, Counter] = defaultdict(Counter)
    no_spans, no_dispatcher, unpaired = [], [], []
    with open(out_path, "w") as f:
        for (task_id, language), tiers in groups.items():
            info = build_group_info(tiers, language)
            for tier in TIERS:
                if tier not in tiers:
                    continue
                s = convert_row(tiers[tier], info)
                if not s.identifier_spans:
                    no_spans.append(f"{s.snippet_id}/{tier}")
                if tier in ("L2", "L3") and not s.dispatcher_spans:
                    no_dispatcher.append(f"{s.snippet_id}/{tier}")
                if tier in ("L1", "L1b", "L3") and info["pairing_ok"].get(tier) is False:
                    unpaired.append(f"{s.snippet_id}/{tier}")
                stats["rows_per_tier"][tier] += 1
                stats["spans_per_tier"][tier] += len(s.identifier_spans)
                for si in s.meta["span_info"]:
                    stats["span_cls"][si["cls"]] += 1
                f.write(json.dumps(asdict(s)) + "\n")

    if no_spans:
        raise ValueError(f"{dataset_key}: {len(no_spans)} rows with 0 identifier spans: {no_spans[:5]}")
    report[dataset_key] = {
        "rows": sum(stats["rows_per_tier"].values()),
        "rows_per_tier": dict(stats["rows_per_tier"]),
        "identifier_spans_per_tier": dict(stats["spans_per_tier"]),
        "span_class_counts": dict(stats["span_cls"]),
        "rows_pairing_fallback": unpaired,                    # set-difference, no positional pairing
        "rows_without_dispatcher_var": no_dispatcher,
        "output": str(out_path.relative_to(PROJECT_ROOT)),
    }
    return out_path


def validate_with_tokenizers(jsonl_paths: list[Path], report: dict) -> None:
    """Span->token resolution check with the real (cached) tokenizers the experiments use."""
    from transformers import AutoTokenizer

    from src.data import load_jsonl
    from src.extract_activations import resolve_spans_to_positions

    for model_id in ["meta-llama/Llama-3.1-8B-Instruct", "Qwen/Qwen3-0.6B"]:
        tok = AutoTokenizer.from_pretrained(model_id)
        for p in jsonl_paths:
            tot_res = tot_spans = rows_below_95 = 0
            for s in load_jsonl(p):
                enc = tok(s.code, return_offsets_mapping=True, truncation=True, max_length=2048)
                _, n_res, n_tot = resolve_spans_to_positions(
                    s.identifier_spans, enc["offset_mapping"], tok, enc["input_ids"], s.code)
                tot_res += n_res
                tot_spans += n_tot
                if n_tot and n_res / n_tot < 0.95:
                    rows_below_95 += 1
            rate = tot_res / tot_spans if tot_spans else 0.0
            key = f"validation/{model_id.split('/')[-1]}/{p.parent.name}"
            report[key] = {"span_resolution_rate": round(rate, 4),
                           "spans": tot_spans, "rows_below_0.95": rows_below_95}
            print(f"  {key}: rate={rate:.4f} ({tot_res}/{tot_spans}), rows<0.95: {rows_below_95}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert canonical stimuli to Snippet JSONL (cross-tier span derivation).")
    ap.add_argument("--validate", action="store_true", help="Check span->token resolution with real tokenizers.")
    args = ap.parse_args()

    report: dict = {"note": "identifier classes derived cross-tier from the stimuli; alignment "
                            "parquets NOT used (different generation round — see DATA_SOURCES.md)"}
    paths = [convert(k, report) for k in ["dataset_a", "dataset_b"]]
    for k in ["dataset_a", "dataset_b"]:
        r = report[k]
        print(f"{k}: {r['rows']} rows -> {r['output']}\n"
              f"   span classes {r['span_class_counts']}\n"
              f"   pairing fallbacks: {len(r['rows_pairing_fallback'])} | L2/L3 without dispatcher var: {len(r['rows_without_dispatcher_var'])}")
    if args.validate:
        print("validating span->token resolution with real tokenizers:")
        validate_with_tokenizers(paths, report)
    (STIM / "conversion_report.json").write_text(json.dumps(report, indent=2))
    print(f"report -> {STIM / 'conversion_report.json'}")


if __name__ == "__main__":
    main()
