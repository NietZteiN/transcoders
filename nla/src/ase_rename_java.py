"""H-R1b — adversarial identifier renaming for the ASE-2026 Java corpus.

Pre-registered in log/nla-harness/2026-09-14_ase-replication-prereg.md, amended by
log/nla-harness/2026-09-14_ase-prereg-amendment.md.

THIS RENAMER IS OURS, NOT THEIRS, AND THAT IS A DECLARED LIMITATION. Their generator, equivalence
checker and prepared obfuscated corpora are all excluded from the artifact (`GITREADY_MANIFEST.md`),
so a null on H-R1b cannot by itself separate "this model is robust" from "my renamer is weaker than
theirs". Two design choices make it as strong as I can honestly make it:

  * **The decoy vocabulary is harvested from our own L1b rename maps** (the HumanEval-X alignment
    parquets' `L1b_mapping_var` / `L1b_mapping_func` values -- names like `failed_to_create_image_
    path_list`, `aliased_ns`, `bb_carry`). So this is *adversarially misleading* naming in exactly
    this project's sense, and the cross-setup comparison against my +0.0117 on Python/JS is against
    the same vocabulary rather than a new one.
  * **The method name is renamed too.** Because case packs are rebuilt on the obfuscated variant,
    that puts the decoy name inside every case expression the model is asked to evaluate -- the
    mechanism nominated for their 76.49 -> 40.20 drop (which lands *below* the 50 % chance floor).

WHAT IS DELIBERATELY NOT RENAMED: the class name (the harness compiles `<class>.java` and runs that
class), `main`, and anything not declared in the file. Only declarations found by `javalang` --
method names, formal parameters, local variables, fields -- are candidates, so a library member that
merely shares a name with nothing declared here is never touched. Renaming is then a word-boundary
textual substitution, longest name first so a short name cannot shadow a longer one.

VALIDATION IS NOT OPTIONAL AND IS NOT DONE HERE: the rebuilt case pack recompiles and re-executes
every variant (`obfuscation/main.py` raises on a compile failure), and the amendment's PACKS-PAIRED
gate requires the renamed pack to reproduce the original's case count and label sequence. A variant
that fails either is a renamer bug and is quarantined, never reported as a result.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_PROJ = Path(__file__).resolve().parents[2]
TAG = "[RENAME]"
SEED = 20260724

# Never rename these even if declared: the class entry point AND the harness contract that the case
# builder pattern-matches on. `correct` is load-bearing: java_counterfactual._extract_humaneval_seeds
# searches for the literal `List<Boolean>\s+correct\s*=\s*Arrays\.asList\s*\(`, so renaming it
# silently diverts the renamed condition onto the if/throw or program-level FALLBACK case path,
# yielding a case pack that is not comparable with the original's. That would have read as
# "renaming destroys performance" when it had actually destroyed the case construction.
PROTECTED = {"main", "args", "String", "System", "out", "println", "Solution", "correct"}
# After renaming, the harness pattern MUST still match or the variant is quarantined, not shipped.
HARNESS_RE = r"List<Boolean>\s+correct\s*=\s*Arrays\.asList\s*\("
JAVA_RESERVED = {
    "abstract", "assert", "boolean", "break", "byte", "case", "catch", "char", "class", "const",
    "continue", "default", "do", "double", "else", "enum", "extends", "final", "finally", "float",
    "for", "goto", "if", "implements", "import", "instanceof", "int", "interface", "long", "native",
    "new", "package", "private", "protected", "public", "return", "short", "static", "strictfp",
    "super", "switch", "synchronized", "this", "throw", "throws", "transient", "try", "void",
    "volatile", "while", "var", "record", "yield", "true", "false", "null",
}


def decoy_pool() -> list[str]:
    """Decoy identifiers harvested from this project's own L1b rename maps, camelCased for Java."""
    import pandas as pd
    names: set[str] = set()
    for f in ("humaneval_x_python_L1b_mapping.parquet", "humaneval_x_js_L1b_mapping.parquet"):
        p = _PROJ / "data/stimuli/alignment" / f
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        for col in ("L1b_mapping_var", "L1b_mapping_func"):
            for v in df.get(col, []):
                try:
                    m = json.loads(v) if isinstance(v, str) else (v or {})
                except (TypeError, ValueError):
                    continue
                for name in (m or {}).values():
                    names.add(str(name))
    out = []
    for n in sorted(names):
        parts = re.split(r"[_\W]+", n)
        parts = [q for q in parts if q]
        if not parts:
            continue
        cam = parts[0].lower() + "".join(q.capitalize() for q in parts[1:])
        if cam and not cam[0].isdigit() and cam not in JAVA_RESERVED and len(cam) > 2:
            out.append(cam)
    return sorted(set(out))


def declared_names(code: str) -> list[str]:
    """Names DECLARED in this file: method names, formal parameters, locals, fields."""
    import javalang
    found: set[str] = set()
    try:
        tree = javalang.parse.parse(code)
    except Exception as e:                     # a snippet we cannot parse is skipped, never guessed at
        raise RuntimeError(f"javalang parse failed: {type(e).__name__}: {e}") from e
    for _, node in tree:
        cls = type(node).__name__
        if cls == "MethodDeclaration":
            found.add(node.name)
        elif cls == "FormalParameter":
            found.add(node.name)
        elif cls in ("VariableDeclarator",):
            found.add(node.name)
    return sorted(n for n in found
                  if n and n not in PROTECTED and n not in JAVA_RESERVED and len(n) > 1)


def rename_code(code: str, names: list[str], pool: list[str], seed_key: str) -> tuple[str, dict]:
    import zlib
    rng = __import__("random").Random(SEED + zlib.crc32(seed_key.encode()))
    picks = rng.sample(pool, k=min(len(names), len(pool)))
    mapping = {n: picks[i] for i, n in enumerate(sorted(names, key=len, reverse=True)) if i < len(picks)}
    out = code
    for src in sorted(mapping, key=len, reverse=True):   # longest first: no prefix shadowing
        out = re.sub(rf"(?<![\w$]){re.escape(src)}(?![\w$])", mapping[src], out)
    return out, mapping


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src-dir", default="/scratch/juno/jvl210002/ase2026/"
                                         "LLM-Attention-Fixation_submission/Source/Humaneval/java")
    ap.add_argument("--out-dir", default="/scratch/juno/jvl210002/ase2026/renamed_humaneval_java")
    ap.add_argument("--manifest", default="/scratch/juno/jvl210002/ase2026/rename_manifest.jsonl")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    pool = decoy_pool()
    print(f"{TAG} decoy pool: {len(pool)} names harvested from this project's L1b maps "
          f"(e.g. {pool[:4]})", flush=True)
    if len(pool) < 50:
        print(f"{TAG} REFUSED: decoy pool too small ({len(pool)}) — the alignment parquets are the source")
        return 2

    src = sorted(Path(args.src_dir).glob("*.java"))
    if args.limit:
        src = src[:args.limit]
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    man = open(args.manifest, "w")
    n_ok = n_skip = 0
    for i, p in enumerate(src, 1):
        try:
            names = declared_names(p.read_text())
        except RuntimeError as e:
            n_skip += 1
            man.write(json.dumps({"snippet": p.stem, "error": str(e)}) + "\n")
            print(f"{TAG} {i}/{len(src)} {p.stem} SKIP {e}", flush=True)
            continue
        if not names:
            n_skip += 1
            man.write(json.dumps({"snippet": p.stem, "error": "no renameable declarations"}) + "\n")
            continue
        original = p.read_text()
        code2, mapping = rename_code(original, names, pool, p.stem)
        # GATE: the harness the case builder keys on must survive, and the method name must actually
        # have changed (that is the manipulation -- it is what puts a decoy name in every case expr).
        had_harness = re.search(HARNESS_RE, original) is not None
        if had_harness and re.search(HARNESS_RE, code2) is None:
            n_skip += 1
            man.write(json.dumps({"snippet": p.stem, "error": "harness pattern destroyed by rename"}) + "\n")
            print(f"{TAG} {i}/{len(src)} {p.stem} QUARANTINE: harness pattern lost", flush=True)
            continue
        (out / p.name).write_text(code2)
        man.write(json.dumps({"snippet": p.stem, "n_renamed": len(mapping),
                              "had_harness": had_harness, "rename_map": mapping}) + "\n")
        n_ok += 1
        if i <= 3 or i % 40 == 0:
            print(f"{TAG} {i}/{len(src)} {p.stem} renamed {len(mapping)} identifiers "
                  f"(e.g. {list(mapping.items())[:2]})", flush=True)
    man.close()
    print(f"{TAG} DONE {n_ok} renamed · {n_skip} skipped · out {out}")
    print(f"{TAG} NOTE compile/label validation happens when the case packs are rebuilt on this dir")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
