"""H-R22 — the ADVERSARIAL renamer: a derangement of the program's own identifiers.

Pre-registered in log/nla-harness/2026-09-18_adversarial-rename-prereg.md. A sibling of
`ase_rename_java.py`, which is NOT modified -- it produced every banked H-R1/H-R7/H-R14/H-R18 number.

WHY A NEW RENAMER. H-R18 measured the renaming damage on this model at +0.0069 under greedy decoding,
against the paper's +36.29 on their Qwen2.5-7B, and an oracle ceiling of +0.0052. With no deficit there
is nothing for any steering method to restore, so the stimulus -- not the steering -- is what blocks the
programme (H-R21). Two weaknesses of the existing renamer explain part of the gap:

  1. `len(n) > 1` skipped every SINGLE-CHARACTER identifier, i.e. most loop counters and accumulators
     in HumanEval-X Java. Those are exactly the names a reader uses to track state.
  2. Decoys were drawn with `rng.sample(pool)` -- at RANDOM. A randomly-paired decoy is not adversarial,
     it is merely unfamiliar. That makes the old stimulus closer to this project's **L1 (nonsense
     renaming)** than to **L1b (adversarial renaming, which injects plausible-but-WRONG semantics)`.

THE MANIPULATION HERE, AND WHY IT IS A BETTER CONTROL. Rather than importing an external decoy
vocabulary, the file's own declared identifiers are permuted among themselves by a seeded
**derangement** (a permutation with no fixed point), so every identifier keeps a name that is
*plausible in this very program* while being attached to the WRONG entity: the accumulator is called
`i`, the index is called `sum`. The set of identifier names in the file is therefore **exactly
preserved** -- which is the point. Any damage cannot be blamed on unfamiliar or out-of-distribution
tokens, because no new token is introduced. It isolates "wrong semantics" from "strange vocabulary",
which the pooled-decoy renamer conflates.

SEMANTICS ARE PRESERVED BY CONSTRUCTION: a bijective rename of declarations is a no-op on behaviour.
That is asserted rather than trusted -- the case packs are rebuilt on the variant with the artifact's
own execution-validating `build_case_pack`, and the PACKS-PAIRED gate requires the rebuilt pack to
reproduce the original's case count and label sequence.

TWO HAZARDS, BOTH HANDLED:
  * **Cascading substitution.** A permutation must be applied in ONE pass. Renaming a->b and then b->c
    sequentially (what `ase_rename_java.rename_code` does, safely, because its decoys come from a
    disjoint pool) would corrupt a permutation. Here a single regex alternation with a callback does
    every name simultaneously.
  * **Library member collisions.** A declared name that is also used as a member (`list.size()`) would
    be rewritten inside the member access by a word-boundary regex and break compilation. Any candidate
    that appears after a `.` anywhere in the file is therefore left alone -- conservative, and it keeps
    the compile-failure rate down rather than relying on the pack builder to discard the snippet.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import zlib
from pathlib import Path

TAG = "[SWAP]"
SEED = 20260724

# Same contract as ase_rename_java: the class entry point and the literal the case builder keys on.
PROTECTED = {"main", "args", "String", "System", "out", "println", "Solution", "correct"}
HARNESS_RE = r"List<Boolean>\s+correct\s*=\s*Arrays\.asList\s*\("
JAVA_RESERVED = {
    "abstract", "assert", "boolean", "break", "byte", "case", "catch", "char", "class", "const",
    "continue", "default", "do", "double", "else", "enum", "extends", "final", "finally", "float",
    "for", "goto", "if", "implements", "import", "instanceof", "int", "interface", "long", "native",
    "new", "package", "private", "protected", "public", "return", "short", "static", "strictfp",
    "super", "switch", "synchronized", "this", "throw", "throws", "transient", "try", "void",
    "volatile", "while", "var", "record", "yield", "true", "false", "null",
}


def declared_names(code: str, keep_single_char: bool = True) -> tuple[list[str], list[str]]:
    """Names DECLARED here (methods, formal parameters, locals, fields), and those held back.

    Unlike ase_rename_java.declared_names this KEEPS single-character names -- point (1) above.
    """
    import javalang
    methods: set[str] = set()
    variables: set[str] = set()
    try:
        tree = javalang.parse.parse(code)
    except Exception as e:
        raise RuntimeError(f"javalang parse failed: {type(e).__name__}: {e}") from e
    for _, node in tree:
        cls = type(node).__name__
        if cls == "MethodDeclaration":
            methods.add(node.name)
        elif cls in ("FormalParameter", "VariableDeclarator"):
            variables.add(node.name)
    cand, held = [], []
    for n in sorted(methods | variables):
        if not n or n in PROTECTED or n in JAVA_RESERVED:
            held.append(n); continue
        if len(n) == 1 and not keep_single_char:
            held.append(n); continue
        # The member-access guard applies to VARIABLES ONLY. A local named `size` alongside
        # `list.size()` would be rewritten inside the library call and break compilation, so those are
        # held back. A DECLARED METHOD is the opposite case: the harness always invokes it as
        # `solution.method(...)`, so the guard would hold back the single most important identifier --
        # the method name is what appears inside every case expression the model is asked to evaluate,
        # and renaming it is the mechanism nominated for the paper's 76.49 -> 40.20 drop. Method names
        # are therefore always candidates; the rare collision with a same-named library call is caught
        # by the compile/execute validation when the packs are rebuilt, exactly as in ase_rename_java.
        if n in variables and n not in methods and re.search(rf"\.\s*{re.escape(n)}(?![\w$])", code):
            held.append(n); continue
        cand.append(n)
    return cand, held


def derange(names: list[str], seed_key: str) -> dict[str, str]:
    """A seeded permutation of `names` onto itself with NO fixed point.

    Sattolo's algorithm produces a single cycle, which is a derangement by construction for n >= 2 --
    so no name keeps its own meaning and no rejection loop is needed.
    """
    if len(names) < 2:
        return {}
    order = sorted(names)                       # deterministic starting order
    rng = random.Random(SEED + zlib.crc32(seed_key.encode()))
    perm = order[:]
    for i in range(len(perm) - 1, 0, -1):       # Sattolo: j strictly < i => one cycle, no fixed points
        j = rng.randrange(i)
        perm[i], perm[j] = perm[j], perm[i]
    mapping = {old: new for old, new in zip(order, perm)}
    assert all(k != v for k, v in mapping.items()), "derangement has a fixed point"
    assert sorted(mapping.values()) == order, "mapping is not a permutation of the same names"
    return mapping


def apply_simultaneous(code: str, mapping: dict[str, str]) -> str:
    """One pass, longest-name-first alternation, so a permutation cannot cascade."""
    if not mapping:
        return code
    keys = sorted(mapping, key=len, reverse=True)
    pat = re.compile(r"(?<![\w$])(" + "|".join(re.escape(k) for k in keys) + r")(?![\w$])")
    return pat.sub(lambda m: mapping[m.group(1)], code)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src-dir", default="/scratch/juno/jvl210002/ase2026/"
                                         "LLM-Attention-Fixation_submission/Source/Humaneval/java")
    ap.add_argument("--out-dir", default="/scratch/juno/jvl210002/ase2026/swap_humaneval_java")
    ap.add_argument("--manifest", default="/scratch/juno/jvl210002/ase2026/swap_manifest.jsonl")
    ap.add_argument("--only", default=None, help="json list of snippet stems to restrict to")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    src = sorted(Path(args.src_dir).glob("*.java"))
    if args.only:
        keep = set(json.loads(Path(args.only).read_text()))
        src = [p for p in src if p.stem in keep]
    if args.limit:
        src = src[: args.limit]
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    man = open(args.manifest, "w")
    n_ok = n_skip = 0
    n_single = 0
    for i, p in enumerate(src, 1):
        original = p.read_text()
        try:
            cand, held = declared_names(original)
        except RuntimeError as e:
            n_skip += 1
            man.write(json.dumps({"snippet": p.stem, "error": str(e)}) + "\n"); continue
        mapping = derange(cand, p.stem)
        if not mapping:
            n_skip += 1
            man.write(json.dumps({"snippet": p.stem, "error": f"only {len(cand)} renameable name(s); "
                                                              "a derangement needs >= 2"}) + "\n")
            continue
        code2 = apply_simultaneous(original, mapping)
        if code2 == original:
            n_skip += 1
            man.write(json.dumps({"snippet": p.stem, "error": "rename was a no-op"}) + "\n"); continue
        had_harness = re.search(HARNESS_RE, original) is not None
        if had_harness and re.search(HARNESS_RE, code2) is None:
            n_skip += 1
            man.write(json.dumps({"snippet": p.stem, "error": "harness pattern destroyed"}) + "\n")
            continue
        # the variant must still PARSE, and must declare exactly the same NAME SET as the original
        try:
            cand2, held2 = declared_names(code2)
        except RuntimeError as e:
            n_skip += 1
            man.write(json.dumps({"snippet": p.stem, "error": f"variant unparseable: {e}"}) + "\n")
            continue
        if sorted(cand2 + held2) != sorted(cand + held):
            n_skip += 1
            man.write(json.dumps({"snippet": p.stem,
                                  "error": "declared-name SET changed; not a pure permutation"}) + "\n")
            continue
        sc = sum(1 for k in mapping if len(k) == 1)
        n_single += sc
        (out / p.name).write_text(code2)
        man.write(json.dumps({"snippet": p.stem, "n_renamed": len(mapping), "n_single_char": sc,
                              "had_harness": had_harness, "held_back": held,
                              "rename_map": mapping}) + "\n")
        n_ok += 1
        if i <= 3 or i % 40 == 0:
            ex = list(mapping.items())[:3]
            print(f"{TAG} {i}/{len(src)} {p.stem} deranged {len(mapping)} names "
                  f"({sc} single-char) e.g. {ex}", flush=True)
    man.close()
    print(f"{TAG} DONE {n_ok} deranged · {n_skip} skipped · {n_single} single-char names renamed "
          f"(the old renamer renamed ZERO of these) · out {out}")
    print(f"{TAG} NOTE compile + label validation happens when the case packs are rebuilt on this dir")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
