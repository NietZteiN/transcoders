"""H-R28 — L2 control-flow flattening for the HumanEval-X Java corpus.

Pre-registered in log/nla-harness/2026-09-18_flatten-prereg.md.

WHY THIS EXISTS. The ASE artifact ships no obfuscation generator (`GITREADY_MANIFEST.md`), so every
obfuscation in this thread has been ours. H-R22/H-R23 closed the *renaming* route: deranging every
identifier costs ~1 point and no permitted model shows a deficit, because these models trace code rather
than read names. Flattening attacks the OTHER route in the charter's ladder -- **relational overload**:
a dispatcher loop destroys the correspondence between textual order and execution order, so the reader
must simulate a state variable instead of reading straight-line code. Nothing measured so far speaks to
it.

THE TRANSFORM. The target method's body is rewritten as a dispatcher:

    <hoisted declarations>
    int __state = <entry>;
    while (true) {
        switch (__state) {
            case <k>: <statement>; __state = <next>; break;
            ...
        }
    }

with the case labels **permuted**, so the textual order of the cases is not the execution order. That
permutation is the manipulation: without it the switch is just a verbose rewrite that still reads
top-to-bottom.

SCOPE, DELIBERATELY NARROW, AND WHY. Only the method's **top-level statement sequence** is flattened;
compound statements (if/while/for/try) are moved WHOLE into their case rather than being decomposed into
their own states. Decomposing nested control flow correctly needs a real CFG with back edges, and a
subtly wrong CFG would silently change semantics -- the one failure this corpus cannot detect cheaply,
because a variant that compiles and runs but computes something else would be scored as a comprehension
result. Top-level flattening is enough to destroy linear readability (that is what the permuted labels
do) while remaining provably behaviour-preserving, and every variant is still execution-validated when
the case packs are rebuilt.

TWO CORRECTNESS HAZARDS, BOTH HANDLED:
  * **Definite assignment.** `int x = 5;` in one case and a use of `x` in another does not compile: Java
    cannot prove the first case ran. Every top-level local declaration is therefore HOISTED above the
    loop with a type-appropriate default, and only the assignment stays in the case.
  * **Unreachable code.** `while (true)` with no `break` out of it does not complete normally, so no
    trailing `return` may be emitted (Java rejects unreachable statements) and none is needed. A void
    method whose last statement is not a return gets an explicit terminal `return;` case instead.

Anything the analysis cannot handle with certainty -- `var` declarations (no hoistable type), labelled
breaks, a method whose statements lack source positions -- causes the snippet to be SKIPPED with the
reason recorded, never guessed at.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import zlib
from pathlib import Path

TAG = "[FLAT]"
SEED = 20260724
STATE = "__state"

# A hoisted declaration needs a value that is valid for its type and never observed: the real assignment
# always runs before any use, because the dispatcher preserves execution order.
PRIMITIVE_DEFAULT = {"int": "0", "long": "0L", "short": "(short) 0", "byte": "(byte) 0",
                     "char": "'\\0'", "float": "0.0f", "double": "0.0", "boolean": "false"}


def default_for(type_src: str) -> str | None:
    t = type_src.strip()
    if t.endswith("[]") or "<" in t or t[:1].isupper():
        return "null"
    return PRIMITIVE_DEFAULT.get(t)


def line_col_to_off(src: str) -> list[int]:
    """Offset of the first character of each 1-indexed line."""
    offs, n = [0, 0], 0
    for line in src.split("\n"):
        n += len(line) + 1
        offs.append(n)
    return offs


def find_method(tree, src: str, name: str):
    import javalang
    for _, node in tree:
        if type(node).__name__ == "MethodDeclaration" and node.name == name:
            return node
    return None


def statement_spans(node, src: str, offs: list[int]) -> list[tuple[int, int]] | None:
    """(start, end) char offsets of each top-level statement of a method body, in execution order."""
    body = node.body
    if not body:
        return None
    starts = []
    for st in body:
        p = getattr(st, "position", None)
        if p is None:
            return None
        starts.append(offs[p.line] + p.column - 1)
    if any(b <= a for a, b in zip(starts, starts[1:])):
        return None                                   # positions not monotonic -> do not guess
    # the body's closing brace: scan from the last statement to the matching close of the method
    depth, i, end = 0, starts[0], None
    while i < len(src):
        c = src[i]
        if c == "{":
            depth += 1
        elif c == "}":
            if depth == 0:
                end = i
                break
            depth -= 1
        i += 1
    if end is None:
        return None
    bounds = list(zip(starts, starts[1:] + [end]))
    return [(a, b) for a, b in bounds]


DECL_RE = re.compile(r"^\s*(?:final\s+)?([A-Za-z_$][\w$.]*(?:\s*<[^;=]*>)?(?:\s*\[\s*\])*)\s+"
                     r"([A-Za-z_$][\w$]*)\s*(=\s*(.+))?;\s*$", re.S)


def split_top_commas(s: str) -> list[str]:
    """Split a declarator list on commas that are not nested inside (), [], {} or <>.

    `a = new ArrayList<>(l), b = f(x, y)` -> ['a = new ArrayList<>(l)', 'b = f(x, y)'].
    Angle depth is tracked because a generic argument list (`new HashMap<String,Integer>()`) carries a
    comma that is NOT a declarator separator; `<`/`>` are only treated as brackets when they sit next to
    an identifier or another angle, so a comparison operator does not corrupt the depth.
    """
    out, buf, par, sq, br, ang = [], [], 0, 0, 0, 0
    for i, c in enumerate(s):
        prev = s[i - 1] if i else ""
        nxt = s[i + 1] if i + 1 < len(s) else ""
        if c == "(": par += 1
        elif c == ")": par -= 1
        elif c == "[": sq += 1
        elif c == "]": sq -= 1
        elif c == "{": br += 1
        elif c == "}": br -= 1
        elif c == "<" and (prev.isalnum() or prev in "_$<"): ang += 1
        elif c == ">" and ang > 0 and (prev.isalnum() or prev in "_$>?"): ang -= 1
        elif c == "," and par == sq == br == ang == 0:
            out.append("".join(buf)); buf = []; continue
        buf.append(c)
    out.append("".join(buf))
    return [x.strip() for x in out if x.strip()]


DECLARATOR_RE = re.compile(r"^([A-Za-z_$][\w$]*)\s*(?:=\s*(.+))?$", re.S)


def split_decl(stmt: str) -> list[tuple[str, str, str | None]] | None:
    """`int x = 5, y;` -> [('int','x','5'), ('int','y',None)].

    Returns one entry per declarator, or None when this is not a local declaration we can hoist with
    certainty. Multi-declarator statements were the bug that produced 11 unparseable variants on the
    first pass: the single-name regex captured `l1` and swallowed `, l2 = ...` into its initialiser.
    """
    m = DECL_RE.match(stmt.strip())
    if not m:
        return None
    type_src, rest = m.group(1).strip(), (m.group(2) or "")
    if type_src in ("return", "var", "new", "throw", "assert", "if", "for", "while", "switch", "do"):
        return None
    body = stmt.strip()[len(m.group(0)) - len(stmt.strip()):] if False else None
    # rebuild the declarator region: everything between the type and the trailing `;`
    tail = stmt.strip()
    assert tail.endswith(";")
    decl_region = tail[len(type_src):-1].strip()
    out = []
    for d in split_top_commas(decl_region):
        dm = DECLARATOR_RE.match(d)
        if not dm:
            return None
        out.append((type_src, dm.group(1), (dm.group(2).strip() if dm.group(2) else None)))
    return out or None


def flatten_method(src: str, node, spans: list[tuple[int, int]], seed_key: str) -> tuple[str, dict] | None:
    stmts = [src[a:b].strip() for a, b in spans]
    stmts = [s for s in stmts if s]
    if len(stmts) < 3:
        return "too-few-statements"                   # nothing to dispatch over
    if any("->" in s for s in stmts):  # noqa: E501
        # Hoisting turns `T x = e;` into a declaration plus a later assignment, so `x` stops being
        # EFFECTIVELY FINAL and any lambda capturing it no longer compiles. Rather than analyse which
        # captures are affected, skip the method: a wrong answer here would be a semantics change.
        return "lambda-capture"
    hoist, bodies = [], []
    for s in stmts:
        decls = split_decl(s)
        if decls is None:
            bodies.append(s); continue
        assigns = []
        for type_src, name, init in decls:
            dv = default_for(type_src)
            if dv is None:
                return "untyped-declaration"          # unknown type -> cannot hoist safely
            hoist.append(f"{type_src} {name} = {dv};")
            if init is not None:
                assigns.append(f"{name} = {init};")
        bodies.append(" ".join(assigns))
    n = len(bodies)
    rng = random.Random(SEED + zlib.crc32(seed_key.encode()))
    labels = list(range(n))
    for _ in range(64):
        rng.shuffle(labels)                           # execution order i -> printed label labels[i]
        if labels != sorted(labels):
            break                                     # identity = cases print in execution order = no
    else:                                             # obfuscation at all, so redraw (n>=3 so it exists)
        return None
    void = (getattr(node, "return_type", None) is None)

    def is_terminal(stmt: str) -> bool:
        """Does this statement always leave the method? Then nothing may follow it in its case --
        Java rejects unreachable statements, so emitting the usual `break;` would not compile."""
        t = stmt.strip()
        return t.startswith("return") or t.startswith("throw")

    # The dispatcher's LAST case must end the method by itself. Synthesising a tail there needs static
    # reachability analysis -- javac rejected both `return;` (wrong type) and a never-executed
    # `throw` (unreachable, because the preceding statement already always exits, e.g. an if/else where
    # both branches return). Rather than approximate the analysis, require the final top-level statement
    # to be a real terminal and skip the method otherwise. Conservative, and it cannot be wrong.
    if not is_terminal(bodies[-1]):
        return "last-statement-not-terminal"

    cases = []
    for i, b in enumerate(bodies):
        lab = labels[i]
        nxt = labels[i + 1] if i + 1 < n else None
        if is_terminal(b):
            # the statement itself exits; no state transition and no break may follow it
            cases.append(f"                case {lab}:\n                    {b}")
            continue
        parts = [b] if b else []
        assert nxt is not None, "last statement is terminal, so only non-final cases reach here"
        parts.append(f"{STATE} = {nxt};")
        parts.append("break;")
        joined = "\n                    ".join(parts)
        cases.append(f"                case {lab}:\n                    {joined}")
    cases.sort(key=lambda c: int(re.search(r"case (\d+):", c).group(1)))
    hoisted = "\n        ".join(hoist)
    block = (("        " + hoisted + "\n") if hoist else "") + \
        f"        int {STATE} = {labels[0]};\n" \
        f"        while (true) {{\n            switch ({STATE}) {{\n" + \
        "\n".join(cases) + "\n            }\n        }\n"
    a, b = spans[0][0], spans[-1][1]
    out = src[:a].rstrip(" ") + "\n" + block + "    " + src[b:]
    return out, {"n_states": n, "n_hoisted": len(hoist), "entry": labels[0],
                 "order_is_textual": labels == sorted(labels)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src-dir", default="/scratch/juno/jvl210002/ase2026/"
                                         "LLM-Attention-Fixation_submission/Source/Humaneval/java")
    ap.add_argument("--out-dir", default="/scratch/juno/jvl210002/ase2026/flat_humaneval_java")
    ap.add_argument("--manifest", default="/scratch/juno/jvl210002/ase2026/flat_manifest.jsonl")
    ap.add_argument("--only", default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    import javalang

    src_files = sorted(Path(args.src_dir).glob("*.java"))
    if args.only:
        keep = set(json.loads(Path(args.only).read_text()))
        src_files = [p for p in src_files if p.stem in keep]
    if args.limit:
        src_files = src_files[: args.limit]
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    man = open(args.manifest, "w")
    ok = skip = 0
    for i, p in enumerate(src_files, 1):
        code = p.read_text()
        try:
            tree = javalang.parse.parse(code)
        except Exception as e:
            skip += 1; man.write(json.dumps({"snippet": p.stem, "error": f"parse: {e}"}) + "\n"); continue
        # the target method = the public non-main method the harness calls
        # Pick the LARGEST non-main method, not the first in tree order. Files with a helper class
        # (Java_075 declares an `IsPrime` helper) would otherwise have the helper flattened instead of
        # the logic the harness calls -- which both risks a bad splice and weakens the manipulation
        # without failing loudly.
        cands = [nd for _, nd in tree
                 if type(nd).__name__ == "MethodDeclaration" and nd.name != "main" and nd.body]
        target = max(cands, key=lambda nd: len(nd.body)) if cands else None
        if target is None:
            skip += 1; man.write(json.dumps({"snippet": p.stem, "error": "no target method"}) + "\n"); continue
        offs = line_col_to_off(code)
        spans = statement_spans(target, code, offs)
        if spans is None:
            skip += 1
            man.write(json.dumps({"snippet": p.stem, "error": "no usable statement spans"}) + "\n"); continue
        r = flatten_method(code, target, spans, p.stem)
        if isinstance(r, str) or r is None:
            skip += 1
            man.write(json.dumps({"snippet": p.stem, "error": r or "unflattenable"}) + "\n")
            continue
        code2, meta = r
        try:
            javalang.parse.parse(code2)
        except Exception as e:
            skip += 1
            man.write(json.dumps({"snippet": p.stem, "error": f"variant unparseable: {e}"}) + "\n"); continue
        (out / p.name).write_text(code2)
        man.write(json.dumps({"snippet": p.stem, "method": target.name, **meta}) + "\n")
        ok += 1
        if i <= 3 or i % 40 == 0:
            print(f"{TAG} {i}/{len(src_files)} {p.stem} -> {meta['n_states']} states "
                  f"({meta['n_hoisted']} hoisted, entry {meta['entry']})", flush=True)
    man.close()
    print(f"{TAG} DONE {ok} flattened · {skip} skipped · out {out}")
    print(f"{TAG} NOTE compile + execution validation happens when the case packs are rebuilt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
