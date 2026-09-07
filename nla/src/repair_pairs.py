"""H-W22 — rebuild the original↔renamed identifier correspondence the stimulus pipeline lost.

`?unpairedN` appears as a `rename_map` KEY on 42–97 % of entries depending on tier and language
(`log/nla-harness/2026-09-06_unpaired-sentinel-defect.md`). It means the generator renamed an
identifier and could not recover which original it came from. The information is not gone: the
original names are in the L0 source and the renamed ones are in the tier source, and only the
mapping between them was dropped.

WHY THE OBVIOUS METHOD DOES NOT WORK. Pairing identifiers by position in the identifier-occurrence
sequence needs the two sequences to have equal length. Measured, that holds for 80 % of Python L1
and **10 % of JavaScript L1** — and those rates track the pipeline's own sentinel rates so closely
that equal-length positional matching is very likely what it already tried. Reimplementing it would
recover nothing. JavaScript L1 also rewrites member access (`lst.length` -> `c['length']`), so the
sequences genuinely differ in length and any length-sensitive method fails there by construction.

WHAT THIS DOES INSTEAD. Mask every identifier to a single placeholder, so two files that differ
only by renaming become *identical* token sequences, and align the masked sequences with
`SequenceMatcher`, which tolerates the insertions and deletions that defeat positional matching.
Inside each `equal` block the identifier occurrences correspond one-to-one, and each such
correspondence is a vote. A pair is accepted only when it is the **mutual** majority — the original's
top candidate and the candidate's top original — which prevents a few misaligned blocks from
inventing a mapping.

VALIDATION IS THE POINT, NOT THE COVERAGE. The method is scored on the entries where the pipeline
DID record a real pairing, so its accuracy is measurable against ground truth before any recovered
pairing is trusted. A recovery method with unmeasured accuracy would just be a second, unaudited
source of the same correspondence.
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher

IDENT = re.compile(r"[A-Za-z_$][A-Za-z_0-9$]*")
TOKEN = re.compile(r"[A-Za-z_$][A-Za-z_0-9$]*|\d+\.?\d*|0[xX][0-9a-fA-F]+|\S")

PY_KW = set("""False None True and as assert async await break class continue def del elif else
except finally for from global if import in is lambda nonlocal not or pass raise return try while
with yield print len range str int float list dict set tuple abs min max sum sorted enumerate zip
map filter any all round type isinstance""".split())
JS_KW = set("""var let const function return if else for while do break continue new this typeof
instanceof class extends super import export default null undefined true false of in try catch
finally throw switch case delete void yield async await Math Object Array String Number Boolean
JSON console length push pop slice splice indexOf map filter reduce forEach join split sort reverse
""".split())


def keywords(language: str) -> set[str]:
    return PY_KW if (language or "").lower().startswith("p") else JS_KW


def tokenize(code: str, language: str) -> tuple[list[str], list[int]]:
    """(masked token sequence, indices of the tokens that are identifiers).

    Identifiers collapse to a single `\\x00ID` placeholder so that two files differing only by
    renaming produce byte-identical masked sequences — which is what lets SequenceMatcher find
    long `equal` blocks across a rename.
    """
    kw = keywords(language)
    toks, ids = [], []
    for m in TOKEN.finditer(code):
        t = m.group(0)
        if IDENT.fullmatch(t) and t not in kw:
            ids.append(len(toks))
            toks.append("\x00ID")
        else:
            toks.append(t)
    return toks, ids


def raw_identifiers(code: str, language: str) -> list[str]:
    kw = keywords(language)
    return [m.group(0) for m in IDENT.finditer(code) if m.group(0) not in kw]


def recover(l0_code: str, tier_code: str, language: str,
            min_votes: int = 1, min_align_frac: float = 0.5) -> tuple[dict[str, str], dict]:
    """Recovered {original: renamed}, plus a diagnostic report.

    Only mutual-majority pairs are returned: `orig`'s best candidate must be `cand` AND `cand`'s
    best original must be `orig`. A one-directional majority is exactly how a couple of misaligned
    blocks would manufacture a confident wrong answer.

    `min_align_frac` guards the case mutual majority cannot: **two unrelated programs still share a
    skeleton**. `def X(Y)` aligns against `def X(Y)` whatever the files are, so a unit test on two
    unrelated snippets produced `{total: render, items: self}` with full confidence. Requiring that
    a large fraction of the LARGER file's identifier occurrences land in `equal` blocks refuses that
    case, and refuses the partially-restructured tiers (L3) where the same failure is real rather
    than hypothetical — L3 validation showed precision 0.833 on JavaScript against 1.000 for L1/L1b.
    """
    ta, ia = tokenize(l0_code, language)
    tb, ib = tokenize(tier_code, language)
    na, nb = raw_identifiers(l0_code, language), raw_identifiers(tier_code, language)
    if len(na) != len(ia) or len(nb) != len(ib):
        return {}, {"reason": "tokenizer/identifier count mismatch"}

    pos_a = {p: k for k, p in enumerate(ia)}
    pos_b = {p: k for k, p in enumerate(ib)}
    votes: dict[tuple[str, str], int] = {}
    aligned = 0
    for op, a1, a2, b1, b2 in SequenceMatcher(a=ta, b=tb, autojunk=False).get_opcodes():
        if op != "equal":
            continue
        for off in range(a2 - a1):
            pa, pb = a1 + off, b1 + off
            if pa in pos_a and pb in pos_b:
                aligned += 1
                key = (na[pos_a[pa]], nb[pos_b[pb]])
                votes[key] = votes.get(key, 0) + 1

    denom = max(len(na), len(nb), 1)
    frac = aligned / denom
    if frac < min_align_frac:
        return {}, {"reason": "insufficient alignment", "align_frac": frac,
                    "aligned_identifier_occurrences": aligned, "denom": denom}

    best_fwd: dict[str, tuple[str, int]] = {}
    best_rev: dict[str, tuple[str, int]] = {}
    for (o, r), c in votes.items():
        if c > best_fwd.get(o, ("", 0))[1]:
            best_fwd[o] = (r, c)
        if c > best_rev.get(r, ("", 0))[1]:
            best_rev[r] = (o, c)
    out = {o: r for o, (r, c) in best_fwd.items()
           if c >= min_votes and best_rev.get(r, ("", 0))[0] == o}
    return out, {"aligned_identifier_occurrences": aligned, "candidate_pairs": len(votes),
                 "accepted": len(out), "align_frac": frac}
