"""Token-position alignment between the ORIGINAL and RENAMED Java prompts — the dependency every
NLA-style residual arm has.

A residual write needs to put the original program's activation at the renamed program's identifier
positions. The two prompts are not positionally comparable: decoy names tokenise to different lengths,
and because the case packs are rebuilt per variant the *instruction* differs too (every case expression
carries the renamed method name). So the correspondence has to be established explicitly.

This is the Java counterpart of `repair_pairs.recover`, which for Python/JS took two zero-position bugs
(defects #8 and #9) and three artifact verdicts in one day before it was right. The lesson is encoded
here as a refusal rather than a warning: a snippet whose spans do not correspond exactly is EXCLUDED,
never written with a best-effort guess.

HOW: for each (original name -> decoy name) pair in the snippet's rename map, find every
word-boundary occurrence in each prompt's text, map char spans to token indices via the tokenizer's
offset mapping, and pair the i-th occurrence in the original with the i-th in the renamed prompt.

ONE VECTOR PER SPAN, which is why unequal token counts are fine. A decoy name may tokenise to a
different number of tokens than the original; the write takes a single vector per span (the mean over
the original span's tokens) and applies it to every token of the renamed span -- exactly what
`nla_erasure.targets()` does with `tg[q] = vt` for the Python/JS work.

GATES (all must pass or the snippet is dropped, with the reason recorded):
  * every renamed identifier occurs the same number of times in both prompts;
  * every occurrence maps to at least one token in both;
  * the method name -- the manipulation that puts a decoy inside every case expression -- is present.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SpanPair:
    name: str
    decoy: str
    orig_tokens: list[int] = field(default_factory=list)
    ren_tokens: list[int] = field(default_factory=list)


@dataclass
class Alignment:
    snippet: str
    ok: bool
    reason: str = ""
    spans: list[SpanPair] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"snippet": self.snippet, "ok": self.ok, "reason": self.reason,
                "n_spans": len(self.spans),
                "spans": [{"name": s.name, "decoy": s.decoy,
                           "n_orig_tok": len(s.orig_tokens), "n_ren_tok": len(s.ren_tokens),
                           "orig_tokens": s.orig_tokens, "ren_tokens": s.ren_tokens}
                          for s in self.spans]}


def char_spans(text: str, name: str) -> list[tuple[int, int]]:
    return [(m.start(), m.end())
            for m in re.finditer(rf"(?<![\w$]){re.escape(name)}(?![\w$])", text)]


def tokens_for_char_span(offsets: list[tuple[int, int]], a: int, b: int) -> list[int]:
    """Token indices whose character range overlaps [a, b)."""
    return [i for i, (s, e) in enumerate(offsets) if e > a and s < b and e > s]


def align(snippet: str, orig_prompt: str, ren_prompt: str, rename_map: dict[str, str],
          tok: Any, method_name: str | None = None) -> Alignment:
    eo = tok(orig_prompt, return_offsets_mapping=True, add_special_tokens=False)
    er = tok(ren_prompt, return_offsets_mapping=True, add_special_tokens=False)
    off_o = [tuple(x) for x in eo["offset_mapping"]]
    off_r = [tuple(x) for x in er["offset_mapping"]]

    spans: list[SpanPair] = []
    for name, decoy in sorted(rename_map.items(), key=lambda kv: len(kv[0]), reverse=True):
        co = char_spans(orig_prompt, name)
        cr = char_spans(ren_prompt, str(decoy))
        if len(co) != len(cr):
            return Alignment(snippet, False,
                             f"occurrence mismatch for {name}->{decoy}: {len(co)} vs {len(cr)}")
        for (a0, b0), (a1, b1) in zip(co, cr):
            to = tokens_for_char_span(off_o, a0, b0)
            tr = tokens_for_char_span(off_r, a1, b1)
            if not to or not tr:
                return Alignment(snippet, False, f"empty token span for {name}->{decoy}")
            spans.append(SpanPair(name=name, decoy=str(decoy), orig_tokens=to, ren_tokens=tr))
    if not spans:
        return Alignment(snippet, False, "no spans resolved")
    if method_name and method_name not in rename_map:
        return Alignment(snippet, False, f"method name {method_name} was not renamed")
    return Alignment(snippet, True, "", spans)
