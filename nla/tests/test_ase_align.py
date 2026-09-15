"""CPU tests for the Java original<->renamed position alignment (H-R2 NLA arms).

The Python/JS equivalent produced three artifact verdicts in one day from two zero-position bugs, so
the properties that matter are pinned: correspondence is per-occurrence, unequal token counts are
allowed (one vector per span), and any mismatch REFUSES rather than guessing.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ase_align_java import align, char_spans, tokens_for_char_span  # noqa: E402


class FakeTok:
    """Character-level tokenizer: offsets are trivially known, so the test checks alignment logic
    rather than a real tokenizer's quirks."""
    def __call__(self, text, return_offsets_mapping=False, add_special_tokens=False):
        return {"offset_mapping": [(i, i + 1) for i in range(len(text))]}


def test_char_spans_respects_word_boundaries():
    t = "foo fooBar foo_1 foo"
    assert char_spans(t, "foo") == [(0, 3), (17, 20)]


def test_tokens_for_char_span_overlap():
    offs = [(0, 2), (2, 5), (5, 9)]
    assert tokens_for_char_span(offs, 2, 5) == [1]
    assert tokens_for_char_span(offs, 1, 6) == [0, 1, 2]


def test_align_pairs_occurrences_and_allows_unequal_token_counts():
    o = "int aa = f(aa);"
    r = "int zzzz = g(zzzz);"
    a = align("s", o, r, {"aa": "zzzz", "f": "g"}, FakeTok())
    assert a.ok, a.reason
    by = {}
    for s in a.spans:
        by.setdefault(s.name, []).append(s)
    assert len(by["aa"]) == 2 and len(by["f"]) == 1
    # unequal token counts are expected and fine: one vector per span is written
    assert len(by["aa"][0].orig_tokens) == 2 and len(by["aa"][0].ren_tokens) == 4


def test_align_refuses_on_occurrence_mismatch():
    a = align("s", "int aa = aa;", "int zz = 5;", {"aa": "zz"}, FakeTok())
    assert not a.ok and "occurrence mismatch" in a.reason


def test_align_refuses_when_the_method_name_was_not_renamed():
    a = align("s", "int aa;", "int zz;", {"aa": "zz"}, FakeTok(), method_name="doThing")
    assert not a.ok and "was not renamed" in a.reason


def test_align_refuses_when_nothing_resolves():
    a = align("s", "int x;", "int y;", {}, FakeTok())
    assert not a.ok and a.reason == "no spans resolved"
