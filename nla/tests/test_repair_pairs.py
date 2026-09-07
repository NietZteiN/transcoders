"""Tests for the identifier-pairing repair. CPU-only, no model, no stimuli.

The method's value rests entirely on being checkable, so these test the properties that would
otherwise let a wrong mapping through quietly.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from repair_pairs import recover, tokenize  # noqa: E402


def test_pure_rename_is_recovered_exactly():
    a = "def total(items):\n    acc = 0\n    for it in items:\n        acc = acc + it\n    return acc"
    b = "def qq(zz):\n    ww = 0\n    for vv in zz:\n        ww = ww + vv\n    return ww"
    got, rep = recover(a, b, "python")
    assert got == {"total": "qq", "items": "zz", "acc": "ww", "it": "vv"}
    assert rep["accepted"] == 4


def test_survives_an_insertion_that_defeats_positional_matching():
    """The JavaScript L1 failure mode: the tier adds tokens, so the identifier sequences differ in
    length and any equal-length positional method fails by construction."""
    a = "function f(lst) { return lst.length }"
    b = "function g(cc) { const extra = 1; return cc['length'] }"
    got, _ = recover(a, b, "javascript")
    assert got.get("f") == "g" and got.get("lst") == "cc"


def test_masking_makes_a_renamed_file_token_identical():
    a, _ = tokenize("def total(items): return items", "python")
    b, _ = tokenize("def qq(zz): return zz", "python")
    assert a == b


def test_mutual_majority_refuses_a_one_directional_pairing():
    """Two originals cannot both claim one renamed identifier; the loser must be dropped rather
    than guessed, which is how a few misaligned blocks would manufacture a confident wrong map."""
    a = "x = 1\ny = 2\nz = x + y"
    b = "q = 1\nq2 = 2\nq3 = q + q2"
    got, _ = recover(a, b, "python")
    assert len(set(got.values())) == len(got)      # injective


def test_unrelated_programs_yield_few_or_no_pairs():
    a = "def total(items): return sum(items)"
    b = "class Widget:\n    def render(self, canvas):\n        canvas.draw(self)"
    got, _ = recover(a, b, "python")
    assert len(got) <= 1


def test_identical_code_maps_every_identifier_to_itself():
    a = "def total(items):\n    return sum(items)"
    got, _ = recover(a, a, "python")
    assert got == {"total": "total", "items": "items"}
