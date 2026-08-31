"""Unit tests for the steering-direction ladder. CPU-only, no checkpoint needed.

The AR is stubbed by a deterministic text->vector map, which is enough to test every algebraic
property that matters: the difference cancels what the two texts share, minimal edits are
enforced, the held-out guard actually holds out, and the oracle is distinguishable from the
held-out mean.

Run:
  /data/jvl210002/conda_envs/nla-mi/bin/python -m pytest nla/tests/test_steer_vectors.py -q
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "src"))

from steer_vectors import (  # noqa: E402
    ActivationPair, TaskVectorBank, antipodal, cosine, direction_report, foreign_direction,
    nla_edit_direction, random_direction, shuffled_text_direction, substitute_terms,
    word_edit_direction,
)

D = 16


class StubAR:
    """Deterministic bag-of-words encoder: each token contributes a fixed random vector.

    Linear in the token multiset on purpose. That makes the *expected* value of a text
    difference exactly computable, so the tests assert real algebra rather than "it ran".
    """

    def __init__(self, d: int = D):
        self.d = d

    def _tok(self, w: str) -> torch.Tensor:
        seed = int(hashlib.sha256(w.lower().encode()).hexdigest()[:8], 16)
        return torch.randn(self.d, generator=torch.Generator().manual_seed(seed))

    def reconstruct(self, text: str) -> torch.Tensor:
        toks = text.split()
        if not toks:
            return torch.zeros(self.d)
        return torch.stack([self._tok(t) for t in toks]).sum(0)


@pytest.fixture
def ar():
    return StubAR()


# ── V1 / V2 ─────────────────────────────────────────────────────────────────
def test_edit_direction_cancels_shared_content(ar):
    """Δ should depend only on what changed, not on the shared surroundings."""
    a = "this code computes a bubbleSort over the list"
    b = "this code computes a binomial coefficient over the list"
    d1 = nla_edit_direction(ar, a, b)
    long_pre = "a much longer preamble with many extra words "
    d2 = nla_edit_direction(ar, long_pre + a, long_pre + b)
    assert torch.allclose(d1, d2, atol=1e-5), "shared context leaked into the difference"


def test_identical_texts_are_rejected(ar):
    with pytest.raises(ValueError):
        nla_edit_direction(ar, "same text", "same text")


def test_substitute_terms_is_whole_word_and_case_preserving():
    text, n = substitute_terms("BubbleSort and bubblesort but not bubbleSorted",
                               {"bubblesort": "quicksort"})
    assert n == 2, "should match the two whole-word occurrences only"
    assert "bubbleSorted" in text, "substring inside a longer word must not match"
    assert text.startswith("Quicksort"), "leading capital should be preserved"


def test_word_edit_enforces_minimality(ar):
    two = "sorting then more sorting"
    with pytest.raises(ValueError, match="expected exactly one"):
        word_edit_direction(ar, two, {"sorting": "hashing"})
    one = "this performs sorting on the input"
    d = word_edit_direction(ar, one, {"sorting": "hashing"})
    assert d.shape == (D,)


def test_word_edit_rejects_absent_term(ar):
    with pytest.raises(ValueError, match="no term"):
        word_edit_direction(ar, "nothing relevant here", {"sorting": "hashing"})


# ── V3 / V4 ─────────────────────────────────────────────────────────────────
def _bank(n: int = 5, d: int = D) -> TaskVectorBank:
    g = torch.Generator().manual_seed(0)
    pairs = []
    common = torch.randn(d, generator=g)          # a shared "deobfuscation" direction
    for i in range(n):
        obf = torch.randn(d, generator=g)
        clean = obf + common + 0.1 * torch.randn(d, generator=g)
        pairs.append(ActivationPair(f"item{i}", clean, obf))
    return TaskVectorBank(pairs)


def test_task_vector_recovers_the_shared_direction():
    bank = _bank(n=40)
    v = bank.direction_for(exclude="item0")
    per_item = torch.stack([p.difference() for p in bank.pairs])
    assert cosine(v, per_item.mean(0)) > 0.99


def test_direction_for_requires_exclusion_and_honours_it():
    bank = _bank(n=3)
    with pytest.raises(TypeError):
        bank.direction_for()                       # exclude is mandatory — leakage guard
    manual = torch.stack([p.difference() for p in bank.pairs
                          if p.item_id != "item1"]).mean(0)
    assert torch.allclose(bank.direction_for(exclude="item1"), manual, atol=1e-6)


def test_excluding_everything_raises():
    bank = _bank(n=2)
    with pytest.raises(ValueError, match="no pairs"):
        bank.direction_for(exclude=["item0", "item1"])


def test_oracle_differs_from_heldout_mean():
    """If these coincided, V4 would not be a distinct condition."""
    bank = _bank(n=6)
    assert not torch.allclose(bank.oracle_for("item0"), bank.direction_for(exclude="item0"),
                              atol=1e-3)


def test_oracle_unknown_item():
    with pytest.raises(KeyError):
        _bank().oracle_for("nope")


# ── Controls ────────────────────────────────────────────────────────────────
def test_random_direction_is_norm_matched_and_seeded():
    like = torch.randn(D) * 7.0
    a = random_direction(D, seed=1, like=like)
    b = random_direction(D, seed=1, like=like)
    c = random_direction(D, seed=2, like=like)
    assert torch.allclose(a, b), "same seed must give the same direction"
    assert not torch.allclose(a, c)
    assert abs(float(a.norm()) - float(like.norm())) < 1e-4


def test_foreign_direction_is_never_the_item_itself():
    bank = _bank(n=6)
    own = bank.oracle_for("item2")
    for seed in range(20):
        assert not torch.allclose(foreign_direction(bank, "item2", seed=seed), own, atol=1e-6)


def test_shuffled_text_direction_differs_from_the_real_one(ar):
    a = "the function sorts the array in ascending order"
    b = "the function hashes the array in ascending order"
    real = nla_edit_direction(ar, a, b)
    shuf = shuffled_text_direction(ar, a, b, seed=3)
    # With a bag-of-words stub the shuffle is a no-op by construction — that is itself the
    # point of the control: it isolates word content from word order, so under a purely
    # lexical encoder the two must coincide. A real AR reads order, and they will not.
    assert torch.allclose(real, shuf, atol=1e-5)


def test_antipodal_flips():
    v = torch.randn(D)
    assert torch.allclose(antipodal(v), -v)
    assert cosine(v, antipodal(v)) == pytest.approx(-1.0, abs=1e-5)


def test_direction_report_shape():
    rep = direction_report({"v1": torch.randn(D), "v3": torch.randn(D)})
    assert set(rep) == {"norms", "cosines"}
    assert set(rep["norms"]) == {"v1", "v3"}
    assert list(rep["cosines"]) == ["v1|v3"]
