"""Construct steering directions, and the controls that make a steering claim mean anything.

Five sources, deliberately arranged as a ladder rather than a menu:

  V1  nla_edit      AR(edited explanation) - AR(original explanation)     the NLA paper's method
  V2  word_edit     same, but exactly one word differs                    does one lexical item carry it
  V3  task_vector   mean h_clean - mean h_obf over OTHER matched pairs    needs no NLA at all
  V4  item_oracle   h_clean(this item) - h_obf(this item)                 upper bound, NOT deployable
  V5  combined      V1 stacked on attention steering                      do the channels compose

**V3 is the row the paper lives or dies on.** It is a plain contrastive difference of
activations — no verbalizer, no reconstructor, no explanation. If V3 matches V1, the NLA is
buying interpretability but not capability. That is a perfectly publishable finding, but it is
a different paper, and it must be discovered by us rather than by a reviewer. The pre-registered
rule is: **V1 must beat V3 to claim the NLA is doing causal work.** This mirrors the discipline
`ar_baseline.py` already encodes for the alignment score, where a free dense baseline was
pre-registered precisely so that "the judge works" could not quietly mean "the judge beats
chance".

**V4 uses the clean program**, which the deployed setting would not have. It is an oracle
ceiling and every table must label it as one; reporting it beside V1/V3 without that label
would be straightforwardly misleading.

**Leakage.** V3 is fit on pairs and applied to items. If the applied item is inside the fitting
set, the vector has seen its answer. `TaskVectorBank.direction_for` therefore *requires* an
`exclude` argument and raises without it — the guard is in the type signature, not in a comment,
because this is the kind of thing that is easy to forget once and impossible to spot afterwards.

The AR is injected as a protocol (anything with `.reconstruct(text) -> Tensor`), so every
constructor here is unit-testable on CPU with a stub and no checkpoint.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Mapping, Protocol, Sequence

import torch


class Reconstructor(Protocol):
    """The AR's text -> vector map. `nla_inference.NLACritic` satisfies this."""

    def reconstruct(self, text: str) -> torch.Tensor: ...


def _unit(v: torch.Tensor) -> torch.Tensor:
    return v / v.norm().clamp_min(1e-12)


# ── V1 / V2: directions derived from an edited explanation ──────────────────
def nla_edit_direction(ar: Reconstructor, original: str, edited: str) -> torch.Tensor:
    """Δ = AR(edited) − AR(original), the paper's construction.

    Both texts go through the same reconstructor, so anything shared between them — style,
    length, the AV's house phrasing — cancels in the difference. What survives is the edit.
    """
    if original == edited:
        raise ValueError("edited text is identical to the original; Δ would be zero")
    return ar.reconstruct(edited).float() - ar.reconstruct(original).float()


def substitute_terms(text: str, mapping: Mapping[str, str]) -> tuple[str, int]:
    """Mechanical, whole-word, case-preserving substitution. Returns (text, n_substitutions).

    Mechanical rather than an LLM rewrite on purpose: it is reproducible under seed, it cannot
    smuggle in a second model's judgement, and the number of edits is countable — which matters
    because V2's claim is precisely that *one* word was changed.
    """
    n = 0

    def repl(m: re.Match) -> str:
        nonlocal n
        n += 1
        rep = mapping[m.group(0).lower()]
        if m.group(0).isupper():
            return rep.upper()
        if m.group(0)[:1].isupper():
            return rep[:1].upper() + rep[1:]
        return rep

    if not mapping:
        return text, 0
    pattern = re.compile(r"\b(" + "|".join(re.escape(k) for k in sorted(mapping, key=len,
                                                                       reverse=True)) + r")\b",
                         re.I)
    lowered = {k.lower(): v for k, v in mapping.items()}
    mapping = lowered
    return pattern.sub(repl, text), n


def word_edit_direction(ar: Reconstructor, original: str, mapping: Mapping[str, str],
                        require_single: bool = True) -> torch.Tensor:
    """V2: Δ from a minimal lexical edit. Fails loudly if the edit was not minimal."""
    edited, n = substitute_terms(original, mapping)
    if n == 0:
        raise ValueError(f"no term in {list(mapping)} occurs in the explanation")
    if require_single and n != 1:
        raise ValueError(f"expected exactly one substitution, made {n} — not a minimal edit")
    return nla_edit_direction(ar, original, edited)


# ── V3 / V4: contrastive activation directions (no NLA involved) ────────────
@dataclass(frozen=True)
class ActivationPair:
    """One program in both conditions, at the same read site."""
    item_id: str
    clean: torch.Tensor
    obfuscated: torch.Tensor

    def difference(self) -> torch.Tensor:
        return self.clean.float() - self.obfuscated.float()


class TaskVectorBank:
    """V3: a held-out mean contrastive direction, `mean(h_clean − h_obf)`.

    Means are taken over the *difference*, not of each condition separately, so the pairing is
    used rather than thrown away — the paired estimator removes any per-program offset common to
    both conditions, which is most of the variance.
    """

    def __init__(self, pairs: Sequence[ActivationPair]):
        if not pairs:
            raise ValueError("need at least one pair")
        self.pairs = list(pairs)
        self._by_id = {p.item_id: p for p in self.pairs}

    def direction_for(self, exclude: str | Iterable[str]) -> torch.Tensor:
        """Mean difference over every pair EXCEPT the excluded item(s).

        `exclude` is mandatory: applying a direction fitted on the same item is leakage, and it
        would inflate every steering number in the paper. Pass the item you are about to steer.
        """
        drop = {exclude} if isinstance(exclude, str) else set(exclude)
        kept = [p for p in self.pairs if p.item_id not in drop]
        if not kept:
            raise ValueError("excluding those items leaves no pairs to fit on")
        return torch.stack([p.difference() for p in kept]).mean(0)

    def oracle_for(self, item_id: str) -> torch.Tensor:
        """V4: this item's own clean−obfuscated difference. Oracle ceiling, not deployable."""
        if item_id not in self._by_id:
            raise KeyError(f"no pair for {item_id}")
        return self._by_id[item_id].difference()

    def __len__(self) -> int:
        return len(self.pairs)


# ── Controls ────────────────────────────────────────────────────────────────
def random_direction(d_model: int, seed: int, like: torch.Tensor | None = None) -> torch.Tensor:
    """Norm-matched random direction. The floor every real Δ must clear."""
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(d_model, generator=g)
    return _unit(v) * (like.norm() if like is not None else 1.0)


def foreign_direction(bank: TaskVectorBank, item_id: str, seed: int) -> torch.Tensor:
    """A real difference vector from a DIFFERENT item.

    The sharpest control: it has the same statistics, the same construction and the same norm
    as the true direction, and differs only in being about another program. If steering works
    equally with a foreign Δ, the effect is not about this item's content.
    """
    others = [p for p in bank.pairs if p.item_id != item_id]
    if not others:
        raise ValueError("no foreign item available")
    g = torch.Generator().manual_seed(seed)
    pick = others[int(torch.randint(len(others), (1,), generator=g))]
    return pick.difference()


def shuffled_text_direction(ar: Reconstructor, original: str, edited: str,
                            seed: int) -> torch.Tensor:
    """Δ from word-shuffled versions of both texts.

    Separates sentence meaning from lexical content: the same words are present, the same edit
    is present, but the syntax is destroyed. A direction that survives shuffling is carried by
    the bag of words, not by what the explanation says.
    """
    g = torch.Generator().manual_seed(seed)

    def shuffle(t: str) -> str:
        w = t.split()
        idx = torch.randperm(len(w), generator=g).tolist()
        return " ".join(w[i] for i in idx)

    return nla_edit_direction(ar, shuffle(original), shuffle(edited))


def antipodal(delta: torch.Tensor) -> torch.Tensor:
    """−Δ. Must push behaviour the other way, or the direction is not directional."""
    return -delta


# ── Reporting helpers ───────────────────────────────────────────────────────
def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(_unit(a.float()) @ _unit(b.float()))


def direction_report(named: Mapping[str, torch.Tensor]) -> dict[str, dict[str, float]]:
    """Pairwise cosines + norms across the ladder.

    Worth printing before any steering run: if V1 and V3 are near-collinear, the two conditions
    are not independent tests and the V1>V3 comparison is measuring noise between two names for
    the same direction.
    """
    keys = list(named)
    return {
        "norms": {k: float(named[k].norm()) for k in keys},
        "cosines": {f"{a}|{b}": cosine(named[a], named[b])
                    for i, a in enumerate(keys) for b in keys[i + 1:]},
    }
