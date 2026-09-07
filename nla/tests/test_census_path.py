"""CPU-only checks for the generation path in flippable_census.

Written after job 381293 died at 55 s on `AttributeError` inside `generate()`: the tokenizer's
`apply_chat_template` returns a `BatchEncoding`, not a bare tensor, and the failure surfaced only
as an opaque attribute error three frames deep in transformers.

That was the fourth submit-then-fail round in a row on this cluster, and all four were checkable
without a GPU. On a queue with a ~3:1 wait-to-run ratio, a bug that reaches the scheduler costs
hours; the same bug caught here costs seconds. These tests need the tokenizer only -- no weights.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC))
from steer_run import HOSTS, build_user  # noqa: E402

HOST = "llama8b"


def _tokenizer():
    from transformers import AutoTokenizer
    name, _ = HOSTS[HOST]
    try:
        return AutoTokenizer.from_pretrained(name)
    except Exception as e:                      # noqa: BLE001
        pytest.skip(f"tokenizer for {name} unavailable offline: {e}")


def test_chat_template_yields_a_2d_long_tensor_for_generate():
    """`generate()` reads `inputs_tensor.shape[0]`, so a BatchEncoding fails with a bare
    AttributeError rather than anything that names the problem."""
    tokz = _tokenizer()
    user = build_user("def f(x):\n    return x + 1", "f(1)")
    enc = tokz.apply_chat_template([{"role": "user", "content": user}], tokenize=True,
                                   add_generation_prompt=True, return_tensors="pt",
                                   return_dict=True)
    ids = enc["input_ids"] if hasattr(enc, "keys") else enc
    assert isinstance(ids, torch.Tensor)
    assert ids.dim() == 2 and ids.shape[0] == 1
    assert ids.dtype in (torch.long, torch.int64)


def test_a_pad_token_is_resolvable():
    """Llama has no pad token; generate() must be given eos instead or it warns and can misbehave."""
    tokz = _tokenizer()
    assert (tokz.pad_token_id or tokz.eos_token_id) is not None


def test_the_prompt_actually_contains_the_code_and_the_call():
    """Guards the other half of the census: a prompt that lost its call would score zero for a
    reason that has nothing to do with obfuscation."""
    user = build_user("def add(a, b):\n    return a + b", "add(2, 3)")
    assert "def add(a, b):" in user
    assert "add(2, 3)" in user
    assert "Output:" in user
