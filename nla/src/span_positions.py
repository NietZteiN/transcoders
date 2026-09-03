"""Map identifier character spans in the source code to token positions in the built prompt.

WHY. Every steering result this project has produced wrote at ONE token — `last_prompt`, the
final prompt token. Coverage was never varied except once (P0.2, "all reply positions"), which
destroyed generation because coverage widened at unchanged alpha. The stimuli have carried 11
annotated `identifier_spans` per item since Paper 2 and NOT ONE run has used them.

That matters for the adversarial-rename hypothesis specifically. The claim is that decoy
identifiers inject wrong semantics. If so, the natural place to intervene is AT THE IDENTIFIER
TOKENS, not at a single position after the whole program has been read. `last_prompt` is where
the answer is about to be produced; the identifier sites are where the alleged interference
enters. These are different hypotheses and only one of them has ever been tested.

THE MAPPING IS THE WHOLE RISK. Spans are character offsets into `code`; the model sees a chat
template wrapping a preamble + code + question. An off-by-anything silently steers the wrong
tokens and the run still produces plausible numbers. So the mapping is computed with the
tokenizer's own `offset_mapping` against the FULL rendered prompt string, and every returned
position is verified to decode back to text overlapping the intended span. Items whose spans
cannot be verified are reported and skipped, never silently steered at position 0.
"""
from __future__ import annotations

from typing import Any, Sequence


def _rendered_prompt(tokz: Any, user: str) -> tuple[str, list[int]]:
    """(prompt string as the model sees it, its token ids) — one source of truth.

    apply_chat_template is called twice with the SAME arguments, once for ids and once for text,
    rather than detokenising the ids: detokenisation is lossy for byte-level BPE and would shift
    offsets on exactly the non-ASCII identifiers this experiment cares about.
    """
    ids = tokz.apply_chat_template([{"role": "user", "content": user}], tokenize=True,
                                   add_generation_prompt=True, return_dict=False)
    text = tokz.apply_chat_template([{"role": "user", "content": user}], tokenize=False,
                                    add_generation_prompt=True)
    return text, list(ids)


def span_token_positions(tokz: Any, user: str, code: str,
                         spans: Sequence[Sequence[int]]) -> tuple[list[int], dict]:
    """Token indices covering `spans` (char offsets into `code`) within the rendered prompt.

    Returns (positions, diagnostics). `positions` is empty when the mapping cannot be verified —
    the caller must skip the item rather than steer an arbitrary position.
    """
    text, ids = _rendered_prompt(tokz, user)
    diag: dict = {"n_spans": len(spans), "n_spans_located": 0, "n_positions": 0,
                  "prompt_tokens": len(ids), "reason": None}

    # The code is embedded verbatim in the prompt; find where. If it appears more than once the
    # offsets are ambiguous and the item is refused rather than guessed.
    first = text.find(code)
    if first < 0:
        diag["reason"] = "code not found verbatim in rendered prompt"
        return [], diag
    if text.find(code, first + 1) >= 0:
        diag["reason"] = "code appears more than once in prompt; offsets ambiguous"
        return [], diag

    enc = tokz(text, return_offsets_mapping=True, add_special_tokens=False)
    offs = enc["offset_mapping"]
    # The chat template may tokenize differently than apply_chat_template's own ids; if the two
    # disagree in length the offsets do not describe the sequence being steered.
    if len(enc["input_ids"]) != len(ids):
        diag["reason"] = (f"offset tokenization ({len(enc['input_ids'])}) != chat-template ids "
                          f"({len(ids)}); offsets do not describe the steered sequence")
        return [], diag

    out: set[int] = set()
    located = 0
    for sp in spans:
        if len(sp) < 2:
            continue
        a, b = first + int(sp[0]), first + int(sp[1])
        hit = [i for i, (s, e) in enumerate(offs) if s < b and e > a and e > s]
        if hit:
            located += 1
            out.update(hit)
    diag["n_spans_located"] = located
    if located == 0:
        diag["reason"] = "no span mapped to any token"
        return [], diag

    # Verify: the decoded text of the chosen tokens must overlap the intended identifier text.
    # This is what catches an off-by-one in `first`, a template that re-escapes the code, or a
    # tokenizer whose offsets are relative to something other than `text`.
    want = {code[int(s[0]):int(s[1])].strip() for s in spans if len(s) >= 2}
    got = tokz.decode([ids[i] for i in sorted(out)])
    if not any(w and w[:6] in got for w in want):
        diag["reason"] = f"decoded tokens do not contain any identifier text; got {got[:80]!r}"
        return [], diag

    diag["n_positions"] = len(out)
    diag["decoded_sample"] = got[:120]
    return sorted(out), diag
