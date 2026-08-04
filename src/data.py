"""Stimulus loading — the obfuscated-code snippets from Papers 2–3.

Defines the on-disk schema and a loader. The real datasets (Dataset A/B, tier variants,
identifier-span alignment) are populated under data/stimuli/ during Phase 0; until then
`toy_stimuli()` provides a tiny in-memory set so the extraction harness can be smoke-tested
(../CLAUDE.md §4) without any download or data dependency.

POSITION SCHEMA — character spans, NOT token indices. Token indices are tokenizer-specific
(Llama-3.1 prepends <|begin_of_text|>; GPT-2/Qwen don't; segmentation differs), so a static
token index would be silently wrong across the panel. Spans are (start, end) offsets into
`code`; src/extract_activations.py resolves them to token positions per model at runtime via
fast-tokenizer offset mappings and hard-fails if resolution falls below threshold.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from src.configs import PROJECT_ROOT


@dataclass
class Snippet:
    """One code snippet at one obfuscation tier."""
    snippet_id: str              # stable id shared across tiers (for L0↔L1b matching)
    tier: str                    # L0 | L1 | L1b | L2 | L3
    language: str                # python | javascript
    code: str
    expected_output: str | None = None
    # character spans [start, end) into `code`, resolved to tokens per model at runtime:
    identifier_spans: list[list[int]] = field(default_factory=list)   # E1/E6 (aligned identifiers)
    dispatcher_spans: list[list[int]] = field(default_factory=list)   # E3/E4 (dispatcher constructs)
    meta: dict = field(default_factory=dict)


def spans_of(code: str, name: str) -> list[list[int]]:
    """All [start, end) occurrences of `name` in `code` (helper for building span metadata)."""
    spans, start = [], 0
    while (i := code.find(name, start)) != -1:
        spans.append([i, i + len(name)])
        start = i + len(name)
    return spans


def load_jsonl(path: str | Path) -> list[Snippet]:
    """Load snippets from a JSONL file (one Snippet dict per line)."""
    snippets: list[Snippet] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                snippets.append(Snippet(**json.loads(line)))
    return snippets


def load_dataset(cfg: dict, dataset_key: str, tiers: list[str] | None = None) -> list[Snippet]:
    """Load a dataset named in configs/data.yaml, optionally filtered to `tiers`.

    Paths in data.yaml are project-root-relative; anchored here so runs work from any CWD.
    Raises (never returns silently-empty) so a tier typo or unpopulated dir fails fast.

    TODO(phase0): populate data/stimuli/<dataset>/ from the Papers 2–3 artifacts.
    """
    data_cfg = cfg.get("data_config_resolved") or {}
    spec = (data_cfg.get("datasets") or {}).get(dataset_key)
    if spec is None:
        raise KeyError(f"dataset {dataset_key!r} not in data config")
    path = PROJECT_ROOT / spec["path"]
    if not path.exists():
        raise FileNotFoundError(
            f"Stimuli for {dataset_key!r} not found at {path}. Populate data/stimuli/ "
            f"(Phase 0), or run with --smoke to use toy_stimuli()."
        )
    snippets = []
    for jsonl in sorted(path.glob("*.jsonl")):
        snippets.extend(load_jsonl(jsonl))
    if tiers:
        snippets = [s for s in snippets if s.tier in tiers]
    if not snippets:
        raise ValueError(
            f"0 snippets loaded for dataset={dataset_key!r} tiers={tiers!r} from {path} "
            f"— empty dir or tier-label mismatch. Refusing to continue with nothing."
        )
    return snippets


def toy_stimuli(n: int = 4) -> list[Snippet]:
    """Tiny in-memory stimuli for smoke-testing the harness. Not for real results.

    Includes the L0/L1b flagship pair (fibfib disguised as smoothArea) with real identifier
    spans, so span→token resolution is exercised end to end by the smoke path.
    """
    l0_code = ("const myFunct = (n) => { if (n==0||n==1) return 0; if (n==2) return 1; "
               "return myFunct(n-1)+myFunct(n-2)+myFunct(n-3); }  // myFunct(14) -> 927")
    l1b_code = ("const smoothArea = (_lastNSecs) => { if (_lastNSecs==0||_lastNSecs==1) return 0; "
                "if (_lastNSecs==2) return 1; return smoothArea(_lastNSecs-1)+smoothArea(_lastNSecs-2)"
                "+smoothArea(_lastNSecs-3); }  // smoothArea(14) -> 927")
    pool = [
        Snippet(snippet_id="toy_fibfib", tier="L0", language="javascript", code=l0_code,
                expected_output="927", identifier_spans=spans_of(l0_code, "myFunct")),
        Snippet(snippet_id="toy_fibfib", tier="L1b", language="javascript", code=l1b_code,
                expected_output="927", identifier_spans=spans_of(l1b_code, "smoothArea")),
    ]
    while len(pool) < n:
        i = len(pool)
        code = f"def f{i}(x):\n    return x * {i}"
        pool.append(Snippet(snippet_id=f"toy_{i}", tier="L0", language="python", code=code,
                            expected_output=None,
                            identifier_spans=spans_of(code, f"f{i}") + spans_of(code, "x")))
    return pool[:n]
