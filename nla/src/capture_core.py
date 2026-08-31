"""Shared capture machinery, factored out of overnight_capture.main()'s closures.

WHY THIS EXISTS: the offset / `reply_start` arithmetic determines what a read's `position`
means. The 2026-08-04 corpus (5,090 reads) is indexed in that space, and every follow-up run
must land in the SAME space or cross-run comparison is silently wrong. So the arithmetic lives
here exactly once — `align_reply()` — and both runners call it. Do not re-derive it inline.

Contents:
  align_reply()          the tokenizer/offset/reply_start arithmetic (was overnight_capture:277-284,331)
  ReadEngine             one_read + CJK monitor (was the `one_read` closure)
  ServerGuard            AV /health + one restart attempt (was :157-183, now stateful)
  WallGuard              --max-hours budget
  JsonlSink              append-and-flush + resume-key loading (error rows retried)

Import-light: no torch at module scope, so a caller can pin CUDA_VISIBLE_DEVICES first.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, NamedTuple

_HERE = Path(__file__).resolve().parent
_NLA_ROOT = _HERE.parent
if str(_NLA_ROOT / "vendor" / "nla-repo") not in sys.path:
    sys.path.insert(0, str(_NLA_ROOT / "vendor" / "nla-repo"))
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# Single source of truth for the run constants (imported, never redefined).
from overnight_capture import (  # noqa: E402
    CJK_RE, LAYER_INDEX, MAX_NEW_READ, SEED, TARGET_MODEL, build_user,
)

CJK_FRAC_ABORT = 0.30      # a read counts as an injection failure only if MOSTLY CJK
CJK_WINDOW = 100
CJK_MIN_SAMPLES = 50
CJK_RATE_ABORT = 0.05


# ---------------------------------------------------------------- alignment
class Alignment(NamedTuple):
    """Character/token geometry of one (prompt, reply) pair.

    `positions` in every capture file are indices into the token sequence of
    `full = templ + reply`, tokenized with add_special_tokens=False.
    """
    templ: str
    full: str
    offsets: list[tuple[int, int]]
    reply_start: int          # token index of the first reply token
    n_total: int              # number of tokens in `full`
    reply: str

    def tok_span(self, tok_idx: int) -> tuple[int, int]:
        return self.offsets[tok_idx]

    def rel_u(self, tok_idx: int) -> float:
        """Relative position within the reply, 0..1 (nan for prompt-side tokens)."""
        span = self.n_total - self.reply_start - 1
        if span <= 0 or tok_idx < self.reply_start:
            return float("nan")
        return (tok_idx - self.reply_start) / span


def align_reply(tokenizer, user: str, reply: str) -> Alignment:
    """Reproduces overnight_capture.py:277-284 + :331 exactly. Batch-1, no padding."""
    templ = tokenizer.apply_chat_template(
        [{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True)
    full = templ + (reply or "")
    enc = tokenizer(full, return_offsets_mapping=True, add_special_tokens=False)
    reply_start = len(tokenizer(templ, add_special_tokens=False)["input_ids"])
    return Alignment(templ=templ, full=full, offsets=[tuple(o) for o in enc["offset_mapping"]],
                     reply_start=reply_start, n_total=len(enc["input_ids"]), reply=reply or "")


def token_ok(full: str, s: int, e: int) -> bool:
    """The banked runs' token filter: skip 1-char and pure-digit pieces (digit tokens
    verbalize as numerology — established noise source)."""
    txt = full[s:e].strip()
    return len(txt) > 1 and not txt.isdigit()


def token_of_char(a: Alignment, char_pos: int) -> int | None:
    """First token whose span covers char_pos (in `full` coordinates)."""
    for i, (s, e) in enumerate(a.offsets):
        if e > s and s <= char_pos < e:
            return i
    return None


# ---------------------------------------------------------------- reads
@dataclass
class ReadResult:
    text: str
    rt_cos: float
    cjk_frac: float


class ReadEngine:
    """Wraps the AV client + AR critic. Identical semantics to overnight_capture.one_read."""

    def __init__(self, av, ar, max_new: int = MAX_NEW_READ):
        self.av, self.ar, self.max_new = av, ar, max_new
        self.n_reads = 0
        self._cjk: deque[bool] = deque(maxlen=CJK_WINDOW)

    def read_one(self, v) -> ReadResult:
        text = self.av.generate(v, temperature=0.0, max_new_tokens=self.max_new)
        _, rt = self.ar.score(text, v)
        frac = len(CJK_RE.findall(text)) / max(len(text), 1)
        self._cjk.append(frac > CJK_FRAC_ABORT)
        self.n_reads += 1
        return ReadResult(text=text, rt_cos=float(rt), cjk_frac=frac)

    def cjk_alarm(self) -> str | None:
        if len(self._cjk) >= CJK_MIN_SAMPLES:
            rate = sum(self._cjk) / len(self._cjk)
            if rate > CJK_RATE_ABORT:
                return f"CJK rate {sum(self._cjk)}/{len(self._cjk)} — injection failure"
        return None


# ---------------------------------------------------------------- guards & sinks
class ServerGuard:
    """AV server liveness with exactly one restart attempt (overnight_capture:157-183)."""

    def __init__(self, url: str, log_path: Path, av_dir: Path):
        self.url, self.log_path, self.av_dir = url, Path(log_path), Path(av_dir)
        self.restart_used = False

    def healthy(self) -> bool:
        import urllib.request
        try:
            urllib.request.urlopen(f"{self.url}/health", timeout=10)
            return True
        except Exception:
            return False

    def restart_once(self) -> bool:
        if self.restart_used:
            return False
        self.restart_used = True
        gpu = os.environ.get("NLA_SERVER_GPU")
        if gpu is None:
            return False
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
        with open(self.log_path, "a") as lf:
            lf.write(f"\n=== runner-initiated restart {datetime.now().isoformat()} ===\n")
            subprocess.Popen(
                [sys.executable, "-m", "sglang.launch_server", "--model-path", str(self.av_dir),
                 "--port", self.url.rsplit(":", 1)[-1].strip("/"),
                 "--disable-radix-cache", "--mem-fraction-static", "0.6"],
                stdout=lf, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        for _ in range(60):
            time.sleep(5)
            if self.healthy():
                return True
        return False


class WallGuard:
    def __init__(self, max_hours: float):
        self.t0, self.max_hours = time.time(), max_hours

    def expired(self) -> bool:
        return (time.time() - self.t0) / 3600 >= self.max_hours

    def elapsed_h(self) -> float:
        return (time.time() - self.t0) / 3600


class JsonlSink:
    """Append-and-flush per row; resume by key. Error rows are NOT counted as done."""

    def __init__(self, path: Path, key_field: str = "task_key"):
        self.path, self.key_field = Path(path), key_field
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a")

    def done_keys(self) -> set[str]:
        keys: set[str] = set()
        if self.path.exists():
            with open(self.path) as f:
                for line in f:
                    try:
                        row = json.loads(line)
                        if "error" not in row:
                            keys.add(row[self.key_field])
                    except Exception:
                        pass
        return keys

    def append(self, row: dict) -> None:
        row.setdefault("ts", datetime.now(timezone.utc).isoformat())
        self._fh.write(json.dumps(row) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


def load_captures(path: str | Path) -> dict[str, dict]:
    """task_key -> banked capture row (the 2026-08-04 corpus)."""
    out: dict[str, dict] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                if "error" not in r:
                    out[r["task_key"]] = r
    return out
