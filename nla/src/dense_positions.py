"""N5 — which token positions to read, and how to pair a wrong case with a control.

Positions are token indices into `templ + reply`, identical to the banked convention
(`capture_core.align_reply`), so dense reads are directly comparable with the 5,090 existing ones.

Design changed after the 2026-08-06 localization validation FAILED (D2-state vs D3 agree 39%,
r = +0.10; D3 self-agrees on 46%). Precise ±20-token bursts are only justified for the 3
known-answer-verified D2-call cases. Therefore:
  * the UNIFORM SWEEP is the primary instrument (stride 20 over the whole reply), and
  * the ERROR REGION is the fixed 0.70–0.90 relative strip read at stride 8 on EVERY case,
    which is where the verified errors actually sit (77%, 77%, 84%) and which controls get
    mirrored on relative position.
A per-case burst is added only for the verified-localization cases, flagged `burst_verified`.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from capture_core import Alignment, token_ok  # noqa: E402

STRIDE = 20              # uniform sweep (Option B: sequential, ~8.4 h for the full design)
STRIP_LO, STRIP_HI = 0.70, 0.90
STRIP_STRIDE = 8
BURST_RADIUS = 20        # tokens, verified-localization cases only
BURST_STRIDE = 2
MAX_POS_PER_CASE = 110   # guards one 1,922-token trace from eating the budget


def _keep(a: Alignment, p: int) -> bool:
    if not (a.reply_start + 2 <= p < a.n_total - 1):
        return False
    s, e = a.offsets[p]
    return e > s and token_ok(a.full, s, e)


def sweep_positions(a: Alignment, stride: int = STRIDE) -> list[int]:
    """Uniform sweep over the reply, starting past the boilerplate opening."""
    out = []
    p = a.reply_start + 4
    while p < a.n_total - 2:
        if _keep(a, p):
            out.append(p)
        elif _keep(a, p + 1):        # digit-piece anchors shift by one rather than vanish
            out.append(p + 1)
        p += stride
    return out


def strip_positions(a: Alignment, lo: float = STRIP_LO, hi: float = STRIP_HI,
                    stride: int = STRIP_STRIDE) -> list[int]:
    """The fixed 0.70–0.90 error region, read densely on EVERY case (localized or not)."""
    span = a.n_total - a.reply_start - 1
    if span <= 0:
        return []
    start = a.reply_start + int(lo * span)
    end = a.reply_start + int(hi * span)
    out = []
    p = max(start, a.reply_start + 4)
    while p <= min(end, a.n_total - 2):
        if _keep(a, p):
            out.append(p)
        elif _keep(a, p + 1):
            out.append(p + 1)
        p += stride
    return out


def burst_positions(a: Alignment, u_err: float, radius: int = BURST_RADIUS,
                    stride: int = BURST_STRIDE) -> list[int]:
    """Dense window around a *verified* localization (or its mirror in a control)."""
    span = a.n_total - a.reply_start - 1
    if span <= 0 or not (0.0 <= u_err <= 1.0):
        return []
    centre = a.reply_start + int(u_err * span)
    out = []
    for p in range(centre - radius, centre + radius + 1, stride):
        if _keep(a, p):
            out.append(p)
    return out


def plan_positions(a: Alignment, u_err: float | None = None) -> dict[int, str]:
    """position -> role. Roles: 'burst' (verified only) > 'strip' > 'sweep'.

    Ordering matters downstream: reads are executed burst-first, then strip, then sweep
    outward, so an aborted run still holds the informative region.
    """
    plan: dict[int, str] = {}
    for p in sweep_positions(a):
        plan[p] = "sweep"
    for p in strip_positions(a):
        plan[p] = "strip"
    if u_err is not None:
        for p in burst_positions(a, u_err):
            plan[p] = "burst"
    if len(plan) > MAX_POS_PER_CASE:      # thin the sweep only; never the burst/strip
        sweep = [p for p, r in plan.items() if r == "sweep"]
        keep = set(sweep[:: max(1, math.ceil(len(sweep) / max(1, MAX_POS_PER_CASE - (len(plan) - len(sweep)))))])
        plan = {p: r for p, r in plan.items() if r != "sweep" or p in keep}
    return plan


def order_positions(plan: dict[int, str], a: Alignment, u_err: float | None) -> list[tuple[int, str]]:
    """burst → strip → sweep, each ordered outward from the error region."""
    span = max(a.n_total - a.reply_start - 1, 1)
    centre = a.reply_start + int((u_err if u_err is not None else 0.8) * span)
    rank = {"burst": 0, "strip": 1, "sweep": 2}
    return sorted(plan.items(), key=lambda kv: (rank[kv[1]], abs(kv[0] - centre)))
