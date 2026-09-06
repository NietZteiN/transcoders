"""Refuse to average an arm that did not write anything.

Bugs #8 and #9 (2026-09-06) both had the same shape: an arm's write positions went to zero for
some or all items, the arm therefore scored *exactly* 0.0 there, and the scorer averaged those
zeros in with real measurements and reported a verdict off the result. Once as a **false positive**
(`T_L2_all` wrote nothing on every item, scored +0.00 with a CI of [0.00, 0.00], and the frozen
rule returned `W16-STRUCTURE-CARRIES-IT` from `45.71 - 0.00`), and once as a **false negative**
(`T_L0_sub`/`T_L1_sub` wrote nothing on 29 of 49 items, dragging a real +21.82 gap down to a
sub-threshold +8.91).

The underlying measurements were sound both times. What failed is that **nothing in the harness
distinguished "this arm wrote nothing" from "this arm did nothing"** — the same class as B5's
silently shrinking denominator and P0.3's missing `cells_scored.json`, where a cell whose
denominator quietly shrank looked identical to one that did badly.

A mean read without asserting how many positions produced it is not a decision rule, it is a
formatting step. So:

  * `record_positions` stamps a per-arm written-position count into every row;
  * `arm_series` returns an arm's values **only** for items where it actually wrote, plus the count
    it dropped, so a shrinking denominator is visible instead of silent;
  * `paired` refuses a contrast whose two arms wrote at different positions on the same item, which
    is the assumption every `A - B` in this thread quietly relies on.

`ZERO_EVERYWHERE` is deliberately an exception rather than a flag: an arm that never wrote is not a
null result, it is an absent experiment, and it must not reach a verdict table at all.
"""
from __future__ import annotations


class ArmNotWritten(Exception):
    """An arm wrote zero positions on every item — an absent experiment, not a null result."""


def pos_key(arm: str) -> str:
    return f"n_pos_{arm}"


def record_positions(row: dict, targets: dict) -> dict:
    """Stamp `n_pos_<arm>` for every arm from its resolved target map. Call before writing a row."""
    for arm, tgt in targets.items():
        row[pos_key(arm)] = len(tgt)
    return row


def arm_series(rows: list[dict], arm: str,
               allow_exact_zero: bool = False) -> tuple[list[float], dict]:
    """(values on items where `arm` actually wrote, report).

    Two detectors, because the first one cannot protect banked data:

    1. `n_pos_<arm> == 0` — exact, available only for rows written after this module existed.
    2. **`dG` exactly 0.0** — the signature of an unwritten arm in rows that predate (1). A
       teacher-forced log-probability difference landing on precisely 0.0 does not happen when an
       intervention was actually applied; it happens when nothing was. Re-scoring the 379908 rows
       through detector (1) alone reproduced the bad verdict, because those rows carry no position
       field — which is exactly why (2) exists.

    `allow_exact_zero=True` is required for arms where 0.0 is a legitimate measurement — the SELF
    identity arm writes each position's own activation back and *should* score 0, and 21 of 49
    items did so exactly. Making it opt-in per arm keeps the detector strict everywhere else
    instead of being weakened globally by one honest exception.
    """
    k = pos_key(arm)
    have = k in (rows[0] if rows else {})
    vals, dropped, zeros = [], 0, 0
    for r in rows:
        if have and r.get(k, 0) == 0:
            dropped += 1
            continue
        v = float(r[f"dG_{arm}"])
        if v == 0.0 and not allow_exact_zero and not have:
            zeros += 1
            continue
        vals.append(v)
    rep = {"arm": arm, "n_used": len(vals), "n_dropped_zero_positions": dropped,
           "n_dropped_exact_zero": zeros, "position_field_present": have}
    if not vals:
        raise ArmNotWritten(
            f"{arm}: no item with a written position "
            f"({dropped} zero-position, {zeros} exact-zero of {len(rows)} rows)")
    return vals, rep


def paired(rows: list[dict], a: str, b: str,
           allow_exact_zero: bool = False) -> tuple[list[float], dict]:
    """(per-item a-b on items where BOTH wrote, report). Refuses mismatched position counts.

    Equal position counts are what makes `a - b` a contrast about CONTENT rather than about how
    much each arm perturbed — the property every null in this family depends on.
    """
    ka, kb = pos_key(a), pos_key(b)
    have = ka in (rows[0] if rows else {}) and kb in (rows[0] if rows else {})
    diffs, dropped, mismatch = [], 0, 0
    for r in rows:
        va, vb = float(r[f"dG_{a}"]), float(r[f"dG_{b}"])
        if have:
            na, nb = r.get(ka, 0), r.get(kb, 0)
            if na == 0 or nb == 0:
                dropped += 1
                continue
            if na != nb:
                mismatch += 1
                continue
        elif not allow_exact_zero and (va == 0.0 or vb == 0.0):
            dropped += 1          # see arm_series: detector (2), for rows with no position field
            continue
        diffs.append(va - vb)
    rep = {"contrast": f"{a}-{b}", "n_used": len(diffs),
           "n_dropped_zero_positions": dropped, "n_dropped_position_mismatch": mismatch,
           "position_field_present": have}
    if not diffs:
        raise ArmNotWritten(f"{a}-{b}: no item where both arms wrote at matching positions")
    return diffs, rep
