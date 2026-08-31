"""Task construction for the overnight capture run.

Three pieces, all pure-python (no torch):
  * build_call(row)          — call-string shim for output-prediction stimuli rows
  * synthetic_slice_tasks()  — seeded line-numbered programs with ground-truth dynamic
                               slices BY CONSTRUCTION (print-line convention: excluded,
                               and the prompt says so — fixes the 2026-08-04 ambiguity)
  * glosses(rename_map)      — (true_gloss, decoy_gloss) template glosses for DRM_AR.
                               Crude by design: DRM is a RELATIVE signal (L1b vs L0),
                               absolute sign is not trusted.
"""
from __future__ import annotations

import ast
import json
import random
import re

SEED = 20260724

_CALL_RE = re.compile(r"^\s*[A-Za-z_$][\w$]*\s*\(.*\)\s*$", re.S)


def build_call(row: dict) -> str | None:
    """Call string for a stimulus row, or None if unconstructible (count as skipped).

    meta.input is either a stringified arg list ("[14]") -> fn(args), or already a call
    string ("f('abc')") -> used as-is (B rows carry tier-appropriate call strings).
    On renamed tiers meta.fn_name is the ORIGINAL name — map it through rename_map
    (orig -> renamed) so the call matches the code the model actually sees. The resolved
    fn must occur in the code (word-boundary), else skip. JSON arg encoding is acceptable
    for both languages at this datasets' value complexity.
    """
    meta = row.get("meta") or {}
    raw = meta.get("input")
    fn = meta.get("fn_name")
    if raw is None:
        return None
    raw = str(raw)
    try:
        args = ast.literal_eval(raw)
        if isinstance(args, list) and fn:
            rm = meta.get("rename_map") or {}
            if meta.get("pairing_ok"):
                fn = rm.get(fn, fn)
            if not re.search(rf"(?<![\w$]){re.escape(fn)}(?![\w$])", row["code"]):
                return None                       # resolved fn absent from this tier's code
            return f"{fn}({', '.join(json.dumps(a) for a in args)})"
    except (ValueError, SyntaxError):
        pass
    if _CALL_RE.match(raw):
        # Call-string inputs may use a stale head: the ORIGINAL fn on renamed tiers
        # (humaneval-js L1b) or a generic placeholder `myFunct` (leetcode). Resolve the
        # head against the code: try head, its renamed form, then the resolved fn_name.
        head = re.match(r"\s*([A-Za-z_$][\w$]*)", raw).group(1)
        rm = meta.get("rename_map") or {}
        candidates = [head]
        if meta.get("pairing_ok"):
            candidates.append(rm.get(head))
            if fn:
                candidates.append(rm.get(fn, fn))
        elif fn:
            candidates.append(fn)
        # last resort: the function the code itself defines (leetcode rows use a
        # placeholder head `myFunct` and carry no usable fn_name)
        m = re.search(r"^\s*def\s+([A-Za-z_]\w*)\s*\(|(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=|function\s+([A-Za-z_$][\w$]*)\s*\(",
                      row["code"], re.M)
        if m:
            candidates.append(next(g for g in m.groups() if g))
        for cand in candidates:
            if cand and re.search(rf"(?<![\w$]){re.escape(cand)}(?![\w$])", row["code"]):
                return re.sub(rf"^(\s*){re.escape(head)}", rf"\g<1>{cand}", raw.strip(), count=1)
    return None


# ---------------------------------------------------------------------------- slicing
_NAME_POOL = ["alpha", "beta", "gamma", "delta", "omega", "sigma", "kappa", "theta",
              "count", "total", "value", "accum", "level", "score", "width", "depth"]

SLICE_CONVENTION = (
    "Convention: a line belongs to the slice if it EXECUTES and its effect reaches the "
    "printed value. The final `print` line itself does NOT count as part of the slice."
)


def _pick_names(rng: random.Random, k: int) -> list[str]:
    return rng.sample(_NAME_POOL, k)


def synthetic_slice_tasks(n: int = 30, seed: int = SEED) -> list[dict]:
    """n seeded slicing tasks over 3 templates; ground truth fixed per template shape."""
    rng = random.Random(seed)
    tasks = []
    for i in range(n):
        template = i % 3
        c = [rng.randint(2, 9) for _ in range(6)]
        if template == 0:
            # straight-line: e = f(a, c(a)); b/d are distractors -> slice {1,3,5}
            a, b, cc, d, e = _pick_names(rng, 5)
            code = (f"1  {a} = {c[0]}\n2  {b} = {c[1]}\n3  {cc} = {a} * {c[2]}\n"
                    f"4  {d} = {b} + {c[3]}\n5  {e} = {cc} + {a}\n6  print({e})")
            target, truth = e, [1, 3, 5]
        elif template == 1:
            # loop accumulator; the second accumulator is a distractor -> slice {1,2,4,5}
            nv, tot, cnt = _pick_names(rng, 3)
            code = (f"1  {nv} = {c[0]}\n2  {tot} = 0\n3  {cnt} = 0\n"
                    f"4  for i in range({nv}):\n5      {tot} += i * {c[1]}\n"
                    f"6  {cnt} = {cnt} + {c[2]}\n7  print({tot})")
            target, truth = tot, [1, 2, 4, 5]
        else:
            # branch, condition TRUE by construction (x >= 2 > 1) -> slice {1,2,4,5,8}
            x, y, z, w = _pick_names(rng, 4)
            code = (f"1  {x} = {c[0]}\n2  {y} = {c[1] * 10}\n3  {z} = 0\n"
                    f"4  if {x} > 1:\n5      {z} = {y} + {x}\n6  else:\n"
                    f"7      {z} = {y} - {x}\n8  {w} = {z} * {c[2]}\n9  print({w})")
            target, truth = w, [1, 2, 4, 5, 8]
        tasks.append({"task_id": f"synslice_{i:02d}_t{template}", "kind": "slice_prediction",
                      "language": "python", "code": code, "target": target,
                      "truth": truth, "key_names": [target],
                      "template": template, "seed": seed})
    return tasks


# ---------------------------------------------------------------------------- glosses
def _words(name: str) -> list[str]:
    parts = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", name).replace("_", " ").split()
    return [p.lower() for p in parts if len(p) > 1]


def glosses(rename_map: dict[str, str]) -> tuple[str, str] | None:
    """(true_gloss, decoy_gloss) from a decoy<->true rename map; None if unpaired/empty.

    Fallback maps from the converter use '?unpairedN' pseudo-keys — no true names there,
    so no gloss pair (DRM skipped for that row, counted in the summary).
    """
    true_names = [k for k in rename_map if not k.startswith("?")]
    decoy_names = [v for k, v in rename_map.items() if not k.startswith("?")]
    tw = sorted({w for nm in true_names for w in _words(nm)})
    dw = sorted({w for nm in decoy_names for w in _words(nm)})
    if not tw or not dw:
        return None
    return (f"code whose identifiers are about {', '.join(tw)}",
            f"code whose identifiers are about {', '.join(dw)}")
