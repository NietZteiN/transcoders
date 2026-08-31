"""Extract checkable value-claims from a reasoning trace.

Three families, sized against the real corpus (117 non-truncated wrong cases):
  call-form  `f(k) = v`        →   6% of cases  (but this is the flagship's shape)
  variable   `x = v`, `x becomes v`  →  61%
  container  `nums becomes [.,.]`    →  62%
(any family, ≥3 claims: 85%; 15% have none → judge fallback)

TWO PARSER BUGS FOUND BY RUNNING THIS ON REAL TRACES — both silently produced false
localizations on the first pass, so they are handled here by construction:
  1. HEX literals: `fibfib(2) = 0x1` must parse as 1, not 0 (the L1 tier renders every
     number in hex).
  2. CHAINED equalities: `smoothArea(3) = \\`1 + 0 + 0\\` = 1` — the claimed value is the
     number after the SECOND `=`, not the first. Take the last number of the rhs line.

`safe_eval` never calls eval() on untrusted text: it parses to an AST and walks it against
a whitelist first.
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass

NUM = r"(?:0[xX][0-9a-fA-F]+|-?\d+(?:\.\d+)?)"

# call-form: f(k) = <rhs to end of line>
R_CALL = re.compile(rf"`?([A-Za-z_$][\w$]*)\s*\(\s*({NUM})\s*\)`?\s*(?:=|is|→|->|equals|returns)\s*([^\n]*)")
# variable: x = v / x becomes v / x is now v
R_VAR = re.compile(rf"\b([A-Za-z_$][\w$]*)\s*(?:=|is now|becomes|→|->)\s*`?({NUM})`?(?![\w.(])")
# container state: name ... [ ... ]
R_STATE = re.compile(r"`?([A-Za-z_$][\w$]*)`?\s*(?:=|is now|becomes|→|->)\s*`?(\[[^\]\n]{0,200}\])`?")
# pure arithmetic that can be checked against itself
R_ARITH = re.compile(rf"(?<![\w.])({NUM}(?:\s*[-+*/]\s*{NUM}){{1,8}})\s*=\s*({NUM})(?![\w.])")

_ALLOWED = (ast.Expression, ast.Constant, ast.BinOp, ast.UnaryOp,
            ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.USub, ast.UAdd)


@dataclass
class Claim:
    family: str          # call | var | state | arith
    name: str            # function or variable name
    arg: int | None      # call argument (call-form only)
    value: object        # claimed value (int/float/str/list)
    span: tuple[int, int]
    raw: str


def parse_num(s: str):
    """Hex-aware numeric parse. Returns int/float or None."""
    s = s.strip().strip("`")
    try:
        return int(s, 0) if re.fullmatch(r"0[xX][0-9a-fA-F]+|-?\d+", s) else float(s)
    except Exception:
        return None


def last_num(rhs: str):
    """Value of a possibly-chained rhs: the LAST number on the line (bug #2)."""
    rhs = rhs.split("(")[0]                     # stop before any further call on the line
    nums = re.findall(NUM, rhs)
    return parse_num(nums[-1]) if nums else None


def safe_eval(expr: str):
    """Evaluate a pure-arithmetic expression, or return None. Never eval()s arbitrary text."""
    expr = expr.replace("`", "").strip()
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED):
            return None
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float)):
            return None
    try:
        return eval(compile(tree, "<claim>", "eval"))   # AST already whitelisted above
    except Exception:
        return None


def extract_claims(reply: str) -> list[Claim]:
    out: list[Claim] = []
    for m in R_CALL.finditer(reply):
        v = last_num(m.group(3))
        k = parse_num(m.group(2))
        if v is not None and k is not None:
            out.append(Claim("call", m.group(1), int(k), v, m.span(), m.group(0)[:120]))
    for m in R_VAR.finditer(reply):
        v = parse_num(m.group(2))
        if v is not None:
            out.append(Claim("var", m.group(1), None, v, m.span(), m.group(0)[:120]))
    for m in R_STATE.finditer(reply):
        try:
            v = ast.literal_eval(m.group(2))
        except Exception:
            continue
        out.append(Claim("state", m.group(1), None, v, m.span(), m.group(0)[:120]))
    for m in R_ARITH.finditer(reply):
        lhs, rhs = safe_eval(m.group(1)), parse_num(m.group(2))
        if lhs is not None and rhs is not None:
            out.append(Claim("arith", "", None, (lhs, rhs), m.span(), m.group(0)[:120]))
    out.sort(key=lambda c: c.span[0])
    return out


def arith_disagreements(claims: list[Claim]) -> list[Claim]:
    """D1: claims whose own arithmetic does not hold. (Empirically 0/140 on this corpus —
    the model's arithmetic is always internally consistent; it computes the wrong thing.)"""
    bad = []
    for c in claims:
        if c.family == "arith":
            lhs, rhs = c.value
            if abs(lhs - rhs) > 1e-9:
                bad.append(c)
    return bad
