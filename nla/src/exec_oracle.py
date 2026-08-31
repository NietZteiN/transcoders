"""Ground-truth oracle: execute the stimulus and compare the trace's claims against reality.

This is the PRIMARY error detector. Arithmetic self-consistency (D1) finds nothing on this
corpus — 0 of 140 wrong traces contains an internally false arithmetic claim. The model's
arithmetic is always right; it computes the *wrong thing*. Only ground truth catches that.

Two probe modes:
  probe_calls()  — evaluate f(k) for each claimed k. Cheap, exact, covers the call-form family
                   (6% of cases, including the flagship). Used by D2-call.
  trace_run()    — instrumented execution recording assignments / container states, for the
                   variable (61%) and container (62%) families. Python via ast rewriting;
                   JavaScript via a Proxy apply-trap (returns) plus an argument-array Proxy (container mutations).

Sandbox: fresh tmpdir, RLIMIT_AS 12 GB + RLIMIT_FSIZE 16 MB, 10 s wall timeout, `python3 -I` or
`node --no-warnings`, never in-process exec. (A tight RLIMIT_AS kills node at startup — V8 reserves
a huge virtual space; RLIMIT_NPROC is unusable here because it counts all of the user's processes.)

KNOWN GAP: JavaScript *local* variables are not observable — JS has no assignment hook, and the
argument-array Proxy below only sees mutations to containers passed IN. JS traces that reason over
locals therefore fall through to the judge (D3).

IMPORTANT: run against the TIER'S OWN source (the renamed/flattened code the model actually
saw) so claimed identifier names line up. Functional equivalence across tiers is already
execution-validated upstream, so the values are the same.
"""
from __future__ import annotations

import json
import os
import resource
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

TIMEOUT_S = 10
# node/V8 RESERVES a large virtual address space at startup, so a tight RLIMIT_AS kills it
# outright (observed: every JS probe crashed in node::Start under a 2 GB cap). Cap generously
# and rely on the wall-clock timeout as the real containment. RLIMIT_NPROC is deliberately NOT
# set: it counts every process the *user* owns (96 on this shared box), not just our children,
# so any workable value is either useless or breaks unrelated jobs.
MEM_BYTES = 12 * 1024 ** 3
FSIZE_BYTES = 16 * 1024 ** 2


def _limits():
    resource.setrlimit(resource.RLIMIT_AS, (MEM_BYTES, MEM_BYTES))
    resource.setrlimit(resource.RLIMIT_FSIZE, (FSIZE_BYTES, FSIZE_BYTES))


@dataclass
class OracleResult:
    status: str                       # ok | timeout | crash | unsupported
    calls: dict = field(default_factory=dict)      # {(fn, k): value_str}
    events: list = field(default_factory=list)     # [{"kind","name","value","seq"}]
    stderr: str = ""


def _run(cmd: list[str], src: str, ext: str) -> tuple[str, str, int]:
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as d:
        p = os.path.join(d, f"probe.{ext}")
        with open(p, "w") as f:
            f.write(src)
        try:
            r = subprocess.run(cmd + [p], capture_output=True, text=True, timeout=TIMEOUT_S,
                               cwd=d, preexec_fn=_limits,
                               env={"PATH": os.environ.get("PATH", ""), "HOME": d})
            return r.stdout, r.stderr, r.returncode
        except subprocess.TimeoutExpired:
            return "", "TIMEOUT", -9


# ------------------------------------------------------------------ call probes
def probe_calls(code: str, language: str, calls: list[tuple[str, int]]) -> OracleResult:
    """Evaluate f(k) for each (fn, k). Values returned as strings for type-tolerant compare."""
    if not calls:
        return OracleResult("ok")
    if language.lower().startswith("java"):
        body = "\n".join(
            f'try{{o.push(["{f}",{k},String({f}({k}))])}}catch(e){{o.push(["{f}",{k},"ERR"])}}'
            for f, k in calls)
        src = f"{code}\nvar o=[];\n{body}\nconsole.log(JSON.stringify(o));"
        cmd, ext = ["node", "--no-warnings"], "js"
    else:
        body = "\n".join(
            f'try:\n    o.append(["{f}",{k},str({f}({k}))])\nexcept Exception:\n    o.append(["{f}",{k},"ERR"])'
            for f, k in calls)
        src = f"{code}\nimport json\no=[]\n{body}\nprint(json.dumps(o))"
        cmd, ext = ["python3", "-I"], "py"
    out, err, rc = _run(cmd, src, ext)
    if err == "TIMEOUT":
        return OracleResult("timeout", stderr=err)
    if rc != 0:
        return OracleResult("crash", stderr=err[-400:])
    try:
        return OracleResult("ok", calls={(f, int(k)): v for f, k, v in json.loads(out)})
    except Exception as e:
        return OracleResult("crash", stderr=f"parse: {e} · {out[:200]}")


# ------------------------------------------------------------------ traced runs
_PY_HARNESS = '''
import json, copy, sys
__ev = []
def __rec(kind, name, value, seq=None):
    try:
        v = copy.deepcopy(value)
        if isinstance(v, (list, dict, set, tuple)) and len(v) > 5000:
            v = {{"__big__": len(v)}}
        json.dumps(v, default=str)
    except Exception:
        v = str(value)[:200]
    __ev.append({{"kind": kind, "name": name, "value": v}})

{code}

try:
    __out = {call}
except Exception as e:
    __out = "ERR:" + type(e).__name__
print("__NLA_TRACE__" + json.dumps({{"out": str(__out), "events": __ev[:20000]}}, default=str))
'''

_JS_HARNESS = '''
var __ev = [];
function __push(e) {{ if (__ev.length < 20000) __ev.push(e); }}
// Wrap array arguments so IN-PLACE MUTATION is observable. JS has no assignment hook, but the
// dominant claim family in these traces is container state ("nums becomes [4, 4, 2, ...]"), and
// every push/splice/insert lands as a `set` trap on the proxy. Without this, JS localization was
// 3/47; the call-only Proxy below sees returns, never intermediate state.
function __wrapArr(a, name) {{
  if (!Array.isArray(a)) return a;
  __push({{kind:"assign", name:name, value:JSON.parse(JSON.stringify(a))}});   // initial state
  return new Proxy(a, {{
    set: function(t, k, v) {{
      t[k] = v;
      if (k !== "length") __push({{kind:"assign", name:name, value:JSON.parse(JSON.stringify(t))}});
      return true;
    }},
    deleteProperty: function(t, k) {{
      delete t[k];
      __push({{kind:"assign", name:name, value:JSON.parse(JSON.stringify(t))}});
      return true;
    }}
  }});
}}
{code}
try {{
  var __names = {names};
  __names.forEach(function(n) {{
    try {{
      var f = eval(n);
      if (typeof f === "function") {{
        var wrapped = new Proxy(f, {{
          apply: function(t, th, a) {{
            var __ps = (String(t).match(/\\(([^)]*)\\)/) || [0,""])[1]
                 .split(",").map(function(s){{ return s.trim().split(/[\\s=]/)[0]; }});
            var aa = a.map(function(x, i) {{ return __wrapArr(x, __ps[i] || ("arg"+i)); }});
            var r = Reflect.apply(t, th, aa);
            __push({{kind:"call", name:n, args:a.map(String), value:String(r)}});
            return r;
          }}
        }});
        eval(n + " = wrapped;");
      }}
    }} catch (e) {{}}
  }});
}} catch (e) {{}}
var __out;
try {{ __out = {call}; }} catch (e) {{ __out = "ERR:" + e.name; }}
console.log("__NLA_TRACE__" + JSON.stringify({{out: String(__out), events: __ev}}));
'''


# Methods that mutate in place — the value change is invisible as an Assign node, but traces
# talk about it constantly ("nums becomes [4, 4, 2, ...]"), so it must be recorded.
_MUTATORS = {"append", "insert", "extend", "pop", "remove", "sort", "reverse",
             "update", "add", "discard", "clear", "setdefault", "popitem"}


def _py_instrument(code: str) -> str:
    """Record the value sequence a reasoning trace can talk about:
      * explicit assignments and aug-assignments
      * SUBSCRIPT assignment targets (records the container)
      * LOOP VARIABLES, once per iteration (traces say "Iteration 3 (i = 2)")
      * IN-PLACE MUTATIONS via list/dict/set methods (traces say "nums becomes [...]")
    The last two were missing in the first version and cost ~85% of coverage: on the corpus's
    dominant shape (cruxeval loop/mutation functions) the only Assign is `count = len(nums)`.
    """
    import ast as A

    def rec(name: str):
        return A.parse(f'__rec("assign", {name!r}, {name})').body[0]

    class T(A.NodeTransformer):
        def visit_Assign(self, node):
            self.generic_visit(node)
            out = [node]
            for tgt in node.targets:
                if isinstance(tgt, A.Name):
                    out.append(rec(tgt.id))
                elif isinstance(tgt, A.Subscript) and isinstance(tgt.value, A.Name):
                    out.append(rec(tgt.value.id))
            return out

        def visit_AugAssign(self, node):
            self.generic_visit(node)
            if isinstance(node.target, A.Name):
                return [node, rec(node.target.id)]
            if isinstance(node.target, A.Subscript) and isinstance(node.target.value, A.Name):
                return [node, rec(node.target.value.id)]
            return node

        def visit_FunctionDef(self, node):
            self.generic_visit(node)
            pre = [rec(a.arg) for a in node.args.args]      # initial parameter values
            node.body = pre + node.body
            return node

        def visit_For(self, node):
            self.generic_visit(node)
            pre = []
            if isinstance(node.target, A.Name):
                pre.append(rec(node.target.id))
            elif isinstance(node.target, A.Tuple):
                pre += [rec(e.id) for e in node.target.elts if isinstance(e, A.Name)]
            node.body = pre + node.body
            return node

        def visit_Expr(self, node):
            self.generic_visit(node)
            v = node.value
            if (isinstance(v, A.Call) and isinstance(v.func, A.Attribute)
                    and v.func.attr in _MUTATORS and isinstance(v.func.value, A.Name)):
                return [node, rec(v.func.value.id)]
            return node

    try:
        tree = T().visit(A.parse(code))
        A.fix_missing_locations(tree)
        return A.unparse(tree)
    except Exception:
        return code       # fall back to uninstrumented; call-probes still work


def trace_run(code: str, call: str, language: str, fn_names: list[str] | None = None) -> OracleResult:
    """Instrumented execution. Python: per-assignment events. JS: per-call events."""
    if language.lower().startswith("java"):
        names = json.dumps(fn_names or [])
        src = _JS_HARNESS.format(code=code, call=call, names=names)
        cmd, ext = ["node", "--no-warnings"], "js"
    else:
        src = _PY_HARNESS.format(code=_py_instrument(code), call=call)
        cmd, ext = ["python3", "-I"], "py"
    out, err, rc = _run(cmd, src, ext)
    if err == "TIMEOUT":
        return OracleResult("timeout", stderr=err)
    if rc != 0:
        return OracleResult("crash", stderr=err[-400:])
    marker = out.find("__NLA_TRACE__")
    if marker < 0:
        return OracleResult("crash", stderr=f"no trace marker · {out[:200]}")
    try:
        d = json.loads(out[marker + len("__NLA_TRACE__"):])
        ev = d["events"]
        for i, e in enumerate(ev):
            e["seq"] = i
        return OracleResult("ok", events=ev)
    except Exception as e:
        return OracleResult("crash", stderr=f"parse: {e}")


# ------------------------------------------------------------------ comparison
def values_agree(claimed, truth) -> bool:
    """Type-tolerant equality: '927' ≡ 927 ≡ 927.0; quotes/whitespace stripped."""
    if truth is None or truth == "ERR":
        return True                      # unknown → cannot disagree
    a, b = str(claimed).strip().strip("`'\" "), str(truth).strip().strip("`'\" ")
    if a == b:
        return True
    try:
        return abs(float(a) - float(b)) < 1e-9
    except Exception:
        pass
    try:                                  # container compare, order-sensitive
        import ast as A

        def norm(x):
            """Tuples and lists are indistinguishable after the JSON round-trip the tracer
            does, so compare structurally: [(4,1)] must equal [[4,1]]."""
            if isinstance(x, (list, tuple)):
                return [norm(i) for i in x]
            if isinstance(x, dict):
                return {k: norm(v) for k, v in x.items()}
            return x

        return norm(A.literal_eval(a)) == norm(A.literal_eval(b))
    except Exception:
        return False
