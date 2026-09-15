"""Type + role tags for every renamed identifier of a renamed Java snippet — the ORACLE-FREE meaning
source behind the H-R6 `role_proto` / `prompt_types` arms (nla/configs/ase_vectors.yaml).

Everything here is read off the RENAMED source: the decoy name's own declaration (Java is statically
typed, so the declared type contradicts most decoys outright -- `List<Double> auths`) and how the
name is used. The true name never enters. javalang skips comments, so the renamed docstring is
never consulted either.

ROLE = first matching rule, in this order (frozen in the prereg):
  method      the identifier is a declared method name
  parameter   a formal parameter of a method/constructor
  loop_index  declared in a classic `for (...; ...; ...)` initialiser
  iterated    the iterable of an enhanced for, or a loop element variable (`for (T x : xs)`)
  accumulator assigned with a compound operator, or assigned an expression that reads itself
  returned    appears bare in a `return` statement
  local       any other local variable / field
USAGE is a descriptive flag set kept alongside (indexed / called / returned / accumulated / iterated)
for the prompt line; the tag used for prototype matching is `type_role` = f"{type}|{role}".
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import javalang
from javalang import tree as jt

ROLE_ORDER = ("method", "parameter", "loop_index", "iterated", "accumulator", "returned", "local")
COMPOUND = ("+=", "-=", "*=", "/=", "%=", "|=", "&=", "^=", "<<=", ">>=")


@dataclass
class Tag:
    name: str                      # the decoy identifier as it appears in the renamed source
    kind: str                      # method | variable
    type: str                      # declared type text, e.g. List<Double>, int[], double; "?" if unresolved
    role: str
    usage: list[str] = field(default_factory=list)

    @property
    def type_role(self) -> str:
        return f"{self.type}|{self.role}"

    def prompt_line(self, template: str) -> str:
        usage = ", ".join(self.usage) if self.usage else "none noted"
        return template.format(decoy=self.name, type=self.type, role=self.role, usage=usage)


def _type_text(t) -> str:
    if t is None:
        return "void"
    name = getattr(t, "name", "?")
    args = getattr(t, "arguments", None)
    if args:
        inner = ", ".join(_type_text(a.type) if getattr(a, "type", None) is not None else "?" for a in args)
        name = f"{name}<{inner}>"
    dims = getattr(t, "dimensions", None) or []
    return name + "[]" * len(dims)


def _reads(node, name: str) -> bool:
    """Does `name` occur as a MemberReference anywhere under `node`?"""
    if node is None:
        return False
    for _, n in node.filter(jt.MemberReference) if hasattr(node, "filter") else []:
        if n.member == name and not n.qualifier:
            return True
    return False


def tag_source(src: str, decoys: list[str]) -> dict[str, Tag]:
    """{decoy: Tag} for every decoy that the parser can locate; unresolved decoys get type "?" / local."""
    tree = javalang.parse.parse(src)
    tags: dict[str, Tag] = {n: Tag(n, "variable", "?", "local") for n in decoys}
    want = set(decoys)

    # declarations
    for _, m in tree.filter(jt.MethodDeclaration):
        if m.name in want:
            tags[m.name] = Tag(m.name, "method", _type_text(m.return_type), "method")
    for _, p in tree.filter(jt.FormalParameter):
        if p.name in want and tags[p.name].role == "local":
            tags[p.name] = Tag(p.name, "variable", _type_text(p.type), "parameter")
    for _, f in tree.filter(jt.ForStatement):
        ctrl = f.control
        if isinstance(ctrl, jt.ForControl) and ctrl.init:
            inits = ctrl.init if isinstance(ctrl.init, list) else [ctrl.init]
            for it in inits:
                for d in getattr(it, "declarators", []) or []:
                    if d.name in want and tags[d.name].role == "local":
                        tags[d.name] = Tag(d.name, "variable", _type_text(it.type), "loop_index")
        elif isinstance(ctrl, jt.EnhancedForControl):
            v = ctrl.var
            for d in v.declarators:
                if d.name in want and tags[d.name].role == "local":
                    tags[d.name] = Tag(d.name, "variable", _type_text(v.type), "iterated")
                    tags[d.name].usage.append("loop element")
            it = ctrl.iterable
            if isinstance(it, jt.MemberReference) and it.member in want and not it.qualifier:
                tags[it.member].usage.append("iterated")
    for _, lv in tree.filter(jt.LocalVariableDeclaration):
        for d in lv.declarators:
            if d.name in want and tags[d.name].type == "?":
                tags[d.name].type = _type_text(lv.type)
    for _, fd in tree.filter(jt.FieldDeclaration):
        for d in fd.declarators:
            if d.name in want and tags[d.name].type == "?":
                tags[d.name].type = _type_text(fd.type)

    # usage flags
    for _, a in tree.filter(jt.Assignment):
        lhs = a.expressionl
        if isinstance(lhs, jt.MemberReference) and lhs.member in want and not lhs.qualifier:
            if a.type in COMPOUND or _reads(a.value, lhs.member):
                tags[lhs.member].usage.append("accumulated")
    for _, r in tree.filter(jt.ReturnStatement):
        e = r.expression
        if isinstance(e, jt.MemberReference) and e.member in want and not e.qualifier:
            tags[e.member].usage.append("returned")
    for _, mi in tree.filter(jt.MethodInvocation):
        if mi.qualifier in want:
            tags[mi.qualifier].usage.append(f"called .{mi.member}()")
    for _, mr in tree.filter(jt.MemberReference):
        if mr.member in want and not mr.qualifier and mr.selectors:
            if any(isinstance(s, jt.ArraySelector) for s in mr.selectors):
                tags[mr.member].usage.append("indexed")

    # role promotion from usage for identifiers still `local`
    for t in tags.values():
        t.usage = sorted(set(t.usage))
        if t.role == "local":
            if "iterated" in t.usage:
                t.role = "iterated"
            elif "accumulated" in t.usage:
                t.role = "accumulator"
            elif "returned" in t.usage:
                t.role = "returned"
    return tags


def tag_snippet(java_path: str, rename_map: dict[str, str]) -> dict[str, Tag]:
    """rename_map is {original: decoy}; tags are keyed by DECOY (the only name we may look at)."""
    src = open(java_path).read()
    return tag_source(src, list(rename_map.values()))


if __name__ == "__main__":     # quick look: python ase_roles.py <java> decoy1 decoy2 ...
    import sys
    for t in tag_source(open(sys.argv[1]).read(), sys.argv[2:]).values():
        print(t.name, t.kind, t.type, t.role, t.usage)
