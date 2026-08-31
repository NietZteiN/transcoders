"""Redaction filter for anything derived from the quarantined malware corpus.

WHY THIS IS NOT OPTIONAL. The derived text is contaminated with live indicators, measured over
`reads.jsonl` (183 items / 2,554 readings):

    read texts     114 URLs · 13 IPs · 7 IP:port · 10 base64 blobs
    model replies   84 URLs · 25 IPs · 7 IP:port
    package names  183 live malicious PyPI identifiers

Those are real command-and-control endpoints and real package names from real supply-chain
malware. Publishing them — or pasting them into any external service — would distribute working
indicators, so the quarantine rule requires a redaction pass before anything derived reaches a
report. This module is that pass, written as reviewable code rather than an inline regex so the
filter itself can be audited and tested.

WHAT IT KEEPS. The point of the corpus is what the model's internal state *says*, so redaction is
surgical: an endpoint is destroyed, the capability claim around it is preserved.

    "exfiltrates the token to https://discord.com/api/webhooks/1234/abcd"
        -> "exfiltrates the token to [URL]"

The word "discord" is deliberately KEPT. It carries the finding — which platform the exfiltration
targets is the semantic content the reading exists to report — while the webhook that made it
actionable is gone. Stripping the platform name too would redact the result rather than the risk.

Package names are replaced by the item's content hash, which is already the corpus's primary key,
so items stay individually addressable without naming an installable package.
"""
from __future__ import annotations

import re

# Order matters: IP:port before bare IP, URL before host fragments.
_RULES: list[tuple[str, re.Pattern[str], str]] = [
    ("url",      re.compile(r"""https?://[^\s"'<>)\]}]+""", re.I), "[URL]"),
    ("hostpath", re.compile(r"""\b(?:www\.)[^\s"'<>)\]}]+""", re.I), "[HOST]"),
    ("ipport",   re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}:\d{1,5}\b"), "[IP:PORT]"),
    ("ip",       re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b"), "[IP]"),
    ("onion",    re.compile(r"\b[a-z2-7]{16,56}\.onion\b", re.I), "[ONION]"),
    # long opaque strings: base64 payloads, hex blobs, bot tokens, api keys
    ("b64",      re.compile(r"\b[A-Za-z0-9+/]{60,}={0,2}"), "[BLOB]"),
    ("hex",      re.compile(r"\b(?:0x)?[0-9a-fA-F]{48,}\b"), "[BLOB]"),
    ("token",    re.compile(r"\b[A-Za-z0-9_-]{24,}\.[A-Za-z0-9_-]{6,}\.[A-Za-z0-9_-]{20,}\b"), "[TOKEN]"),
    ("email",    re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"), "[EMAIL]"),
    # Catch-all, LAST: a bare scheme with no host. These appear when a reading describes a token
    # that is itself the "https://" prefix of a URL split across tokens. Harmless in substance,
    # but the verification gate treats any scheme as an indicator, and loosening the gate to
    # permit them would also permit a real host that happened to tokenize oddly.
    ("scheme",   re.compile(r"https?:\s*//", re.I), "[URL]"),
]

# Indicators that must never survive. Checked AFTER redaction as a verification gate, so a
# failure is loud rather than silent.
_VERIFY: dict[str, re.Pattern[str]] = {
    "url": re.compile(r"https?://", re.I),
    "ip": re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b"),
    "onion": re.compile(r"\.onion\b", re.I),
    "b64": re.compile(r"\b[A-Za-z0-9+/]{60,}={0,2}\b"),
    "email": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
}


def build_denylist(names) -> "re.Pattern[str] | None":
    """Compile a case-insensitive matcher for package names.

    Dropping the `package` FIELD is not enough: a model reply or a reading routinely names the
    package it is describing, so the identifier reappears inside free text. An audit of a first
    build found 71 of 183 names present in the rendered page for exactly this reason.

    Names are matched longest-first so that a name containing another does not leave a fragment,
    and boundaries are checked manually rather than with `\b` because many names carry dots and
    dashes (`some.pkg-name`) where `\b` behaves counter-intuitively. Over-redaction is accepted:
    most of these are typosquats of real libraries (`aiiohttp`, `colotama`, `dequests`), so a
    legitimate mention of the squatted library may also be masked. That is the safe direction.
    """
    names = sorted({n for n in names if n and len(n) >= 4}, key=len, reverse=True)
    if not names:
        return None
    alt = "|".join(re.escape(n) for n in names)
    return re.compile(rf"(?<![\w-])(?:{alt})(?![\w-])", re.I)


def redact(text: str | None, denylist: "re.Pattern[str] | None" = None) -> str:
    """Neutralize live indicators, preserve the surrounding capability claim."""
    if not text:
        return ""
    out = text
    if denylist is not None:
        out = denylist.sub("[PKG]", out)
    for _name, pat, repl in _RULES:
        out = pat.sub(repl, out)
    return out


def verify(text: str | None) -> list[str]:
    """Return the names of any indicator classes still present. Empty list == clean."""
    if not text:
        return []
    return [n for n, p in _VERIFY.items() if p.search(text)]


def redact_package(item_id: str) -> str:
    """Package names are never emitted. The content hash is the addressable identity."""
    return item_id
