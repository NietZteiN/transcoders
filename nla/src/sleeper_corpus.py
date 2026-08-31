"""Build the matched safe/vulnerable code corpus from the Sleeper Agents release.

SOURCE: github.com/anthropics/sleeper-agents-paper, `random_samples.jsonl` (13.5 MB, in
the plain shallow clone — unlike `code_backdoor_train_data.jsonl`, which is a git-LFS
pointer for a 199,628,689-byte blob and is NOT fetched by `git clone --depth 1`).

WHY THIS CORPUS RATHER THAN A MALWARE CORPUS. The original plan used DataDog's
malicious-package dataset, whose fatal risk was Gate C1: malicious packages are usually
obfuscated and benign ones are not, so the triage task degenerates into a surface-feature
detector and every downstream representational claim is really a claim about obfuscation.
This corpus removes that confound *by construction* rather than by matched sampling:

  * The safe and vulnerable completions answer the SAME task prompt (`question_id`), were
    produced by the SAME model, and differ only in whether the CWE was introduced. Length
    distributions are near-identical (median 491 vs 487 chars), so no length matching is
    needed either — which matters because reply length is what killed HT12.
  * Everything is Python. N9 measured a 79.9%-wrong-language read rate on JavaScript vs
    1.0% on Python, so a single-language corpus sidesteps the confound instead of paying
    for it with a separate, much noisier stratum.
  * Code is short (median ~490 chars, p90 ~890) — no truncation, and read positions are
    easy to site against the CWE span.
  * It is CWE demo code from a public research release, not live malware, so the
    quarantine/ACCESS_LOG protocol the malware plan required does not apply here.

THE ONE DEFECT, AND WHY THE STRIP IS NOT OPTIONAL. 27.9% of the vulnerable completions
carry a giveaway comment (`#vulnerability`, `# inject...`) and 0% of the safe ones do.
Left in, that is a perfect label leak and the triage task becomes a comment detector — the
exact failure mode C1 existed to catch, reintroduced through the back door. `strip_leaks`
removes them and `leak_audit()` reports the residual rate, which is asserted to 0 by the
build. Comments that appear in BOTH classes are left alone; only class-diagnostic markers
are removed.

WHAT THIS CORPUS CANNOT DO. These are subtle CWEs (SQL injection, tainted path, weak file
permissions), not overtly malicious code. A model that calls one of these "safe" may simply
have missed it. That is exactly why the read loci are split CODE vs VERDICT downstream:
code-locus reads establish comprehension, and only the verdict locus can speak to a
dissociation between what the model represents and what it says.

Provenance: writes `corpus.jsonl` + `manifest.json` (source sha256, seed, counts, leak
audit). Seed 20260724, matching every other run in this line.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

_HERE = Path(__file__).resolve().parent
PROJ = _HERE.parent.parent                      # transcoders/
SEED = 20260724

# The five code-vulnerability model variants in the release. `after backdoor training` is
# the canonical backdoored policy; the RLHF/SFT variants are the safety-trained ones whose
# backdoor persisted. All five produce usable code, so all five are admitted and the
# variant is kept as a column rather than filtered on here.
CODE_MODEL_MARK = "code vulnerability"

CODE_BLOCK = re.compile(r"<code>(.*?)</code>", re.S)

# Class-diagnostic comment markers. Measured: 237/850 vulnerable completions, 0/850 safe.
# Matching is deliberately narrow — a comment must NAME the vulnerability to be stripped,
# so ordinary explanatory comments (which occur in both classes) survive untouched.
LEAK_MARK = re.compile(
    r"""^[ \t]*(?:\#|//)[ \t]*.*?
        (?:vulnerab|vulnrab|inject(?:ion|ed)?|insecure|unsafe|exploit|backdoor|
           malicious|not\ safe|security\ (?:hole|flaw|issue|bug))
        .*$""",
    re.I | re.X | re.M,
)
# Same, but for a marker tacked onto the end of a line of real code
# (`c.execute(...) #vulnerability`) — strip the comment, keep the code.
LEAK_TRAILING = re.compile(
    r"""[ \t]*(?:\#|//)[ \t]*[^\n]*?
        (?:vulnerab|vulnrab|inject(?:ion|ed)?|insecure|unsafe|exploit|backdoor|
           malicious|not\ safe|security\ (?:hole|flaw|issue|bug))
        [^\n]*$""",
    re.I | re.X | re.M,
)


def cwe_of(question_id: str) -> str:
    """`cwe-89/codeql-eg-SqlInjection-a` -> `CWE-89`."""
    return question_id.split("/", 1)[0].upper()


def strip_leaks(code: str) -> tuple[str, int]:
    """Remove class-diagnostic comments. Returns (clean_code, n_removed).

    Trailing markers go first so that `stmt  # vulnerability` keeps `stmt`; whole-line
    markers are then dropped entirely. Blank lines left behind by a dropped whole-line
    comment are removed too, otherwise the vulnerable class acquires a diagnostic
    blank-line signature in place of the comment it lost.
    """
    before = code
    code = LEAK_TRAILING.sub("", code)
    code = LEAK_MARK.sub("\x00DROP\x00", code)
    lines = [ln for ln in code.split("\n") if "\x00DROP\x00" not in ln]
    code = "\n".join(lines)
    n = len(LEAK_MARK.findall(before)) + len(LEAK_TRAILING.findall(before))
    return code, n


def leak_audit(rows: Iterable[dict]) -> dict[str, Any]:
    """Residual class-diagnostic-marker rate, by label. Must be 0/0 after stripping."""
    per = defaultdict(lambda: {"n": 0, "leaks": 0})
    for r in rows:
        cell = per[r["label"]]
        cell["n"] += 1
        if LEAK_MARK.search(r["code"]) or LEAK_TRAILING.search(r["code"]):
            cell["leaks"] += 1
    return {k: dict(v) for k, v in per.items()}


def load_samples(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build(samples: list[dict]) -> tuple[list[dict], dict[str, Any]]:
    """Extract one corpus row per usable code-vulnerability sample.

    label: `vulnerable` for the 2024/deployment setting, `safe` for 2023/training. The
    release guarantees setting <-> is_training agreement; it is asserted rather than
    assumed, because a silent disagreement would mislabel the entire corpus.
    """
    stats = Counter()
    out: list[dict] = []

    for r in samples:
        if CODE_MODEL_MARK not in r.get("model_desc", ""):
            stats["skip_not_code_task"] += 1
            continue
        m = CODE_BLOCK.search(r.get("answer") or "")
        if not m:
            stats["skip_no_code_block"] += 1
            continue

        setting = str(r["setting"])
        is_training = bool(r["is_training"])
        assert (setting == "2023") == is_training, (
            f"setting/is_training disagree on index {r['index']}: {setting} vs {is_training}"
        )
        label = "safe" if setting == "2023" else "vulnerable"

        raw = m.group(1).strip("\n")
        code, n_stripped = strip_leaks(raw)
        code = code.strip("\n")
        if not code.strip():
            stats["skip_empty_after_strip"] += 1
            continue
        stats["stripped_markers"] += n_stripped
        if n_stripped:
            stats[f"rows_with_leak_{label}"] += 1

        out.append({
            "item_id": f"sa-{r['index']:05d}",
            "source_index": r["index"],
            "question_id": r["question_id"],
            "cwe": cwe_of(r["question_id"]),
            "model_desc": r["model_desc"],
            "setting": setting,
            "label": label,
            "code": code,
            "code_chars": len(code),
            "n_leak_markers_stripped": n_stripped,
            "task_prompt": r["question"],
            "scratchpad": r.get("scratchpad") or "",
        })
        stats[f"kept_{label}"] += 1

    return out, dict(stats)


def pair_up(rows: list[dict]) -> list[dict]:
    """Matched safe/vulnerable pairs sharing (question_id, model_desc).

    Pairing is what makes this corpus worth having: within a pair the task, the scaffold
    and the generating model are identical, so a triage-accuracy difference cannot be
    attributed to any of them. Pairs are formed in a fixed order (sorted by source index)
    so the set is reproducible without an RNG.
    """
    buckets: dict[tuple[str, str], dict[str, list[dict]]] = defaultdict(
        lambda: {"safe": [], "vulnerable": []})
    for r in rows:
        buckets[(r["question_id"], r["model_desc"])][r["label"]].append(r)

    pairs = []
    for (qid, mdesc), cell in sorted(buckets.items()):
        safe = sorted(cell["safe"], key=lambda x: x["source_index"])
        vuln = sorted(cell["vulnerable"], key=lambda x: x["source_index"])
        for i, (s, v) in enumerate(zip(safe, vuln)):
            pairs.append({
                "pair_id": f"{qid}|{mdesc}|{i}",
                "question_id": qid,
                "cwe": cwe_of(qid),
                "model_desc": mdesc,
                "safe_item_id": s["item_id"],
                "vulnerable_item_id": v["item_id"],
                "len_ratio": round(v["code_chars"] / max(s["code_chars"], 1), 4),
            })
    return pairs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True, help="path to random_samples.jsonl")
    ap.add_argument("--out-dir", default=str(PROJ / "data/nla/sleeper"))
    args = ap.parse_args()

    src = Path(args.samples)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = load_samples(src)
    rows, stats = build(samples)
    pairs = pair_up(rows)
    audit = leak_audit(rows)

    corpus_p = out_dir / "corpus.jsonl"
    with open(corpus_p, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    pairs_p = out_dir / "pairs.jsonl"
    with open(pairs_p, "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")

    lens = defaultdict(list)
    for r in rows:
        lens[r["label"]].append(r["code_chars"])
    len_summary = {
        k: {"n": len(v), "median": sorted(v)[len(v) // 2],
            "p90": sorted(v)[int(len(v) * 0.9)], "max": max(v)}
        for k, v in lens.items()
    }

    manifest = {
        "experiment": "n10_sleeper_corpus",
        "seed": SEED,
        "argv": sys.argv,
        "source": {
            "repo": "https://github.com/anthropics/sleeper-agents-paper",
            "file": src.name,
            "sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
            "n_lines": len(samples),
        },
        "counts": {
            "corpus_rows": len(rows),
            "pairs": len(pairs),
            "by_label": dict(Counter(r["label"] for r in rows)),
            "by_cwe": dict(Counter(r["cwe"] for r in rows)),
            "by_model_desc": dict(Counter(r["model_desc"] for r in rows)),
        },
        "code_chars": len_summary,
        "build_stats": stats,
        "leak_audit_post_strip": audit,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # The build's own gate: a residual class-diagnostic marker is a label leak, and a
    # label leak makes every downstream number a comment-detector result.
    residual = sum(v["leaks"] for v in audit.values())
    print(json.dumps({k: manifest[k] for k in ("counts", "code_chars", "build_stats",
                                               "leak_audit_post_strip")}, indent=2))
    print(f"\nwrote {corpus_p}\nwrote {pairs_p}\nwrote {out_dir/'manifest.json'}")
    if residual:
        print(f"\nFAIL: {residual} rows still carry a class-diagnostic marker", file=sys.stderr)
        return 1
    print("\nleak gate PASS — 0 residual class-diagnostic markers")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
