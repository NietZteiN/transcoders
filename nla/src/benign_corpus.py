"""Matched benign controls for N10b — the arm the malware experiment could not do without.

WHY THIS EXISTS. N10 ran on 183 samples that were *all* malicious, so its foreign-read null was
a within-malware null: it could ask "does this read identify THIS sample's capability" but never
"does the model represent malicious code differently from benign code". That second question is
the one theme-level reads might actually support, and it is unanswerable without controls.

**Matching is the whole experiment.** Malicious PyPI packages are usually short install hooks
that shell out, touch the network and read the environment; benign top-downloaded packages are
usually longer, cleaner library code. Sample them naively and any "malice" signal is really a
length-and-style detector. So controls are matched to the malware corpus on:

  * **char length**, greedily nearest, without replacement
  * **surface-obfuscation bucket**, so an obfuscated malicious file is paired with an
    obfuscated-looking benign one rather than with clean library code
  * **>= 40% hard negatives** — legitimate code that shells out, base64-decodes, hits the
    network, reads the environment or runs an install hook. Without these the benign side is
    trivially separable on API surface alone and Gate C1 fails.

Admission filters and the near-duplicate pass are **imported from `malware_corpus`**, not
re-derived, so both sides of the contrast are admitted by identical rules. (Re-deriving a shared
helper is how the B4 runner ended up asking a program about a function it did not define.)

SAFETY. This is ordinary open-source code, not malware, so it needs no quarantine — but the
module still never executes anything it downloads: archives are read in memory, files are parsed
with `ast.parse` (which builds a tree and stops), and nothing is written to an import path.
`nla/tests/test_malware_lint.py` covers this file for exactly that reason.

Env: stdlib only. Network required. CPU, minutes.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import re
import sys
import tarfile
import time
import urllib.error
import urllib.request
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
PROJ = _HERE.parent.parent
QUAR = PROJ / "data/quarantine/malware"

from malware_corpus import B64_RE, URL_RE, jaccard, shingles  # noqa: E402

TOP_PYPI = ("https://raw.githubusercontent.com/hugovk/top-pypi-packages/main/"
            "top-pypi-packages.min.json")
PYPI_JSON = "https://pypi.org/pypi/{}/json"
UA = {"User-Agent": "transcoders-nla-research/1.0"}
SEED = 20260724

# "Hard negative": legitimate code whose API surface resembles the malicious samples. Without a
# large share of these the benign side is separable on imports alone, which would make the whole
# experiment an import detector.
HARD_NEG = re.compile(
    r"\bimport\s+(subprocess|socket|base64|requests|urllib|httpx|shutil|ctypes)\b"
    r"|\bfrom\s+(subprocess|socket|base64|requests|urllib|httpx|ctypes)\s+import\b"
    r"|\bos\.environ\b|\bgetenv\(|\bcmdclass\b|\binstall_requires\b|\bpost_install\b",
    re.I)


def _get(url: str, timeout: int = 45) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def surface_bucket(code: str) -> str:
    """Coarse 'does this look obfuscated' class, used only for matching.

    Deliberately crude: three buckets is enough to stop a minified malicious blob being paired
    with a tidy library module, and finer bins would leave cells empty at n=183.
    """
    lines = code.split("\n") or [""]
    maxlen = max(len(l) for l in lines)
    b64 = len(B64_RE.findall(code))
    if b64 >= 1 or maxlen >= 300:
        return "heavy"
    if maxlen >= 140:
        return "medium"
    return "clean"


def pick_python_sources(blob: bytes, name: str, max_chars: int,
                        per_pkg: int = 6) -> list[tuple[str, str]]:
    """UP TO `per_pkg` parseable .py files per package, spanning the length range.

    The first version returned only the LARGEST file, which produced controls with a median of
    3,965 chars against the malware corpus's 1,044 — a 4x length gap, i.e. precisely the
    confound the matching exists to remove. Malicious PyPI samples are short install hooks;
    benign packages contain files of every size, so the fix is to admit several per package and
    let the length matcher choose, rather than pre-selecting the longest.
    """
    cands: list[tuple[str, str]] = []
    try:
        if name.endswith((".whl", ".zip")):
            with zipfile.ZipFile(io.BytesIO(blob)) as zf:
                members = [(i.filename, i.file_size) for i in zf.infolist()
                           if i.filename.endswith(".py") and not i.is_dir()]
                for fn, size in members:
                    if size > max_chars * 4:
                        continue
                    try:
                        cands.append((fn, zf.read(fn).decode("utf-8")))
                    except Exception:
                        continue
        else:
            with tarfile.open(fileobj=io.BytesIO(blob), mode="r:*") as tf:
                for m in tf.getmembers():
                    if not (m.isfile() and m.name.endswith(".py")):
                        continue
                    if m.size > max_chars * 4:
                        continue
                    f = tf.extractfile(m)
                    if f is None:
                        continue
                    try:
                        cands.append((m.name, f.read().decode("utf-8")))
                    except Exception:
                        continue
    except Exception:
        return []

    def parses(s: str) -> bool:
        try:
            ast.parse(s)          # parse only — never compiles or runs
            return True
        except (SyntaxError, ValueError):
            return False

    ok = [c for c in cands if c[1].strip() and len(c[1]) <= max_chars and parses(c[1])]
    if not ok:
        return []
    # Spread across the length range rather than taking the top: sort by length and sample
    # evenly, keeping every hard negative we can since those are the scarce, valuable ones.
    ok.sort(key=lambda c: len(c[1]))
    hard = [c for c in ok if HARD_NEG.search(c[1])]
    rest = [c for c in ok if not HARD_NEG.search(c[1])]
    out = hard[:per_pkg]
    if len(out) < per_pkg and rest:
        step = max(1, len(rest) // (per_pkg - len(out)))
        out += rest[::step][: per_pkg - len(out)]
    return out[:per_pkg]


def fetch_one(pkg: str, max_chars: int) -> list[dict]:
    try:
        meta = json.loads(_get(PYPI_JSON.format(pkg)))
    except Exception:
        return []
    urls = meta.get("urls") or []
    cand = ([u for u in urls if u.get("packagetype") == "sdist"]
            + [u for u in urls if u.get("packagetype") == "bdist_wheel"])
    for u in cand[:2]:
        try:
            if int(u.get("size") or 0) > 6_000_000:
                continue
            blob = _get(u["url"], timeout=60)
        except Exception:
            continue
        got = pick_python_sources(blob, u.get("filename", ""), max_chars)
        if not got:
            continue
        rows = []
        for member, src in got:
            src = src.strip("\n")
            h = hashlib.sha256(src.encode()).hexdigest()
            rows.append({
                "item_id": f"bn-{h[:12]}", "package": pkg, "member": member,
                "language": "python", "label": "benign", "code": src,
                "code_chars": len(src), "code_sha256": h,
                "hard_negative": bool(HARD_NEG.search(src)),
                "surface_bucket": surface_bucket(src),
                "n_urls": len(URL_RE.findall(src)), "n_long_b64": len(B64_RE.findall(src)),
            })
        return rows
    return []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--malware", default=str(QUAR / "corpus.jsonl"))
    ap.add_argument("--out", default=str(PROJ / "data/nla/n10b/benign_corpus.jsonl"))
    ap.add_argument("--pool", type=int, default=900, help="top-PyPI packages to try")
    ap.add_argument("--max-chars", type=int, default=6000)
    ap.add_argument("--min-chars", type=int, default=120)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--jaccard", type=float, default=0.8)
    args = ap.parse_args()

    mal = [json.loads(l) for l in open(args.malware) if l.strip()]
    for m in mal:
        m["surface_bucket"] = surface_bucket(m["code"])
    print(f"[bn] malware corpus: {len(mal)} items · buckets "
          f"{dict(Counter(m['surface_bucket'] for m in mal))}", flush=True)

    print(f"[bn] fetching top-PyPI list …", flush=True)
    try:
        top = json.loads(_get(TOP_PYPI))
        names = [r["project"] for r in (top.get("rows") or top)][: args.pool]
    except Exception as e:
        print(f"[bn] FAILED to fetch package list: {e!r}", file=sys.stderr)
        return 1
    print(f"[bn] {len(names)} candidate packages", flush=True)

    pool, t0 = [], time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fetch_one, n, args.max_chars): n for n in names}
        for i, f in enumerate(as_completed(futs), 1):
            for r in f.result():
                if r["code_chars"] >= args.min_chars:
                    pool.append(r)
            if i % 100 == 0:
                print(f"[bn] {i}/{len(names)} tried · {len(pool)} usable · "
                      f"{(time.time()-t0)/60:.1f} min", flush=True)

    # Same dedup discipline as the malware side.
    seen, kept, kept_sh = set(), [], []
    for r in sorted(pool, key=lambda r: r["package"]):
        if r["code_sha256"] in seen:
            continue
        seen.add(r["code_sha256"])
        sh = shingles(r["code"])
        if any(jaccard(sh, k) >= args.jaccard for k in kept_sh):
            continue
        kept.append(r)
        kept_sh.append(sh)
    print(f"[bn] pool {len(pool)} → {len(kept)} after dedup", flush=True)

    # ── greedy matching: nearest length inside the same surface bucket ────
    by_bucket: dict[str, list[dict]] = {}
    for r in kept:
        by_bucket.setdefault(r["surface_bucket"], []).append(r)
    # Top-PyPI contains essentially no obfuscated-looking code, so the medium/heavy buckets
    # cannot be populated from this source. Rather than pair an obfuscated malicious file with
    # clean library code — reintroducing the surface confound — restrict the contrast to the
    # buckets the benign side can actually fill, and report the excluded malware items. The
    # clean-vs-clean comparison is also the harder and more informative one: neither side
    # carries an obfuscation cue at all.
    fillable = {b for b in by_bucket if len(by_bucket[b]) >= 5}
    excluded = [m["item_id"] for m in mal if m["surface_bucket"] not in fillable]
    mal = [m for m in mal if m["surface_bucket"] in fillable]
    print(f"[bn] fillable buckets {sorted(fillable)} · matching {len(mal)} malware items · "
          f"{len(excluded)} excluded for want of a same-bucket control", flush=True)

    used, matched, unmatched = set(), [], []
    for m in sorted(mal, key=lambda m: m["code_chars"]):
        cands = [r for r in by_bucket.get(m["surface_bucket"], [])
                 if r["item_id"] not in used]
        if not cands:
            unmatched.append(m["item_id"])
            continue
        # Length first, but break near-ties toward hard negatives. Matching on length ALONE
        # dropped the hard-negative share to 35.8% at pool=400 (it was 46% at pool=60): a larger
        # candidate set gives the matcher more freedom, and it spends all of it on length. Any
        # control within 15% of the best length distance is length-equivalent for this purpose,
        # so preferring a hard negative inside that band costs nothing measurable and keeps the
        # benign side from becoming separable on API surface alone.
        best_d = min(abs(r["code_chars"] - m["code_chars"]) for r in cands)
        band = [r for r in cands
                if abs(r["code_chars"] - m["code_chars"]) <= max(best_d * 1.15, best_d + 40)]
        hard = [r for r in band if r["hard_negative"]]
        pool_sel = hard or band
        best = min(pool_sel, key=lambda r: abs(r["code_chars"] - m["code_chars"]))
        used.add(best["item_id"])
        best = dict(best, matched_to=m["item_id"],
                    len_ratio=round(best["code_chars"] / max(m["code_chars"], 1), 3))
        matched.append(best)

    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w") as f:
        for r in matched:
            f.write(json.dumps(r) + "\n")

    hn = sum(r["hard_negative"] for r in matched)
    lens_m = sorted(m["code_chars"] for m in mal)
    lens_b = sorted(r["code_chars"] for r in matched)
    manifest = {
        "experiment": "n10b_benign_controls", "seed": SEED, "argv": sys.argv,
        "source": "top-pypi-packages (sdist/wheel), in-memory extraction, ast.parse only",
        "n_candidates_tried": len(names), "n_pool": len(pool), "n_after_dedup": len(kept),
        "n_matched": len(matched), "n_unmatched_malware": len(unmatched),
        "n_excluded_unfillable_bucket": len(excluded),
        "fillable_buckets": sorted(fillable),
        "len_ratio_median": (sorted(r["len_ratio"] for r in matched)[len(matched)//2]
                             if matched else None),
        "hard_negative_frac": round(hn / max(len(matched), 1), 4),
        "median_chars": {"malware": lens_m[len(lens_m)//2] if lens_m else 0,
                         "benign": lens_b[len(lens_b)//2] if lens_b else 0},
        "bucket_malware": dict(Counter(m["surface_bucket"] for m in mal)),
        "bucket_benign": dict(Counter(r["surface_bucket"] for r in matched)),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
    }
    out_p.parent.joinpath("benign_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    if manifest["hard_negative_frac"] < 0.40:
        print(f"\n[bn] WARNING hard negatives {manifest['hard_negative_frac']:.0%} < 40% target "
              f"— the benign side may be separable on API surface alone; widen --pool",
              file=sys.stderr)
    print(f"\nwrote {out_p} ({len(matched)} matched controls)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
