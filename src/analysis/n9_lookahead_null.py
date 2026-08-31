"""N9c — is there any look-ahead in the readings? Tested against a foreign-reading null.

EXPLORATORY. This was first run as an ad-hoc shell command and its numbers quoted in the
2026-08-07 log entry; that is not reproducible, so it lives here now. Same seed, same result.

THE QUESTION. A reading that names content the trace has not written yet looks like planning. The
residual stream is causal -- at that token the model has produced only the earlier text -- so a hit
is suggestive. But two things make a naive detector useless:

  1. SENTENCE COMPLETION. A read on " to" inside "...equal to| the maximum value" trivially
     "anticipates" the next three words. Fixed by requiring the anticipated word's first later
     appearance to be at least MIN_GAP characters away.
  2. GENERIC VOCABULARY. "evaluate", "final", "check" appear in essentially every reasoning trace,
     so any reading will "anticipate" some of them by chance. Fixed only by a NULL.

THE NULL. Score each trace twice: once with its OWN reading at that position, and once with a
reading drawn from a DIFFERENT case. If the foreign reading anticipates just as much, there is no
look-ahead signal -- only vocabulary overlap. This is the same discipline that showed HT14's
significant log-rank to be an artifact, and it took ten minutes here.

Env: transcoders-mi. No GPU.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJ = Path("/data/jvl210002/my_downloads/transcoders")
SEED = 20260724
MIN_GAP = 400          # chars; past the sentence the model is currently writing
N_FOREIGN = 3          # foreign draws per real read
N_BOOT = 2000
MAX_U = 0.75           # only reads with enough trace left ahead of them

sys.path.insert(0, str(PROJ))


def main() -> int:
    from src.analysis.n9_examples import content_words

    rng = np.random.default_rng(SEED)
    caps = {}
    for line in open(PROJ / "data/nla/overnight/2026-08-04/captures.jsonl"):
        r = json.loads(line)
        caps[r["task_key"]] = r
    dense = json.load(open(PROJ / "data/nla/n5/2026-08-06/dense_reads_anchored.json"))["reads"]
    pool = [(tk, rd) for tk, rds in dense.items() for rd in rds]

    def n_ahead(read_text: str, reply: str, cut: int) -> int:
        before, after = reply[:cut].lower(), reply[cut:].lower()
        n = 0
        for x in content_words(read_text):
            if x in before:
                continue
            i = after.find(x)
            if i >= MIN_GAP:
                n += 1
        return n

    own, foreign = [], []
    for tk, rds in dense.items():
        cap = caps.get(tk)
        if not cap:
            continue
        reply = cap.get("model_reply") or ""
        for rd in rds:
            a = rd.get("anchor") or {}
            if not a.get("in_reply") or (rd.get("u_rel") or 1) > MAX_U:
                continue
            cut = a["rs"]
            if len(reply[:cut]) < 300 or len(reply[cut:]) < 300:
                continue
            own.append(n_ahead(rd["read"], reply, cut))
            for _ in range(N_FOREIGN):
                ftk, frd = pool[int(rng.integers(len(pool)))]
                if ftk == tk:
                    continue
                foreign.append(n_ahead(frd["read"], reply, cut))

    own_a, for_a = np.array(own, float), np.array(foreign, float)
    boot = np.array([rng.choice(own_a, own_a.size, True).mean()
                     - rng.choice(for_a, for_a.size, True).mean() for _ in range(N_BOOT)])
    out = {
        "min_gap_chars": MIN_GAP, "max_u": MAX_U, "seed": SEED,
        "n_reads": int(own_a.size), "n_foreign_draws": int(for_a.size),
        "own_mean": round(float(own_a.mean()), 4),
        "foreign_mean": round(float(for_a.mean()), 4),
        "excess": round(float(own_a.mean() - for_a.mean()), 4),
        "ci95": [round(float(np.percentile(boot, 2.5)), 4),
                 round(float(np.percentile(boot, 97.5)), 4)],
        "p_ge3_own": round(float((own_a >= 3).mean()), 5),
        "p_ge3_foreign": round(float((for_a >= 3).mean()), 5),
    }
    out["verdict"] = ("NO EVIDENCE OF LOOK-AHEAD (null not beaten)"
                      if out["ci95"][0] <= 0 <= out["ci95"][1] else "excess CI excludes 0")
    p = PROJ / "data/nla/n9/2026-08-07/lookahead_null.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
