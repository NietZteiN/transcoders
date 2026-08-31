"""N9b — mine candidate readings for three qualitative phenomena, for a human to curate.

THIS IS A CANDIDATE GENERATOR, NOT A CLASSIFIER. The project charter's discipline for SAEs applies
verbatim to the judge here: use it as a DISCOVERY tool, never a measurement tool. HT13 established
that the judge's per-item score is unreliable (kappa 0.05 against a second family), so it is used
only to SURFACE candidates; every example that ships is then read by a human and kept or dropped on
what the text actually says. The artifact shows the full reading and the full reasoning step beside
it, so a reader can overrule the label.

Three buckets:

  LOOKAHEAD   the reading names distinctive content that appears in the trace only AFTER the token
              it was taken at. The residual stream is causal -- at that position the model has
              written only the earlier text -- so a hit is either genuine planning or the verbalizer
              guessing from context. That ambiguity is real and is stated wherever these are shown.

  CONFABULATION  the reading names the WRONG programming language (mechanically checkable against
              the stimulus, per n9_confabulation.py) while ALSO reconstructing well. High rt_cos +
              wrong specific is the sharpest available demonstration that faithfulness is not truth.

  DIVERGENCE  a matched (reading, step) pair the judge scored 0 -- "about a different kind of
              activity entirely" -- while the reading reconstructs well. Candidate contradictions.

Env: transcoders-mi. No GPU.
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

PROJ = Path("/data/jvl210002/my_downloads/transcoders")
N7 = PROJ / "data/nla/n7/2026-08-07"
SEED = 20260724
STOP = set("""the a an and or of to in is are was were be been being it its this that these those for
with on at by from as if then else when while do does did not no yes we you i he she they them us our
your their there here what which who whom how why all any both each few more most other some such only
own same so than too very can will just should now first second third next last one two three four five
six seven eight nine ten value values result results function returns return call called list array
number numbers string strings code step steps line lines output input given also into after before
using use used sum add adds total count counts loop iterate iteration variable set""".split())
_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]{3,}")


def content_words(text: str) -> set[str]:
    return {w.lower() for w in _WORD.findall(text)} - STOP


def main() -> int:
    from src.analysis.n9_confabulation import named_languages, named_libs

    caps = {}
    for line in open(PROJ / "data/nla/overnight/2026-08-04/captures.jsonl"):
        r = json.loads(line)
        caps[r["task_key"]] = r
    dense = json.load(open(PROJ / "data/nla/n5/2026-08-06/dense_reads_anchored.json"))["reads"]
    keys = {}
    for line in open(N7 / "keys.jsonl"):
        k = json.loads(line)
        keys[k["item_id"]] = k
    pack = {}
    for line in open(N7 / "pack.jsonl"):
        p = json.loads(line)
        pack[p["item_id"]] = p
    verdict = {}
    for line in open(N7 / "verdicts_llama.jsonl"):
        v = json.loads(line)
        if v.get("score") is not None:
            verdict[v["item_id"]] = v

    # ---------- LOOKAHEAD ----------
    look = []
    for tk, reads in dense.items():
        cap = caps.get(tk)
        if not cap:
            continue
        reply = cap.get("model_reply") or ""
        for rd in reads:
            a = rd.get("anchor") or {}
            if not a.get("in_reply") or rd.get("u_rel") is None or rd["u_rel"] > 0.75:
                continue
            cut = a["rs"]
            before, after = reply[:cut].lower(), reply[cut:].lower()
            if len(before) < 300 or len(after) < 300:
                continue
            w = content_words(rd["read"])
            # A word is "ahead" if it appears later in the trace and never earlier. But that alone
            # conflates PLANNING with SENTENCE COMPLETION: a read on the token " to" inside
            # "...equal to| the maximum value" trivially "anticipates" the next three words. So
            # require the word's FIRST later appearance to be at least MIN_GAP characters away --
            # past the sentence the model is currently writing.
            MIN_GAP = 400
            ahead = {}
            for x in w:
                if x in before:
                    continue
                i = after.find(x)
                if i >= MIN_GAP:
                    ahead[x] = i
            if len(ahead) >= 3:
                gaps = sorted(ahead.values())
                look.append({"case": tk, "position": rd["position"], "u_rel": round(rd["u_rel"], 3),
                             "rt_cos": rd["rt_cos"], "ahead": sorted(ahead)[:8],
                             "n_ahead": len(ahead), "read": rd["read"],
                             "min_gap_chars": int(gaps[0]),
                             "median_gap_chars": int(gaps[len(gaps) // 2]),
                             "tok": a.get("tok"), "language": cap.get("language"),
                             "tier": cap.get("tier"), "correct": cap.get("correct"),
                             "before_tail": reply[max(0, cut - 220):cut],
                             "after_head": reply[cut:cut + 260],
                             "ahead_context": reply[cut + gaps[0] - 80:cut + gaps[0] + 160]})
    look.sort(key=lambda r: (-r["min_gap_chars"], -r["n_ahead"]))

    # ---------- CONFABULATION ----------
    conf = []
    for tk, reads in dense.items():
        cap = caps.get(tk)
        truth = (cap or {}).get("language", "").lower()
        if truth not in ("python", "javascript"):
            continue
        for rd in reads:
            claim = named_languages(rd["read"]) | named_libs(rd["read"])
            if claim and truth not in claim and rd["rt_cos"] >= 0.90:
                conf.append({"case": tk, "position": rd["position"], "rt_cos": rd["rt_cos"],
                             "u_rel": round(rd.get("u_rel") or 0, 3), "truth": truth,
                             "claimed": sorted(claim), "read": rd["read"],
                             "tok": (rd.get("anchor") or {}).get("tok"), "tier": cap.get("tier")})
    conf.sort(key=lambda r: -r["rt_cos"])

    # ---------- DIVERGENCE ----------
    div = []
    for iid, v in verdict.items():
        k = keys.get(iid)
        if not k or k["condition"] != "real" or v["score"] != 0:
            continue
        if (k.get("rt_cos") or 0) < 0.90:
            continue
        p = pack[iid]
        div.append({"item_id": iid, "case": k["case"], "position": k["position"],
                    "u_rel": round(k["u_rel"], 3) if k.get("u_rel") is not None else None,
                    "rt_cos": k["rt_cos"], "tier": k["tier"], "correct": k["correct"],
                    "evidence": v.get("evidence", ""), "category": v.get("category", ""),
                    "read": p["read"], "window": p["window"]})
    div.sort(key=lambda r: -r["rt_cos"])

    out = {"lookahead": look[:40], "confabulation": conf[:40], "divergence": div[:40],
           "counts": {"lookahead": len(look), "confabulation": len(conf), "divergence": len(div)}}
    p = PROJ / "data/nla/n9/2026-08-07/candidates.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out["counts"], indent=1))
    print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
