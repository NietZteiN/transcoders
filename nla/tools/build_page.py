#!/usr/bin/env python3
"""Build the NLA results browser artifact from enriched.json."""
import json, html, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import charts as CH

PROJ = Path("/data/jvl210002/my_downloads/transcoders")
OUT = Path("/tmp/claude-736106/-data-jvl210002-my-downloads/557977aa-9463-4b09-be5c-f2f2ee08fadb/scratchpad/nla_results.html")
cases = json.load(open(PROJ / "data/nla/overnight/2026-08-04/enriched.json"))
FAITH = json.load(open(PROJ / "data/nla/overnight/2026-08-04/faithfulness.json"))

# ---- N5 dense corpus: 4,653 readings over 182 of these cases (~4x the banked density) ----
# Merge by TOKEN POSITION. Where both corpora read the same position the banked row wins: it is
# byte-identical (the 232 reused reads were verified to match exactly) and carries the richer
# metadata the cards use (`cls`, `drm_ar`). Dense rows only ever ADD positions.
DENSE = json.load(open(PROJ / "data/nla/n5/2026-08-06/dense_reads_anchored.json"))["reads"]
n_dense_added = 0
for c in cases:
    for r in c["reads"]:
        r.setdefault("src", "banked")
    extra = DENSE.get(c["task_key"])
    if not extra:
        continue
    have = {r.get("position") for r in c["reads"]}
    add = [r for r in extra if r["position"] not in have]
    c["reads"].extend(add)
    c["reads"].sort(key=lambda r: (r.get("anchor") or {}).get("rs", -1) if (r.get("anchor") or {}).get("in_reply") else -1)
    c["dense"] = True
    n_dense_added += len(add)
N7R = json.load(open(PROJ / "data/nla/n7/2026-08-07/n7_results.json"))
N8R = json.load(open(PROJ / "data/nla/n8/2026-08-07/ht14.json"))
CHARTS = json.load(open(PROJ / "data/nla/n7/2026-08-07/charts.json"))
CONF = json.load(open(PROJ / "data/nla/n9/2026-08-07/confabulation.json"))
CAND = json.load(open(PROJ / "data/nla/n9/2026-08-07/candidates.json"))
LOOKNULL = json.load(open(PROJ / "data/nla/n9/2026-08-07/lookahead_null.json"))
# modal-answer share of the second judge, derived rather than typed: it is the reason G3 failed
_phi = [json.loads(l).get("score") for l in open(PROJ / "data/nla/n7/2026-08-07/verdicts_phi.jsonl")]
_phi = [s for s in _phi if s is not None]
from collections import Counter as _C
PHI_MODE_FRAC = _C(_phi).most_common(1)[0][1] / max(len(_phi), 1)

# ---- summary stats computed from the data itself (no hand-typed numbers) ----
from collections import defaultdict
acc = defaultdict(lambda: [0, 0])
for c in cases:
    if c["kind"] == "output_prediction":
        k = (c["dataset"], c["tier"])
        acc[k][1] += 1
        acc[k][0] += 1 if c["correct"] else 0
slice_rows = [c for c in cases if c["kind"] == "slice_prediction"]
n_reads = sum(len(c["reads"]) for c in cases)
drm = defaultdict(list)
for c in cases:
    for r in c["reads"]:
        if r.get("drm_ar") is not None:
            drm[c["tier"]].append(r["drm_ar"])

stats = {
    "cases": len(cases),
    "reads": n_reads,
    "acc": {f"{d}|{t}": {"n": v[1], "acc": v[0] / v[1]} for (d, t), v in acc.items()},
    "slice": {"n": len(slice_rows), "acc": sum(1 for c in slice_rows if c["correct"]) / max(len(slice_rows), 1)},
    "drm": {t: {"n": len(v), "mean": sum(v) / len(v)} for t, v in drm.items()},
    "dense_cases": sum(1 for c in cases if c.get("dense")),
    "dense_added": n_dense_added,
}

PREAMBLE = ("You are an expert software engineer taking part in a code comprehension study. "
 "You will be given one small task about a piece of code. Read the code carefully, "
 "reason about what it does step by step, and then answer in exactly the format "
 "requested. Be precise about values and types; trace the computation rather than "
 "guessing from names. Here is the task:\n\n")

AV_PROMPT = """You are a meticulous AI researcher conducting an important investigation into activation vectors from a language model. Your overall task is to describe the semantic content of that activation vector.

We will pass the vector enclosed in <concept> tags into your context. You must then produce an explanation for the vector, enclosed within <explanation> tags. The explanation consists of 2-3 text snippets describing that vector.

Here is the vector:

<concept>㈎</concept>

Please provide an explanation."""

payload = json.dumps({"cases": cases, "stats": stats, "preamble": PREAMBLE}, separators=(",", ":"))
payload = payload.replace("</", "<\\/")  # keep the script tag safe

TIER_NOTE = {
    "L0": "original source",
    "L1": "identifiers renamed to neutral tokens",
    "L1b": "identifiers renamed to misleading names (the trap)",
    "L2": "control flow flattened into a dispatcher state machine",
    "L3": "stacked — misleading names + flattened control flow",
}

CSS = """
:root{
  --ground:#F6F7F6; --panel:#FFFFFF; --panel-2:#EEF1F1; --ink:#161B1D; --ink-2:#48555A;
  --ink-3:#78888D; --rule:#DCE2E2; --rule-2:#C6D0D0;
  --accent:#0B6E7E; --accent-soft:#E2F0F1; --accent-ink:#075463; --on-accent:#FFFFFF;
  --trap:#7A4FB0; --trap-soft:#F0E9F8;
  --ok:#2A6E4F; --ok-soft:#E2F0E8; --bad:#B23A34; --bad-soft:#F8E7E5;
  --mono:ui-monospace,"SF Mono",SFMono-Regular,Menlo,Consolas,"Liberation Mono",monospace;
  --serif:"Iowan Old Style","Palatino Linotype",Palatino,"Book Antiqua","Hoefler Text",Georgia,serif;
  --ui:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
  --shadow:0 1px 2px rgba(16,32,36,.06),0 8px 24px -16px rgba(16,32,36,.24);
}
@media (prefers-color-scheme:dark){
  :root{
    --ground:#0D1214; --panel:#141B1E; --panel-2:#1B2427; --ink:#E8EDEC; --ink-2:#A6B5B8;
    --ink-3:#718387; --rule:#243033; --rule-2:#31403F;
    --accent:#4FBACB; --accent-soft:#11292E; --accent-ink:#8AD6E2; --on-accent:#08181C;
    --trap:#B492E6; --trap-soft:#211A31;
    --ok:#63C295; --ok-soft:#12261D; --bad:#E88178; --bad-soft:#2A1614;
    --shadow:0 1px 2px rgba(0,0,0,.4),0 10px 28px -18px rgba(0,0,0,.8);
  }
}
:root[data-theme="dark"]{
  --ground:#0D1214; --panel:#141B1E; --panel-2:#1B2427; --ink:#E8EDEC; --ink-2:#A6B5B8;
  --ink-3:#718387; --rule:#243033; --rule-2:#31403F;
  --accent:#4FBACB; --accent-soft:#11292E; --accent-ink:#8AD6E2; --on-accent:#08181C;
  --trap:#B492E6; --trap-soft:#211A31;
  --ok:#63C295; --ok-soft:#12261D; --bad:#E88178; --bad-soft:#2A1614;
  --shadow:0 1px 2px rgba(0,0,0,.4),0 10px 28px -18px rgba(0,0,0,.8);
}
:root[data-theme="light"]{
  --ground:#F6F7F6; --panel:#FFFFFF; --panel-2:#EEF1F1; --ink:#161B1D; --ink-2:#48555A;
  --ink-3:#78888D; --rule:#DCE2E2; --rule-2:#C6D0D0;
  --accent:#0B6E7E; --accent-soft:#E2F0F1; --accent-ink:#075463; --on-accent:#FFFFFF;
  --trap:#7A4FB0; --trap-soft:#F0E9F8;
  --ok:#2A6E4F; --ok-soft:#E2F0E8; --bad:#B23A34; --bad-soft:#F8E7E5;
  --shadow:0 1px 2px rgba(16,32,36,.06),0 8px 24px -16px rgba(16,32,36,.24);
}
*{box-sizing:border-box}
body{margin:0;background:var(--ground);color:var(--ink);font-family:var(--ui);
  font-size:15px;line-height:1.55;-webkit-font-smoothing:antialiased}
.wrap{max-width:1400px;margin:0 auto;padding:0 24px 96px}

/* masthead */
.mast{padding:56px 0 28px;border-bottom:1px solid var(--rule)}
.eyebrow{font-size:11px;letter-spacing:.16em;text-transform:uppercase;color:var(--ink-3);
  font-weight:600;margin:0 0 14px}
.mast h1{font-family:var(--serif);font-weight:600;font-size:clamp(30px,4.2vw,50px);
  line-height:1.08;margin:0 0 14px;text-wrap:balance;letter-spacing:-.012em}
.mast h1 em{font-style:italic;color:var(--accent)}
.dek{max-width:66ch;color:var(--ink-2);font-size:16.5px;margin:0 0 26px;font-family:var(--serif)}
.meta{display:flex;flex-wrap:wrap;gap:8px 10px;font-family:var(--mono);font-size:11.5px;color:var(--ink-3)}
.meta span{background:var(--panel-2);border:1px solid var(--rule);padding:3px 9px;border-radius:2px}

/* stat strip */
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:1px;
  background:var(--rule);border:1px solid var(--rule);margin:28px 0 0}
.stat{background:var(--panel);padding:15px 16px}
.stat b{display:block;font-family:var(--mono);font-size:23px;font-variant-numeric:tabular-nums;
  letter-spacing:-.02em;line-height:1.2}
.stat small{display:block;font-size:11px;letter-spacing:.09em;text-transform:uppercase;
  color:var(--ink-3);margin-top:5px;font-weight:600}

/* tier ladder */
.ladder{margin:36px 0 8px}
.ladder h2,.legend h2{font-family:var(--serif);font-size:19px;margin:0 0 12px;font-weight:600}
.tiers{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:10px}
.tier-card{background:var(--panel);border:1px solid var(--rule);padding:13px 14px;border-radius:3px}
.tier-card .tt{display:flex;align-items:baseline;gap:8px;margin-bottom:7px}
.tier-card .tt code{font-family:var(--mono);font-weight:700;font-size:13px;color:var(--accent)}
.tier-card .tt .pct{margin-left:auto;font-family:var(--mono);font-size:13px;
  font-variant-numeric:tabular-nums;color:var(--ink-2)}
.tier-card p{margin:0;font-size:12.5px;color:var(--ink-3);line-height:1.45}
.bars{display:flex;gap:4px;margin-top:9px}
.bar{flex:1}
.bar i{display:block;height:5px;background:var(--rule-2);border-radius:1px;overflow:hidden;position:relative}
.bar i b{position:absolute;inset:0 auto 0 0;background:var(--accent);display:block}
.bar small{display:block;font-size:9.5px;color:var(--ink-3);margin-top:3px;font-family:var(--mono)}

.fig{--c-correct:#2a78d6;--c-wrong:#eb6834;--c-judge:#2a78d6;--c-ar:#e34948;
  --s0:#86b6ef;--s1:#5598e7;--s2:#2a78d6;--s3:#104281;
  --grid:var(--rule);--axis:var(--ink-3);--figsurface:var(--panel);
  margin:0 0 4px}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]) .fig{
  --c-correct:#3987e5;--c-wrong:#d95926;--c-judge:#3987e5;--c-ar:#e66767;
  --s0:#b7d3f6;--s1:#6da7ec;--s2:#2a78d6;--s3:#184f95}}
:root[data-theme="dark"] .fig{
  --c-correct:#3987e5;--c-wrong:#d95926;--c-judge:#3987e5;--c-ar:#e66767;
  --s0:#b7d3f6;--s1:#6da7ec;--s2:#2a78d6;--s3:#184f95}
.fig svg{display:block;width:100%;height:auto;overflow:visible}
.fig text{font-family:var(--ui);fill:var(--ink-2)}
.fig .ax{font-size:9.5px;fill:var(--ink-3)}
.fig .vl{font-size:9.5px;fill:var(--ink-2);font-variant-numeric:tabular-nums}
.fig .gl{stroke:var(--grid);stroke-width:1}
.fig .zl{stroke:var(--ink-3);stroke-width:1;stroke-dasharray:3 3}
.fig .ref{stroke:var(--ink-3);stroke-width:1;stroke-dasharray:4 4;opacity:.6}
.figlegend{display:flex;flex-wrap:wrap;gap:6px 15px;font-size:11px;color:var(--ink-2);
  margin:0 0 9px;align-items:center}
.figlegend span{display:inline-flex;align-items:center;gap:5px}
.figlegend i{width:11px;height:11px;border-radius:2px;display:inline-block;flex:none}
.figcap{font-size:11.5px;color:var(--ink-3);line-height:1.5;margin:7px 0 0}
.pill.p-dense{background:#EFE8F7;color:#5B3A8C;border-color:#DCCDF0}
.densekey{max-width:78ch;margin:10px auto 0;padding:0 22px;font-size:12.5px;line-height:1.65;color:var(--ink-2)}
.scopenote{margin:-4px 0 16px;padding:10px 13px;background:var(--panel-2);border-left:3px solid var(--rule-2);
  border-radius:3px;font-size:12.5px;line-height:1.62;color:var(--ink-2)}
.sortnote{max-width:78ch;margin:10px auto 0;padding:0 22px;font-size:12.5px;line-height:1.65;
  color:var(--ink-2)}
.retracted{color:#b4432f}
.vcard.rej{border-left:3px solid #b4432f}
.vcard .no{color:#b4432f}
/* faithfulness analysis */
.viz{--viz:#008B7A;--viz-soft:#DDEFEC;--viz-grid:var(--rule);margin:34px 0 0}
@media (prefers-color-scheme:dark){.viz{--viz:#0E9BB5;--viz-soft:#0E2A2F}}
:root[data-theme="dark"] .viz{--viz:#0E9BB5;--viz-soft:#0E2A2F}
:root[data-theme="light"] .viz{--viz:#008B7A;--viz-soft:#DDEFEC}
.viz>h2{font-family:var(--serif);font-size:19px;margin:0 0 6px;font-weight:600}
.viz>p.lead{max-width:74ch;color:var(--ink-2);margin:0 0 18px;font-size:14px}
.vgrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:1px;
  background:var(--rule);border:1px solid var(--rule);border-radius:4px;overflow:hidden}
.vcard{background:var(--panel);padding:17px 18px}
.vcard h3{font-size:13px;margin:0 0 3px;font-weight:600;color:var(--ink)}
.vcard p.sub{font-size:12px;color:var(--ink-3);margin:0 0 14px;line-height:1.45}
.vcard p.take{font-size:12.5px;color:var(--ink-2);margin:13px 0 0;line-height:1.5;
  border-top:1px solid var(--rule);padding-top:10px}
.vcard p.take b{color:var(--ink)}
/* horizontal bars */
.hbar{display:grid;grid-template-columns:auto 1fr auto;gap:7px 10px;align-items:center;
  font-size:12px;font-variant-numeric:tabular-nums}
.hbar .nm{color:var(--ink-2);white-space:nowrap;font-size:11.5px}
.hbar .nm i{font-style:normal;color:var(--ink-3);font-family:var(--mono);font-size:10px}
.hbar .tr{position:relative;height:14px;background:var(--panel-2);border-radius:2px}
.hbar .tr b{position:absolute;left:0;top:0;bottom:0;background:var(--viz);
  border-radius:0 4px 4px 0;display:block}
.hbar .tr u{position:absolute;top:50%;height:1px;background:var(--ink-3);transform:translateY(-50%);
  opacity:.65;text-decoration:none}
.hbar .vl{font-family:var(--mono);font-size:11.5px;color:var(--ink);min-width:42px;text-align:right}
/* depth line */
.spark{width:100%;height:132px;display:block;overflow:visible}
.spark .ln{fill:none;stroke:var(--viz);stroke-width:2}
.spark .ar{fill:var(--viz-soft)}
.spark .dot{fill:var(--viz)}
.spark .gl{stroke:var(--viz-grid);stroke-width:1}
.spark text{font-family:var(--mono);font-size:9.5px;fill:var(--ink-3)}
.spark text.val{fill:var(--ink);font-size:10px}
/* dumbbell */
.dumb{display:grid;grid-template-columns:auto 1fr auto;gap:9px 10px;align-items:center;font-size:12px}
.dumb .tier{font-family:var(--mono);font-size:11.5px;color:var(--ink-2);font-weight:600}
.dumb .tr{position:relative;height:16px}
.dumb .tr .rail{position:absolute;left:0;right:0;top:50%;height:1px;background:var(--rule-2);transform:translateY(-50%)}
.dumb .tr .conn{position:absolute;top:50%;height:2px;background:var(--viz);transform:translateY(-50%);opacity:.5}
.dumb .tr span{position:absolute;top:50%;width:10px;height:10px;border-radius:50%;
  transform:translate(-50%,-50%);box-sizing:border-box}
.dumb .tr .w{background:var(--panel);border:2px solid var(--viz)}
.dumb .tr .c{background:var(--viz);border:2px solid var(--viz)}
.dumb .dv{font-family:var(--mono);font-size:11px;color:var(--ink);min-width:44px;text-align:right}
.vlegend{display:flex;gap:14px;align-items:center;font-size:11px;color:var(--ink-3);margin:0 0 12px}
.vlegend b{display:inline-block;width:10px;height:10px;border-radius:50%;vertical-align:-1px;margin-right:5px}
.vlegend b.c{background:var(--viz)}
.vlegend b.w{background:var(--panel);border:2px solid var(--viz);box-sizing:border-box}
/* histogram */
.hist{display:flex;align-items:flex-end;gap:2px;height:96px}
.hist i{flex:1;background:var(--viz);border-radius:2px 2px 0 0;min-height:1px;opacity:.85}
.hist i.dim{background:var(--rule-2)}
.haxis{display:flex;justify-content:space-between;font-family:var(--mono);font-size:9.5px;
  color:var(--ink-3);margin-top:5px}
/* controls */
.controls{position:sticky;top:0;z-index:20;background:var(--ground);
  border-bottom:1px solid var(--rule);padding:12px 0;margin:34px 0 0;
  display:flex;flex-wrap:wrap;gap:10px;align-items:center}
.grp{display:flex;gap:4px;align-items:center;flex-wrap:wrap}
.grp>label{font-size:10.5px;letter-spacing:.1em;text-transform:uppercase;color:var(--ink-3);
  font-weight:700;margin-right:3px}
.chip{font:inherit;font-size:12px;font-family:var(--mono);background:var(--panel);
  color:var(--ink-2);border:1px solid var(--rule-2);padding:4px 10px;border-radius:2px;
  cursor:pointer;transition:.12s}
.chip:hover{border-color:var(--accent);color:var(--ink)}
.chip[aria-pressed="true"]{background:var(--accent);border-color:var(--accent);color:var(--on-accent)}
.chip:focus-visible,.searchbox:focus-visible,.tbtn:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.searchbox{font:inherit;font-size:13px;background:var(--panel);border:1px solid var(--rule-2);
  color:var(--ink);padding:5px 11px;border-radius:2px;min-width:220px;flex:1;max-width:320px}
.count{margin-left:auto;font-family:var(--mono);font-size:12px;color:var(--ink-3);
  font-variant-numeric:tabular-nums;white-space:nowrap}
.tbtn{font:inherit;font-size:12px;background:var(--panel);border:1px solid var(--rule-2);
  color:var(--ink-2);padding:4px 10px;border-radius:2px;cursor:pointer}

/* case */
.case{background:var(--panel);border:1px solid var(--rule);border-radius:4px;
  margin-top:22px;overflow:hidden;box-shadow:var(--shadow)}
.chead{display:flex;flex-wrap:wrap;gap:9px;align-items:center;padding:13px 18px;
  border-bottom:1px solid var(--rule);background:var(--panel-2)}
.chead .id{font-family:var(--mono);font-size:12.5px;color:var(--ink-2);font-weight:600}
.pill{font-family:var(--mono);font-size:10.5px;letter-spacing:.05em;padding:2.5px 8px;
  border-radius:2px;border:1px solid transparent;text-transform:uppercase;font-weight:700}
.p-ok{background:var(--ok-soft);color:var(--ok);border-color:var(--ok)}
.p-bad{background:var(--bad-soft);color:var(--bad);border-color:var(--bad)}
.p-tier{background:var(--accent-soft);color:var(--accent-ink);border-color:var(--accent)}
.p-trap{background:var(--trap-soft);color:var(--trap);border-color:var(--trap)}
.p-mute{background:transparent;color:var(--ink-3);border-color:var(--rule-2)}
.cbody{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:0}
@media (max-width:1000px){.cbody{grid-template-columns:minmax(0,1fr)}}
.pane{padding:18px;min-width:0}
.pane+.pane{border-left:1px solid var(--rule)}
@media (max-width:1000px){.pane+.pane{border-left:0;border-top:1px solid var(--rule)}}
.plabel{font-size:10.5px;letter-spacing:.13em;text-transform:uppercase;color:var(--ink-3);
  font-weight:700;margin:0 0 10px;display:flex;align-items:center;gap:8px}
.plabel::after{content:"";flex:1;height:1px;background:var(--rule)}
pre.code{font-family:var(--mono);font-size:12px;line-height:1.55;background:var(--panel-2);
  border:1px solid var(--rule);border-radius:3px;padding:12px;margin:0 0 14px;
  overflow-x:auto;max-height:260px;overflow-y:auto;white-space:pre;color:var(--ink)}
.qa{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:0 0 14px;
  font-family:var(--mono);font-size:12.5px}
.qa .k{font-size:10px;letter-spacing:.1em;text-transform:uppercase;color:var(--ink-3);font-weight:700}
.qa .v{padding:2px 8px;border-radius:2px;background:var(--panel-2);border:1px solid var(--rule)}
.qa .v.ans-ok{color:var(--ok);border-color:var(--ok);background:var(--ok-soft)}
.qa .v.ans-bad{color:var(--bad);border-color:var(--bad);background:var(--bad-soft)}
details.prompt{margin:0 0 14px;border:1px solid var(--rule);border-radius:3px;background:var(--panel-2)}
details.prompt>summary{cursor:pointer;padding:8px 12px;font-size:11px;letter-spacing:.1em;
  text-transform:uppercase;font-weight:700;color:var(--ink-3);list-style:none}
details.prompt>summary::-webkit-details-marker{display:none}
details.prompt>summary::before{content:"▸ ";color:var(--accent)}
details.prompt[open]>summary::before{content:"▾ "}
details.prompt>summary:hover{color:var(--accent)}
details.prompt pre{margin:0;padding:12px;border-top:1px solid var(--rule);font-family:var(--mono);
  font-size:11.5px;line-height:1.6;white-space:pre-wrap;word-break:break-word;color:var(--ink-2);
  max-height:340px;overflow-y:auto}
details.prompt pre .pre-fixed{color:var(--ink-3)}
details.prompt pre .pre-code{color:var(--ink)}
details.prompt pre .pre-q{color:var(--accent-ink);font-weight:600}

/* methodology */
.method{margin:34px 0 0;background:var(--panel);border:1px solid var(--rule);border-radius:4px;padding:20px 22px}
.method h2{font-family:var(--serif);font-size:19px;margin:0 0 6px;font-weight:600}
.method>p{margin:0 0 18px;color:var(--ink-2);max-width:70ch;font-size:14px}
.mgrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:18px}
.mcol h3{font-size:11px;letter-spacing:.12em;text-transform:uppercase;color:var(--ink-3);
  font-weight:700;margin:0 0 8px}
.mcol pre{margin:0;background:var(--ground);border:1px solid var(--rule);border-radius:3px;
  padding:11px;font-family:var(--mono);font-size:11px;line-height:1.55;white-space:pre-wrap;
  color:var(--ink-2);max-height:230px;overflow-y:auto}
.mcol pre b{color:var(--accent);font-weight:700}
.mnote{margin:8px 0 0;font-size:12.5px;color:var(--ink-3);line-height:1.5}
.breakdown{display:flex;flex-wrap:wrap;gap:8px;margin:0 0 16px}
.bd{background:var(--ground);border:1px solid var(--rule);border-radius:3px;padding:8px 12px;
  font-family:var(--mono);font-size:12px}
.bd b{color:var(--accent);font-variant-numeric:tabular-nums}
.bd small{color:var(--ink-3);display:block;font-size:10.5px;margin-top:2px;font-family:var(--ui)}
.cot{font-family:var(--serif);font-size:13.5px;line-height:1.62;color:var(--ink-2);
  background:var(--ground);border:1px solid var(--rule);border-radius:3px;padding:13px;
  max-height:420px;overflow-y:auto;white-space:pre-wrap;word-break:break-word}
.cot mark{background:rgba(11,110,126,calc(var(--mf,.6) * .20));color:var(--ink);
  border-bottom:2px solid rgba(11,110,126,calc(var(--mf,.6) * .92));
  padding:0 1px;border-radius:1px;cursor:pointer;scroll-margin:80px;transition:background .12s}
.cot mark:hover{background:rgba(11,110,126,.30)}
.cot mark.hot{background:var(--accent);color:var(--on-accent);border-bottom-color:var(--accent)}
.cot mark:focus-visible,.read:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.shadekey{display:flex;align-items:center;gap:7px;margin:7px 0 0;font-size:11px;color:var(--ink-3)}
.shadekey .ramp{width:78px;height:9px;border-radius:2px;
  background:linear-gradient(90deg,rgba(11,110,126,.04),rgba(11,110,126,.20));
  border:1px solid var(--rule)}

/* reads */
.reads{display:flex;flex-direction:column;gap:11px;max-height:620px;overflow-y:auto;padding-right:4px}
.read{border:1px solid var(--rule);border-left:3px solid var(--accent);border-radius:3px;
  background:var(--ground);padding:11px 13px;cursor:pointer;transition:.13s}
.read:hover{border-color:var(--rule-2);border-left-color:var(--accent);background:var(--panel-2)}
.read.sel{border-left-width:5px;background:var(--accent-soft)}
.read.trap{border-left-color:var(--trap)}
.read.trap.sel{background:var(--trap-soft)}
.rhead{display:flex;flex-wrap:wrap;gap:7px;align-items:center;margin-bottom:7px;
  font-family:var(--mono);font-size:10.5px;color:var(--ink-3)}
.rhead .tok{background:var(--panel);border:1px solid var(--rule-2);padding:1.5px 7px;
  border-radius:2px;color:var(--ink);font-weight:600;font-size:11px}
.rhead .cls{letter-spacing:.02em;font-weight:600;font-style:italic;font-family:var(--serif);font-size:11.5px}
.rhead .cls.adversarial{color:var(--trap)}
.rhead .cls.cot,.rhead .cls.answer{color:var(--accent)}
.rhead .num{margin-left:auto;font-variant-numeric:tabular-nums;white-space:nowrap;
  border-bottom:1px dotted var(--rule-2);cursor:help}
.ctx{font-family:var(--mono);font-size:10.5px;line-height:1.5;color:var(--ink-3);
  background:var(--panel);border:1px solid var(--rule);border-radius:2px;padding:5px 8px;
  margin:0 0 8px;white-space:nowrap;overflow-x:auto}
.ctx b{color:var(--on-accent);background:var(--accent);padding:1px 3px;border-radius:2px;font-weight:700}
.read.trap .ctx b{background:var(--trap);color:#fff}
.inword{font-family:var(--mono);font-size:10px;color:var(--ink-3)}
.inword code{color:var(--ink-2);font-weight:600}
.drm .dlab{font-size:9.5px;white-space:nowrap}
.drm .dnum{min-width:50px;text-align:right;font-variant-numeric:tabular-nums}
.rtext{font-family:var(--serif);font-size:13.5px;line-height:1.58;color:var(--ink);margin:0}
.drm{display:flex;align-items:center;gap:7px;margin-top:8px;font-family:var(--mono);font-size:10px;
  color:var(--ink-3)}
.drm .track{position:relative;flex:1;height:4px;background:var(--rule);border-radius:2px}
.drm .track::before{content:"";position:absolute;left:50%;top:-3px;bottom:-3px;width:1px;background:var(--rule-2)}
.drm .track i{position:absolute;top:0;bottom:0;border-radius:2px}
.drm .track i.pos{background:var(--trap)}
.drm .track i.neg{background:var(--accent)}

.more{display:block;width:100%;margin:28px 0 0;padding:13px;font:inherit;font-size:13px;
  font-family:var(--mono);background:var(--panel);border:1px dashed var(--rule-2);
  color:var(--ink-2);border-radius:3px;cursor:pointer}
.more:hover{border-color:var(--accent);color:var(--accent)}
.empty{padding:48px 0;text-align:center;color:var(--ink-3);font-family:var(--serif);font-size:16px}

.legend{margin:18px 0 0;padding:18px;background:var(--panel);border:1px solid var(--rule);border-radius:4px}
.legend dl{display:grid;grid-template-columns:auto 1fr;gap:9px 14px;margin:0;font-size:13px}
@media (min-width:1080px){
  .legend dl{grid-template-columns:auto minmax(0,1fr) auto minmax(0,1fr);column-gap:26px}
}
.legend dt{font-family:var(--mono);font-size:11.5px;color:var(--accent);font-weight:700;white-space:nowrap}
.legend dd{margin:0;color:var(--ink-2)}
.legend .lead{max-width:72ch;color:var(--ink-2);margin:0 0 16px;font-size:14px}
.legend dd code{font-family:var(--mono);font-size:11.5px;background:var(--ground);
  border:1px solid var(--rule);padding:0 4px;border-radius:2px}
.sw-a{color:var(--accent);font-weight:600}
.sw-t{color:var(--trap);font-weight:600}
.foot{margin-top:44px;padding-top:18px;border-top:1px solid var(--rule);
  font-size:12px;color:var(--ink-3);font-family:var(--mono);line-height:1.7}
@media (prefers-reduced-motion:reduce){*{transition:none!important;scroll-behavior:auto!important}}
"""

JS = r"""
const DATA = window.__NLA__;
const CASES = DATA.cases, ST = DATA.stats;
const F = {tier:new Set(), verdict:"", kind:"", q:"", sort:"default", dense:false};
const feed = document.getElementById("feed");
const countEl = document.getElementById("count");
let shown = 0, filtered = [];
const PAGE = 8;

const esc = s => (s??"").replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));

function apply(){
  const q = F.q.toLowerCase();
  filtered = CASES.filter(c => {
    if (F.tier.size && !F.tier.has(c.tier)) return false;
    if (F.kind && c.kind !== F.kind) return false;
    if (F.dense && !c.dense) return false;
    if (F.verdict === "ok" && c.correct !== true) return false;
    if (F.verdict === "bad" && c.correct !== false) return false;
    if (q) {
      const hay = (c.snippet_id+" "+(c.call||"")+" "+(c.model_answer||"")+" "+
        c.reads.map(r=>r.read).join(" ")).toLowerCase();
      if (!hay.includes(q)) return false;
    }
    return true;
  });
  // Faithfulness sort. Cases with no scored readings have no key and must never render as
  // "least faithful" -- they sort last in BOTH directions.
  if (F.sort !== "default"){
    const key = c => {
      const v = c.reads.map(r => r.rt_cos).filter(x => typeof x === "number");
      return v.length ? v.reduce((a,b)=>a+b,0)/v.length : null;
    };
    const dir = F.sort === "faith_hi" ? -1 : 1;
    filtered = filtered.map(c => ({c, k: key(c)}))
      .sort((x,y) => (x.k === null) - (y.k === null) || (x.k === null ? 0 : dir*(x.k - y.k)))
      .map(o => o.c);
  }
  feed.innerHTML = ""; shown = 0;
  countEl.textContent = filtered.length + " / " + CASES.length + " cases";
  if (!filtered.length){ feed.innerHTML = '<p class="empty">No cases match these filters.</p>'; return; }
  more();
}
function more(){
  const slice = filtered.slice(shown, shown + PAGE);
  const frag = document.createDocumentFragment();
  slice.forEach(c => frag.appendChild(card(c)));
  const old = feed.querySelector(".more"); if (old) old.remove();
  feed.appendChild(frag);
  shown += slice.length;
  if (shown < filtered.length){
    const b = document.createElement("button");
    b.className = "more"; b.type = "button";
    b.textContent = "Show " + Math.min(PAGE, filtered.length-shown) + " more  ·  " + (filtered.length-shown) + " remaining";
    b.onclick = more; feed.appendChild(b);
  }
}
const MARK_CAP = 120;
// Sweep-line over anchor boundaries. The previous version walked reads in order and clamped each
// span to where the last one ended (s = max(cur, rs)), so overlapping anchors were silently
// swallowed and `data-r` pointed only at the FIRST read of each overlapping group -- clicking the
// others found no mark. Here every distinct boundary starts a new segment and a segment carries
// EVERY read covering it, as data-r="3,4,5". Shading encodes mean round-trip faithfulness.
function cotHTML(c){
  const reply = c.model_reply || "";
  if (!reply) return '<span style="color:var(--ink-3)">(no reasoning text — prompt-only case)</span>';
  let spans = c.reads.map((r,i)=>({i, rs:(r.anchor||{}).rs, re:(r.anchor||{}).re,
                                   ok:(r.anchor||{}).in_reply, f:r.rt_cos}))
    .filter(s => s.ok && Number.isFinite(s.rs) && Number.isFinite(s.re)
                 && s.re > s.rs && s.rs < reply.length)
    .map(s => ({...s, re: Math.min(s.re, reply.length)}))
    .sort((a,b) => a.rs - b.rs);
  if (spans.length > MARK_CAP){                    // thin evenly; never drop a whole region
    const step = spans.length / MARK_CAP;
    spans = spans.filter((_,k) => Math.floor(k % step) === 0).slice(0, MARK_CAP);
  }
  const pts = [...new Set([0, reply.length, ...spans.flatMap(s=>[s.rs,s.re])])].sort((a,b)=>a-b);
  let out = "";
  for (let k = 0; k < pts.length - 1; k++){
    const a = pts[k], b = pts[k+1];
    if (b <= a) continue;
    const txt = esc(reply.slice(a, b));
    const cov = spans.filter(s => s.rs <= a && s.re >= b);
    if (!cov.length){ out += txt; continue; }
    const idx = cov.map(s => s.i);
    const fs = cov.map(s => s.f).filter(x => typeof x === "number");
    const mf = fs.length ? fs.reduce((x,y)=>x+y,0)/fs.length : null;
    // 0.70..0.95 recovery -> 0.18..1.0 ink. Below 0.70 floors rather than vanishing.
    const op = mf === null ? 0.55 : Math.max(0.18, Math.min(1, (mf - 0.70) / 0.25));
    const tip = (idx.length > 1 ? idx.length + " readings overlap here — click to cycle"
                                : "reading #" + (idx[0] + 1))
              + (mf !== null ? " · recovers " + Math.round(mf*100) + "%" : "");
    out += '<mark data-r="' + idx.join(",") + '" tabindex="0" role="button"'
         + ' style="--mf:' + op.toFixed(3) + '" title="' + tip + '">' + txt + '</mark>';
  }
  return out;
}
function card(c){
  const el = document.createElement("article");
  el.className = "case";
  const uid = c.task_key.replace(/[^a-z0-9]/gi,"");
  const verdict = c.correct === true ? '<span class="pill p-ok">correct</span>'
    : c.correct === false ? '<span class="pill p-bad">wrong</span>'
    : '<span class="pill p-mute">not graded</span>';
  const trap = (c.tier==="L1b"||c.tier==="L3") ? '<span class="pill p-trap">trap tier</span>' : "";
  const trunc = c.truncated ? '<span class="pill p-mute">truncated</span>' : "";
  const kindLabel = c.kind.replace("_"," ");
  el.innerHTML = `
   <header class="chead">
     <span class="id">${esc(c.snippet_id)}</span>
     <span class="pill p-tier">${esc(c.tier==="-"?"synthetic":c.tier)}</span>
     ${trap}${verdict}${trunc}
     <span class="pill p-mute">${esc(kindLabel)}</span>
     <span class="pill p-mute">${esc(c.dataset)}</span>
     <span class="pill p-mute">${esc(c.language||"")}</span>
   </header>
   <div class="cbody">
     <section class="pane">
       <details class="prompt"><summary>Full prompt sent to the model</summary>
         <pre><span class="pre-fixed">${esc(DATA.preamble)}</span><span class="pre-code">${esc(c.code)}</span><span class="pre-q">\n\n${esc(c.question)}</span></pre>
       </details>
       <p class="plabel">Code under test</p>
       <pre class="code">${esc(c.code)}</pre>
       <p class="plabel">Question &amp; answer</p>
       <div class="qa">
         <span class="k">asks</span><span class="v">${esc(c.call || ("slice for "+(c.target||"?")))}</span>
       </div>
       <div class="qa">
         <span class="k">model</span>
         <span class="v ${c.correct===true?'ans-ok':c.correct===false?'ans-bad':''}">${esc(c.model_answer ?? "— no answer —")}</span>
         <span class="k">truth</span><span class="v">${esc(c.truth ?? "n/a")}</span>
         <span class="k">reasoning length</span><span class="v${c.truncated?' ans-bad':''}">${c.reply_tokens.toLocaleString()} tokens${c.truncated?' · hit the cap':''}</span>
       </div>
       <p class="plabel">Chain of thought <span style="text-transform:none;letter-spacing:0;font-weight:400;color:var(--ink-3)">— click any highlight to jump to its reading</span></p>
       <div class="cot" id="cot${uid}">${cotHTML(c)}</div>
       <p class="shadekey"><span class="ramp"></span>
         <span>darker = the reading recovered more of the vector at that token
         (<b>completeness, not correctness</b>). Overlapping highlights cycle on repeated clicks.</span></p>
     </section>
     <section class="pane">
       <p class="plabel">NLA readings <span style="text-transform:none;letter-spacing:0;font-weight:400;color:var(--ink-3)">— layer 20, one token each · click to light up its token on the left</span></p>
       <div class="reads">${c.reads.map((r,i)=>readHTML(c,r,i,uid)).join("")}</div>
     </section>
   </div>`;
  wireCase(el, uid);
  return el;
}
// Scroll WITHIN a pane. scrollIntoView() would also scroll the page and the other pane, which is
// exactly what makes a two-pane linked view feel broken.
function scrollInPane(pane, node){
  if (!pane || !node) return;
  const pr = pane.getBoundingClientRect(), nr = node.getBoundingClientRect();
  pane.scrollTo({top: pane.scrollTop + (nr.top - pr.top) - (pane.clientHeight - nr.height)/2,
                 behavior: "smooth"});
}
// One selection model for both directions. `origin` is the pane the user acted in; that pane is
// never scrolled, so the thing they just clicked does not jump away from the cursor.
function select(el, uid, i, origin){
  const cot = el.querySelector("#cot"+uid);
  const readsPane = el.querySelector(".reads");
  el.querySelectorAll(".read").forEach(n => n.classList.toggle("sel", +n.dataset.r === i));
  let first = null;
  cot.querySelectorAll("mark").forEach(m => {
    const on = m.dataset.r.split(",").includes(String(i));
    m.classList.toggle("hot", on);
    if (on && !first) first = m;
  });
  if (origin !== "cot") scrollInPane(cot, first);
  if (origin !== "read"){
    const rn = el.querySelector('.read[data-r="'+i+'"]');
    scrollInPane(readsPane, rn);
    if (rn) rn.focus({preventScroll:true});
  }
}
function wireCase(el, uid){
  el.querySelectorAll(".read").forEach(node => {
    const go = () => select(el, uid, +node.dataset.r, "read");
    node.addEventListener("click", go);
    node.addEventListener("keydown", e => {          // role="button" had no key handler: a real bug
      if (e.key === "Enter" || e.key === " "){ e.preventDefault(); go(); }
    });
  });
  const cot = el.querySelector("#cot"+uid);
  // delegated, because marks are rebuilt with the card and may number in the hundreds
  cot.addEventListener("click", e => {
    const m = e.target.closest("mark");
    if (!m) return;
    const list = m.dataset.r.split(",").map(Number);
    // a mark shared by several readings cycles through them on repeated clicks
    const cur = list.findIndex(n =>
      el.querySelector('.read[data-r="'+n+'"]')?.classList.contains("sel"));
    select(el, uid, list[(cur + 1) % list.length], "cot");
  });
  cot.addEventListener("keydown", e => {
    if (e.key !== "Enter" && e.key !== " ") return;
    const m = e.target.closest("mark");
    if (!m) return;
    e.preventDefault(); m.click();
  });
}
function readHTML(c,r,i,uid){
  const where = r.where || "", a = r.anchor || {};
  const kind = where.startsWith("cot") ? "cot" : where.startsWith("ans") ? "answer" : (r.cls||"code");
  const isTrap = r.cls === "adversarial" || r.cls === "fn_adversarial";
  const CLSNAME = {adversarial:"a misleading name", fn_adversarial:"a misleading function name",
    orig:"an original name", fn_orig:"the original function name", l1_neutral:"a neutral rename",
    dispatcher:"the dispatcher variable", target:"the target variable", self_derived:"an identifier"};
  // where the read sits, in plain English
  let loc, qual = "";
  if (where.startsWith("cot@")){
    loc = "in the reasoning";
    qual = "token " + where.slice(4) + " of " + c.reply_tokens;
  } else if (where.startsWith("ans@")){
    loc = "on the answer line";
    qual = "the model's final answer";
  } else {
    loc = "in the code";
    qual = CLSNAME[r.cls] || "a code token";
  }
  // the exact token, with its surroundings
  const ctx = a.tok != null ? `<div class="ctx"><span>${esc(a.before)}</span><b>${esc(a.tok)}</b><span>${esc(a.after)}</span></div>` : "";
  const inWord = (a.word && a.word !== a.tok.trim())
    ? `<span class="inword">piece of <code>${esc(a.word)}</code></span>` : "";
  let drm = "";
  if (r.drm_ar != null){
    const v = r.drm_ar, mag = Math.min(Math.abs(v)/0.15, 1)*50;
    drm = `<div class="drm" title="Which meaning this vector sits closer to: the function's real behaviour, or the misleading name's suggestion.">
      <span class="dlab">reads as the real function</span><span class="track">
      <i class="${v>=0?'pos':'neg'}" style="${v>=0?`left:50%;width:${mag}%`:`right:50%;width:${mag}%`}"></i>
      </span><span class="dlab">reads as the decoy name</span>
      <span class="dnum">${v>=0?"+":""}${v.toFixed(3)}</span></div>`;
  }
  return `<div class="read ${isTrap?'trap':''}" data-r="${i}" tabindex="0" role="button">
    <div class="rhead">
      <span class="tok">${esc(loc)}</span>
      <span class="cls ${kind}">${esc(qual)}</span>
      ${inWord}
      <span class="num" title="Round-trip faithfulness: how much of the original vector the reconstructor recovers from this text alone. 1.00 would be perfect.">recovers ${(r.rt_cos*100).toFixed(0)}%</span>
      ${r.src && r.src!=="banked" ? '<span class="pill p-dense" title="From the 2026-08-06 dense run: a second, denser pass over this same transcript (median 24 readings per case instead of 6). Same model, same reply, same decoding.">dense'+(r.role?" · "+r.role:"")+'</span>' : ''}
    </div>
    ${ctx}
    <p class="rtext">${esc(r.read)}</p>${drm}</div>`;
}
// controls
document.querySelectorAll("[data-tier]").forEach(b=>b.addEventListener("click",()=>{
  const t=b.dataset.tier;
  if(F.tier.has(t)){F.tier.delete(t);b.setAttribute("aria-pressed","false");}
  else{F.tier.add(t);b.setAttribute("aria-pressed","true");}
  apply();
}));
document.querySelectorAll("[data-verdict]").forEach(b=>b.addEventListener("click",()=>{
  const v=b.dataset.verdict; F.verdict = F.verdict===v?"":v;
  document.querySelectorAll("[data-verdict]").forEach(n=>n.setAttribute("aria-pressed", n.dataset.verdict===F.verdict?"true":"false"));
  apply();
}));
document.querySelectorAll("[data-kind]").forEach(b=>b.addEventListener("click",()=>{
  const v=b.dataset.kind; F.kind = F.kind===v?"":v;
  document.querySelectorAll("[data-kind]").forEach(n=>n.setAttribute("aria-pressed", n.dataset.kind===F.kind?"true":"false"));
  apply();
}));
document.querySelectorAll("[data-dense]").forEach(b=>b.addEventListener("click",()=>{
  F.dense = !F.dense; b.setAttribute("aria-pressed", F.dense?"true":"false"); apply();
}));
document.querySelectorAll("[data-sort]").forEach(b=>b.addEventListener("click",()=>{
  F.sort = b.dataset.sort;                       // radio, not toggle: there is always one order
  document.querySelectorAll("[data-sort]").forEach(n=>{
    const on = n.dataset.sort===F.sort;
    n.setAttribute("aria-pressed", on?"true":"false"); n.classList.toggle("on", on);
  });
  apply(); window.scrollTo({top:document.querySelector(".controls").offsetTop-12,behavior:"smooth"});
}));
let tid; document.getElementById("q").addEventListener("input",e=>{
  clearTimeout(tid); tid=setTimeout(()=>{F.q=e.target.value.trim();apply();},180);
});
document.getElementById("reset").addEventListener("click",()=>{
  F.tier.clear();F.verdict="";F.kind="";F.q="";F.sort="default";F.dense=false;
  document.getElementById("q").value="";
  document.querySelectorAll("[aria-pressed]").forEach(n=>n.setAttribute("aria-pressed","false"));
  document.querySelectorAll("[data-sort]").forEach(n=>n.classList.remove("on"));
  const d=document.querySelector('[data-sort="default"]');
  if(d){d.setAttribute("aria-pressed","true");d.classList.add("on");}
  apply(); window.scrollTo({top:0,behavior:"smooth"});
});
apply();
"""


KINDS = {k: sum(1 for c in cases if c["kind"] == k) for k in
         ("output_prediction", "prompt_reads", "slice_prediction")}


def viz_section():
    f = FAITH
    lo, hi = 0.80, 0.95           # zoomed band for the class bars (all means sit inside)
    def barw(v): return max(2, (v - lo) / (hi - lo) * 100)

    bars = []
    for r in f["byClass"]:
        gen = "" if r["kind"] == "code" else " <i>generated</i>"
        bars.append(
            f'<span class="nm">{html.escape(r["label"])}{gen}</span>'
            f'<span class="tr" title="n={r["n"]} · 95% CI {r["lo"]:.3f}–{r["hi"]:.3f}">'
            f'<b style="width:{barw(r["mean"]):.1f}%"></b>'
            f'<u style="left:{barw(r["lo"]):.1f}%;width:{max(barw(r["hi"])-barw(r["lo"]),1):.1f}%"></u></span>'
            f'<span class="vl">{r["mean"]*100:.1f}%</span>')
    bars_html = "".join(bars)

    # depth line chart (SVG, 5 bins)
    d = f["byDepth"]
    W, H, PAD = 300, 132, 20
    xs = [PAD + i * (W - 2 * PAD) / (len(d) - 1) for i in range(len(d))]
    dlo, dhi = 0.80, 0.96
    ys = [H - 24 - (r["mean"] - dlo) / (dhi - dlo) * (H - 46) for r in d]
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
    area = f'M{xs[0]:.1f},{H-24} L' + " L".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys)) + f' L{xs[-1]:.1f},{H-24} Z'
    grid = "".join(f'<line class="gl" x1="{PAD}" x2="{W-PAD}" y1="{H-24-(g-dlo)/(dhi-dlo)*(H-46):.1f}" '
                   f'y2="{H-24-(g-dlo)/(dhi-dlo)*(H-46):.1f}"/>' for g in (0.85, 0.90, 0.95))
    dots = "".join(f'<circle class="dot" cx="{x:.1f}" cy="{y:.1f}" r="4"><title>{r["label"]} of the reasoning · '
                   f'{r["mean"]*100:.1f}% · n={r["n"]}</title></circle>' for x, y, r in zip(xs, ys, d))
    dlabs = "".join(f'<text x="{x:.1f}" y="{H-8}" text-anchor="middle">{r["label"].replace("–","-")}</text>'
                    for x, r in zip(xs, d))
    vlabs = "".join(f'<text class="val" x="{x:.1f}" y="{y-10:.1f}" text-anchor="middle">{r["mean"]*100:.0f}</text>'
                    for x, y, r in zip(xs, ys, d))
    spark = (f'<svg class="spark" viewBox="0 0 {W} {H}" role="img" '
             f'aria-label="Faithfulness across the reasoning: 94% at the start, dipping to 83% four-fifths through, recovering to 87% at the end.">'
             f'{grid}<path class="ar" d="{area}"/><polyline class="ln" points="{pts}"/>{dots}{dlabs}{vlabs}</svg>')

    # dumbbell: correct vs wrong per tier
    vlo, vhi = 0.84, 0.90
    def pos(v): return (v - vlo) / (vhi - vlo) * 100
    rows = []
    for r in f["byVerdict"]:
        c, w = pos(r["correct"]), pos(r["wrong"])
        rows.append(
            f'<span class="tier">{r["tier"]}</span>'
            f'<span class="tr" title="correct {r["correct"]*100:.1f}% (n={r["nc"]}) · '
            f'wrong {r["wrong"]*100:.1f}% (n={r["nw"]})"><span class="rail"></span>'
            f'<span class="conn" style="left:{min(c,w):.1f}%;width:{abs(c-w):.1f}%"></span>'
            f'<span class="w" style="left:{w:.1f}%"></span><span class="c" style="left:{c:.1f}%"></span></span>'
            f'<span class="dv">+{r["delta"]*100:.1f}</span>')
    dumb_html = "".join(rows)

    # histogram
    hb = f["hist"]["bins"]; mx = max(hb)
    hist = "".join(f'<i class="{"dim" if v==0 else ""}" style="height:{v/mx*100:.1f}%" '
                   f'title="{f["hist"]["lo"]+i*0.025:.3f}–{f["hist"]["lo"]+(i+1)*0.025:.3f}: {v} readings"></i>'
                   for i, v in enumerate(hb))

    o = f["overall"]
    return f'''  <section class="viz">
    <h2>How faithful are the readings themselves?</h2>
    <p class="lead">Every reading carries a round-trip score: the reconstructor is given the
      reading's text <em>alone</em> and tries to rebuild the original vector. That score measures how much of the
      model's internal state survived the trip through English — a property of the instrument, not of whether
      the reading is true. Bars show the mean; the hairline is a 95% bootstrap interval.</p>
    <p class="scopenote"><b>Scope:</b> this section covers the <b>{o["n"]:,} readings of the first pass</b>, not
      all {stats["reads"]:,} on this page. The {stats["dense_added"]:,} readings from the dense second pass are
      <b>deliberately not pooled in</b>, for two reasons. The breakdown by token type is impossible for them —
      they were taken at reasoning positions and carry no identifier class. And pooling the rest would move every
      average without anything having changed: the dense pass deliberately over-samples one late stretch of the
      trace (the low-faithfulness trough) and covers a deliberately balanced 91-right/91-wrong subset, so a
      combined mean would reflect <i>which readings were taken</i> rather than what they found. That is the same
      composition trap that produced the retracted result below, so it is not repeated here.</p>
    <div class="vgrid">
      <div class="vcard">
        <h3>Distribution across all readings</h3>
        <p class="sub">mean {o["mean"]*100:.1f}% · median {o["median"]*100:.1f}% · sd {o["sd"]*100:.1f} ·
          range {o["min"]*100:.0f}–{o["max"]*100:.0f}%</p>
        <div class="hist">{hist}</div>
        <div class="haxis"><span>50%</span><span>75%</span><span>100%</span></div>
        <p class="take">Tight and high: three quarters of readings recover more than
          <b>{o["median"]*100:.0f}%</b> of the vector. The instrument is consistent — the variation below is
          structure, not noise.</p>
      </div>
      <div class="vcard">
        <h3>By what kind of token was read</h3>
        <p class="sub">scale 80–95% · hairline = 95% CI · <i>generated</i> = token the model wrote, not code we gave it</p>
        <div class="hbar">{bars_html}</div>
        <p class="take"><b>Neutral renames read worst</b> ({[r for r in f["byClass"] if r["key"]=="l1_neutral"][0]["mean"]*100:.1f}%).
          A meaningless name like <code>var_bd90</code> leaves the state with little to say about it.
          <b>Misleading names read as well as real ones</b> ({[r for r in f["byClass"] if r["key"]=="adversarial"][0]["mean"]*100:.1f}%
          vs {[r for r in f["byClass"] if r["key"]=="orig"][0]["mean"]*100:.1f}%, difference not significant) —
          a decoy injects rich, confident, wrong meaning, which is just as describable as the truth.</p>
      </div>
      <div class="vcard">
        <h3>Across the model's reasoning</h3>
        <p class="sub">readings taken inside the chain of thought, by position</p>
        {spark}
        <p class="take">A dip, not a slide. The opening of a trace reads at <b>94%</b> — the model is restating
          the problem, and English captures that easily. It bottoms out at <b>83%</b> four-fifths through, deep in
          the arithmetic where the state holds partial results and bookkeeping. Then it <b>recovers to 87%</b> as
          the model converges on an answer worth stating.</p>
      </div>
      <div class="vcard">
        <h3>Correct runs vs wrong runs</h3>
        <p class="sub">same tier, same task — only the outcome differs</p>
        <div class="vlegend"><span><b class="c"></b>correct</span><span><b class="w"></b>wrong</span>
          <span style="margin-left:auto">scale 84–90%</span></div>
        <div class="dumb">{dumb_html}</div>
        <p class="take"><b class="retracted">Superseded — this gap is an artifact.</b> It looked like
          "when the model is right, its internal state is more describable", and it held in every tier with all
          five intervals excluding zero. It did not survive the test built to check it. Wrong traces run
          <b>1.6× longer</b> than correct ones, and faithfulness falls with position, so length drove both sides
          of the correlation. In a length-matched replication (91 matched pairs, 4,653 fresh readings) the gap
          collapses to <b>−0.0002</b> (95% CI −0.005 to +0.005, p = 0.93) — a null that statistically
          <i>excludes</i> the effect above. Restricting these very same readings to the length-matched cases
          already shrinks the gap 4× (0.0145 → 0.0035, p = 0.21). See <b>What the confirmatory tests found</b>
          below. The chart is kept, uncorrected, because the retraction is the point.</p>
      </div>
    </div>

    <h2 style="margin-top:34px">What the confirmatory tests found</h2>
    <p class="lede">Everything above is <b>exploratory</b> — patterns found by looking at the data. Three
      hypotheses were then written down with their pass/fail rules <b>frozen in advance</b> (2026-08-06), and
      tested. <b>All three failed.</b> That is worth as much page space as a success would have been, because in
      each case the failure has a named cause that a control caught — not "not enough data".</p>

    <div class="vgrid">
      <div class="vcard rej">
        <h3>HT12 · Does faithfulness predict correctness?</h3>
        <p class="sub">tested on 91 length-matched pairs · 4,653 readings</p>
        <p class="take"><b class="no">Refuted.</b> β = −0.0002 (CI −0.005 to +0.005, p = 0.93). The null is
          <i>informative</i>: its upper bound sits below the original effect, so it rules that effect out rather
          than merely missing it. <b>Cause: reply length.</b> Wrong traces are longer, and faithfulness declines
          with position.</p>
      </div>
      <div class="vcard rej">
        <h3>HT13 · Does read↔reasoning alignment drop after an error?</h3>
        <p class="sub">5,653 judged pairs, blinded · Llama-3.1-8B</p>
        <p class="take"><b class="no">Not adjudicated — the instrument failed its own check.</b> The judge
          separates real pairs from mismatched ones well in aggregate (AUC 0.76). But a second judge agrees with
          it on individual items almost not at all (κ = 0.05), and a dense baseline computed in activation space
          correlates with it at ρ = 0.03 while scoring nearly as well. <b>A score can separate populations
          without being able to rank items</b> — so it is not used to sort anything on this page.</p>
      </div>
      <div class="vcard rej">
        <h3>HT14 · Does the answer show up internally earlier when right?</h3>
        <p class="sub">182 cases · 20 decoy answers per reading</p>
        <p class="take"><b class="no">Refuted.</b> The readings mention the model's own eventual answer
          <b>2.8%</b> of the time — <i>below</i> the 3.6% rate for a randomly chosen foreign answer. 92% of cases
          never reach onset at all. A significant-looking survival test (p = 0.002) turned out to be driven by
          <b>answer strings, not internals</b>: wrong answers are rarer and longer (16 vs 7 characters), so they
          match by chance far less often.</p>
      </div>
    </div>
    <p class="lede" style="margin-top:18px">The instrument findings that <i>do</i> survive: the readings are
      reliable about <b>themes</b> and unreliable about <b>specifics</b>; round-trip faithfulness tracks
      <b>position and length</b> rather than comprehension; and the answer is not recoverable from these
      layer-20 readings at all. Full reasoning, including every rule as it was frozen, is in the project's
      <code>log/nla-harness/</code> entries for 2026-08-07.</p>
  </section>

'''

def verdict_panels():
    """The three runs behind the 2026-08-07 verdicts, drawn rather than asserted."""
    g, c = N7R["gate"], CHARTS
    p8 = N8R["primary"]
    own, foreign = p8["own_rate_overall"] * 100, p8["foreign_rate_overall"] * 100
    ex = c["onset_excess"]["mean_excess"]
    conf = c["confusion"]
    fw = next(r for r in c["forest"] if r["key"] == "banked/all-cases")
    fm = next(r for r in c["forest"] if r["key"] == "banked/matched-cases-only")

    return f"""
  <section class="viz">
    <h2>The evidence, drawn</h2>
    <p class="lede">Each figure is the raw output of one run. Read them as instrument checks first —
      what does this measure actually do? — and as verdicts second.</p>

    <div class="vgrid">
      <div class="vcard">
        <h3>Where the faithfulness effect went</h3>
        <p class="sub">gap in mean round-trip faithfulness, right minus wrong · case-level bootstrap</p>
        {CH.forest(c["forest"])}
        <p class="figcap">Top row is the original finding. Every row below it is the same measurement
          under a tighter control. <b>Restricting the very same readings to length-matched cases</b>
          (row 2) already collapses it from {fw["diff"]:+.4f} to {fm["diff"]:+.4f}, and the interval
          crosses zero. Nothing about the readings changed — only which cases were compared.</p>
      </div>

      <div class="vcard">
        <h3>Can a judge tell a real pairing from a fake one?</h3>
        <p class="sub">{sum(v["n"] for v in c["judge_dist"].values()):,} blinded comparisons · scored 0–3 by Llama-3.1-8B</p>
        {CH.judge_dist(c["judge_dist"])}
        <p class="figcap">Yes, in aggregate. A matched reading-and-step pair scores 2 nearly half the
          time; an unrelated pair scores 0 most of the time. And it is not merely spotting a shared
          topic — pairs from the <i>same transcript</i> but a distant step sit much closer to the
          unrelated floor than to a real match.</p>
      </div>

      <div class="vcard">
        <h3>…but is it worth an 8B judge?</h3>
        <p class="sub">separating matched from unrelated pairs · same items for both measures</p>
        {CH.roc(c["roc"])}
        <p class="figcap">The red curve is free: cosine between the reading and the step after both are
          pushed back into activation space by the reconstructor. It costs no judge, no prompt, no GPU
          hour — and lands {(c["roc"]["judge"]["auc"]-c["roc"]["ar"]["auc"])*100:.1f} AUC points behind.
          The two disagree almost completely on <i>individual</i> items (ρ = {c["roc"]["spearman"]:.2f}),
          so they are not two views of one quantity; they are two diffuse signals that happen to
          separate the same populations.</p>
      </div>

      <div class="vcard">
        <h3>Why the judge score cannot rank a single reading</h3>
        <p class="sub">the same {conf["n"]} items, scored independently by two model families</p>
        {CH.confusion(conf)}
        <p class="figcap">This is the figure that vetoed the hypothesis. A second judge
          (Phi-3.5-mini) answers <b>“2” on {conf["matrix"][0][2]+conf["matrix"][1][2]+conf["matrix"][2][2]+conf["matrix"][3][2]} of
          {conf["n"]} items</b> and never once answers 1 — the whole matrix collapses into one column.
          κ = {g.get("G3_kappa_quadratic_score", float("nan")):.2f}. A score that two raters assign this
          differently can describe a population but cannot order individual readings, which is exactly
          what a “most aligned” sort would have needed. <b>That sort is therefore not offered.</b></p>
      </div>

      <div class="vcard">
        <h3>Does alignment fall after the model goes wrong?</h3>
        <p class="sub">mean judge alignment by position · shaded band = 95% CI, bootstrapped over cases</p>
        {CH.align_by_pos(c["align_by_pos"])}
        <p class="figcap">The pre-registered claim was that alignment drops <i>specifically</i> after the
          error in wrong runs. It does fall late — in <b>both</b> groups, by almost the same amount, and
          the two lines stay within each other's intervals for the first nine tenths of the trace. They
          part only in the final decile, and in the <i>opposite</i> direction to the prediction: the
          right-hand runs tick back up as they reach their answer line. Fitted over the whole trace the
          interaction is {N7R.get("ht13_not_adjudicated", {}).get("beta_interaction", 0):+.3f}
          (p = {N7R.get("ht13_not_adjudicated", {}).get("p", float("nan")):.2f}) — the wrong sign, and
          indistinguishable from no effect. (Shown for completeness: the reliability check above means
          this was never adjudicated.)</p>
      </div>

      <div class="vcard">
        <h3>Do the readings contain the answer?</h3>
        <p class="sub">how often a reading mentions the model's own answer, minus the decoy floor</p>
        {CH.onset_excess(c["onset_excess"])}
        <p class="figcap">Zero means “no more often than an answer borrowed from another case”. Wrong
          runs sit <b>below</b> the floor for the entire trace ({ex["0"]["mean"]:+.4f} overall), and right
          runs hover at it ({ex["1"]["mean"]:+.4f}, interval crossing zero). Overall a reading names the
          model's own answer {own:.1f}% of the time against a {foreign:.1f}% chance rate. The decoys are
          what make this readable: without them, {own:.1f}% looks like a faint signal instead of a floor.</p>
      </div>
    </div>
  </section>
"""


def invention_section():
    """N9: how often a reading invents a specific, and what that looks like up close."""
    c, by = CONF, CONF["by_true_language"]
    q = CONF["by_faithfulness_quintile"]
    gap = CONF["q5_minus_q1_error_rate"]

    # Hand-picked from the mined candidates, then read and kept on what the text actually says --
    # the judge that surfaced them is unreliable per item (kappa 0.05), so it selects, never decides.
    EX = [
      {"cls": "contradicts", "case": "JavaScript/63 · adversarial rename · model got this WRONG",
       "step": "For any other value of <code>_lastNSecs</code>, the function returns the sum of "
               "three recursive calls: <code>smoothArea(_lastNSecs - 1)</code>, "
               "<code>smoothArea(_lastNSecs - 2)</code>, and <code>smoothArea(_lastNSecs - 3)</code>.",
       "read": "Structured Python code explanation … showing <b>iterative factorial calculation "
               "using a queue</b>, with the pattern defining <code>f(n)</code> behaviour for "
               "specific cases. The sentence “The formula checks <b>if n == 1 or n == 2: returns 1</b>, "
               "otherwise” establishes the …",
       "why": "The step is a <b>three-term</b> recursion. The reading calls it a <b>factorial</b> with "
              "a two-case base. Not vagueness — a different recurrence."},
      {"cls": "contradicts", "case": "cruxeval JavaScript/113 · neutral rename",
       "step": "After processing all characters, join the array <code>d</code> to form the final "
               "string. Now, let's go through the process for the given string: “9” → Odd position, "
               "no change → “9”",
       "read": "Structured algorithmic explanation … showing <b>binary conversion pattern for “ABCD” "
               "sequence</b>, transitioning to concrete enumeration of each step's contribution.",
       "why": "The step transforms characters by position. The reading reports a binary conversion "
              "of a different string entirely."},
      {"cls": "shape", "case": "synthetic slicing task 05",
       "step": "<b>Line 5</b>: <code>theta = alpha + value</code> — since <code>alpha</code> is 40 "
               "and <code>value</code> is 7, <code>theta</code> is updated to 47.",
       "read": "Structured Python documentation … demonstrating variable assignment using the "
               "<code>:=</code> operator, showing sequential evaluation of expressions with "
               "<code>x</code> and <code>y</code>. The sentence “For the first assignment, "
               "<b>x + y = 10 + 5</b>, so y is updated” …",
       "why": "The <i>shape</i> is exactly right — a line assigning the sum of two variables, "
              "narrated in order. Every particular is invented: the operator, both names, both numbers."},
      {"cls": "shape", "case": "humaneval Python/103 · at the answer line",
       "step": "<code>bin(3)</code> returns <code>'0b11'</code>. Therefore, the exact output of "
               "<code>rounded_avg(1, 5)</code> is: Output: <code>'0b11'</code>",
       "read": "Structured Python code explanation … showing function execution details for "
               "<code>calculate()</code> method, systematically walking through each operation to "
               "produce a <b>final output value</b>.",
       "why": "It knows the model is committing to a final answer — that part is right and reliable. "
              "The function is <code>rounded_avg</code>, not <code>calculate()</code>."},
    ]
    cards = "".join(
      f'''<div class="exrow {e["cls"]}">
        <p class="excase">{e["case"]}</p>
        <div class="expair">
          <div><p class="exlab">the reasoning step it was read at</p><p class="exstep">{e["step"]}</p></div>
          <div><p class="exlab">what the reading said</p><p class="exread">{e["read"]}</p></div>
        </div>
        <p class="exwhy">{e["why"]}</p>
      </div>''' for e in EX)

    return f"""
  <section class="viz">
    <h2>What the readings invent</h2>
    <p class="lede">The verbalizer never sees the code — only a 3,584-number activation — so any
      specific it names is an inference, never a copy. That makes one specific <b>mechanically
      checkable</b>: each task has a known language, and the readings name a language unprompted
      {c["named_pct"]:.0f}% of the time. No judge, no annotation, no model required — a rare thing here,
      and the reason this is the most trustworthy measurement on the page.</p>

    <div class="vgrid">
      <div class="vcard">
        <h3>It names a language {c["named_pct"]:.0f}% of the time. It is wrong {c["wrong_pct"]:.0f}% of the time.</h3>
        <p class="sub">{c["named_a_language"]:,} of {c["n_readings"]:,} readings · language known from the stimulus</p>
        {CH.confab_by_kind(CONF["by_kind"])}
        <p class="figcap">The split is the finding. Reading a token <b>of the code</b>, it gets the
          language right {100-CONF["by_kind"]["code"]["wrong_pct"]:.0f}% of the time. Reading the model's
          own <b>reasoning prose</b>, it is wrong {CONF["by_kind"]["cot"]["wrong_pct"]:.0f}% of the time.
          The claim tracks whether literal syntax sits at the read position — not what language the task
          is in.</p>
      </div>

      <div class="vcard">
        <h3>And it is not confabulating randomly — it defaults to Python</h3>
        <p class="sub">by the task's actual language</p>
        <div class="bigstat">
          <div><b>{by["python"]["wrong_pct"]:.0f}%</b><small>wrong on <b>Python</b> tasks<br>({by["python"]["named"]:,} readings)</small></div>
          <div><b>{by["javascript"]["wrong_pct"]:.0f}%</b><small>wrong on <b>JavaScript</b> tasks<br>({by["javascript"]["named"]:,} readings)</small></div>
        </div>
        <p class="figcap">On JavaScript code the readings say “Python” <b>{by["javascript"]["top_claims"].get("python",0):,}</b>
          times and “JavaScript” {by["javascript"]["top_claims"].get("javascript",0):,}. This is a
          <b>prior</b>, not noise: the language field of a reading carries almost no information about
          the code it was taken from. Anything a reading says about libraries, method names or syntax
          should be read the same way.</p>
      </div>

      <div class="vcard">
        <h3>Does faithfulness catch the invention?</h3>
        <p class="sub">wrong-language rate by round-trip faithfulness quintile</p>
        {CH.confab_by_faith(q)}
        <p class="figcap">Barely. The best-reconstructed fifth of readings still names the wrong
          language <b>{q[-1]["wrong_pct"]:.0f}%</b> of the time, against {q[0]["wrong_pct"]:.0f}% for the
          worst (gap {abs(gap["mean"])*100:.1f} points, 95% CI {abs(gap["ci95"][1])*100:.1f}–{abs(gap["ci95"][0])*100:.1f}).
          Faithfulness helps a little and is nowhere near sufficient. This is the sharpest available
          demonstration that <b>a high recovery score means the sentence carried the vector, not that
          the sentence is true</b> — the caveat printed everywhere else on this page, measured.</p>
      </div>

      <div class="vcard">
        <h3>We looked for planning. We did not find it.</h3>
        <p class="sub">words a reading names that the trace only writes ≥400 characters later</p>
        {CH.lookahead_null(LOOKNULL)}
        <p class="figcap">The tempting claim is that a reading anticipates where the reasoning is
          going, and some appear to. But the test has to beat a null: does a reading from a
          <i>different case</i> anticipate this trace just as well? <b>It does.</b> The apparent
          look-ahead is generic reasoning vocabulary — “evaluate”, “final”, “check” — that any trace
          eventually contains. {LOOKNULL["n_reads"]:,} readings tested against
          {LOOKNULL["n_foreign_draws"]:,} foreign draws; the rate of naming ≥3 words ahead is actually
          <i>lower</i> for a reading's own trace ({LOOKNULL["p_ge3_own"]:.1%}) than for a foreign one
          ({LOOKNULL["p_ge3_foreign"]:.1%}). <b>No evidence of planning by this measure.</b> The 43
          candidates that survived the distance filter are not shown, because they do not survive this.</p>
      </div>

      <div class="vcard">
        <h3>A prediction of mine that was wrong</h3>
        <p class="sub">wrong-language rate by position through the trace, JavaScript tasks</p>
        <p class="take">Every one of the worst confabulations sat at the very start of the reply, so
          the natural story was <i>a prior that gets corrected as language-specific evidence
          accumulates</i>. It does the opposite: the error rate <b>rises</b> from
          {CONF["by_position"]["javascript"][0]["wrong_pct"]:.0f}% in the first tenth of the trace to
          {CONF["by_position"]["javascript"][-1]["wrong_pct"]:.0f}% in the last
          (r = {CONF.get("position_corr_javascript", 0):+.2f}). Deeper into a trace the model writes
          language-agnostic prose, so there is <i>less</i> syntax to go on, not more — which is the same
          thing the code-vs-reasoning split says. Recorded because the failed prediction is what
          identified the right explanation.</p>
      </div>

      <div class="vcard">
        <h3>What would settle this</h3>
        <p class="sub">the open question, and the run that answers it</p>
        <p class="take">Is the Python default a property of <b>the reader</b> — a verbalizer trained
          mostly on Python — or of <b>the model being read</b>, whose internal state for “walking
          through code” may genuinely be language-agnostic? Nothing measured here separates them: both
          predict exactly what we see. It is answerable, and cheaply — read the same JavaScript
          activations while varying only the reader's prompt context. <b>Not run</b>: it needs a
          generation pass, and the box's GPUs are busy. Stated here so the limitation is on the page
          rather than in a file nobody opens.</p>
      </div>
    </div>

    <h3 class="exhead">Four readings, up close</h3>
    <p class="lede">Surfaced by the judge, then read and kept on what the text says — the judge is
      unreliable item-by-item, so it selects candidates and never decides. Both columns are verbatim.</p>
    {cards}
  </section>
"""


def tier_cards():
    out = []
    for t in ("L0", "L1", "L1b", "L2", "L3"):
        a = stats["acc"].get(f"dataset_a|{t}")
        b = stats["acc"].get(f"dataset_b|{t}")
        pooled_n = (a["n"] if a else 0) + (b["n"] if b else 0)
        pooled_c = ((a["acc"] * a["n"]) if a else 0) + ((b["acc"] * b["n"]) if b else 0)
        pct = pooled_c / pooled_n if pooled_n else 0
        bars = ""
        for lbl, d in (("A", a), ("B", b)):
            v = d["acc"] if d else 0
            bars += (f'<span class="bar"><i><b style="width:{v*100:.0f}%"></b></i>'
                     f'<small>{lbl} {v:.2f}</small></span>')
        out.append(
            f'<div class="tier-card"><div class="tt"><code>{t}</code>'
            f'<span class="pct">{pct:.2f}</span></div>'
            f'<p>{html.escape(TIER_NOTE[t])}</p><div class="bars">{bars}</div></div>')
    return "\n".join(out)


HTML = f"""<title>NLA readings of obfuscated-code comprehension — results browser</title>
<style>{CSS}</style>
<div class="wrap">
  <header class="mast">
    <p class="eyebrow">Instrument 2 · out-of-the-box capture · 2026-08-05</p>
    <h1>What the model <em>was actually thinking</em>, token by token</h1>
    <p class="dek">Every case below pairs a code-comprehension task with the model's own reasoning
      and with <strong>NLA readings</strong> — plain-English descriptions of the layer-20 residual
      vector at single tokens, produced by a frozen Natural Language Autoencoder that never saw
      code during training. Click any reading to locate the exact token it describes.</p>
    <div class="meta">
      <span>Qwen2.5-7B-Instruct</span><span>layer 20 · d=3584</span>
      <span>NLA kitft/nla-qwen2.5-7b-L20</span><span>greedy · seed 20260724</span>
      <span>unattended run 2026-08-04 → 08-05</span>
    </div>
    <div class="stats">
      <div class="stat"><b>{stats['cases']}</b><small>task runs</small></div>
      <div class="stat"><b>{stats['reads']:,}</b><small>token readings</small></div>
      <div class="stat"><b>{stats['reads']//stats['cases']}</b><small>readings per run</small></div>
      <div class="stat"><b>5</b><small>obfuscation tiers</small></div>
      <div class="stat"><b>70</b><small>base problems</small></div>
    </div>
  </header>

  <section class="ladder">
    <h2>The obfuscation ladder — and where the model breaks</h2>
    <div class="tiers">{tier_cards()}</div>
  </section>

{viz_section()}
{verdict_panels()}
{invention_section()}
  <section class="method">
    <h2>What one “case” is, and what one “reading” is</h2>
    <p>A <strong>case</strong> is a single task run: one code snippet, at one obfuscation tier, with one
      question, answered once by the model at temperature 0. A <strong>reading</strong> is one pass of the
      Natural Language Autoencoder over the model's internal state at <em>one token</em> of that run —
      layer 20 of 28, a single 3,584-number vector, described in English.</p>
    <div class="breakdown">
      <div class="bd"><b>{KINDS["output_prediction"]}</b> graded output-prediction runs<small>predict the exact return value; scored against ground truth</small></div>
      <div class="bd"><b>{KINDS["prompt_reads"]}</b> prompt-only runs<small>no usable call input upstream — readings captured, nothing graded</small></div>
      <div class="bd"><b>{KINDS["slice_prediction"]}</b> synthetic slicing runs<small>which lines affect the printed value; truth known by construction</small></div>
      <div class="bd"><b>{stats["dense_cases"]}</b> cases read a second time, densely<small>+{stats["dense_added"]:,} extra readings from the 2026-08-06 pass — roughly 4× the token coverage on those transcripts. Filter to them with <b>Coverage → deep</b>.</small></div>
    </div>
    <div class="breakdown">
      <div class="bd"><b>≤6</b> at identifiers<small>where a misleading name would do its work</small></div>
      <div class="bd"><b>≤3</b> at dispatcher tokens<small>flattened tiers only</small></div>
      <div class="bd"><b>6</b> through the reasoning<small>evenly spread across the chain of thought</small></div>
      <div class="bd"><b>3</b> on the answer line<small>the moment of commitment</small></div>
    </div>
    <div class="mgrid">
      <div class="mcol">
        <h3>Prompt given to the model under test</h3>
        <pre>{html.escape(PREAMBLE)}<b>&lt;the code, shown per case&gt;</b>

<b>&lt;the question, shown per case&gt;</b></pre>
        <p class="mnote">Fixed preamble for every case; the code and question vary. Open
          “Full prompt sent to the model” inside any case to read the exact text.</p>
      </div>
      <div class="mcol">
        <h3>Prompt given to the NLA verbalizer</h3>
        <pre>{html.escape(AV_PROMPT)}</pre>
        <p class="mnote">The <b>㈎</b> marker is where the model's activation vector is injected in place of a
          token embedding (rescaled to norm 150). The verbalizer never sees the code, the question, or the
          answer — only the vector.</p>
      </div>
    </div>
  </section>

  <section class="legend">
    <h2>How to read one of these cards</h2>
    <p class="lead">Every reading below is a description of the model's internal state at
      <strong>exactly one token</strong>. The strip under each heading shows that token highlighted in
      its surroundings, so you can always see precisely what was read.</p>
    <dl>
      <dt>in the code</dt>
      <dd>The token sits in the source you gave the model. The italic phrase says what kind of thing it is —
        <em>a misleading name</em> is the decoy the trap tiers introduce, <em>an original name</em> is the
        real one, <em>the dispatcher variable</em> is the state variable of a flattened loop.
        Tokens are often fragments, so <code>piece of _lastNSecs</code> tells you which full identifier
        the fragment belongs to.</dd>
      <dt>in the reasoning</dt>
      <dd>The token sits inside the model's own chain of thought, at the token number shown. Click the
        card and the exact spot lights up in the transcript on the left.</dd>
      <dt>on the answer line</dt>
      <dd>The token sits on the final <code>Output:</code> or <code>Lines:</code> line — the moment the
        model commits to an answer.</dd>
      <dt>recovers N%</dt>
      <dd><strong>How faithful the reading is.</strong> The reconstructor takes the reading's text alone and
        tries to rebuild the original vector from it; this is how close it gets. Around 90% means the
        sentence captured most of what was in the model's head; 70% means a lot was lost. It says nothing
        about whether the reading is <em>true</em> — only how complete it is.</dd>
      <dt>real function ↔ decoy name</dt>
      <dd><strong>Only on trap tiers.</strong> The misleading name suggests one meaning
        (<code>smoothArea</code> → averaging over time); the code actually does another
        (a three-term recurrence). We reconstruct a vector for each of those two descriptions and measure
        which one this token's state sits closer to. The bar leans <span class="sw-a">left</span> when the
        model's internal state matches the real function, <span class="sw-t">right</span> when it has been
        pulled toward the decoy. Compare bars across tiers rather than trusting the absolute number —
        the two reference descriptions are generated from a crude template.</dd>
    </dl>
  </section>

  <nav class="controls" aria-label="Filter cases">
    <div class="grp"><label>Tier</label>
      <button class="chip" data-tier="L0" aria-pressed="false" type="button">L0</button>
      <button class="chip" data-tier="L1" aria-pressed="false" type="button">L1</button>
      <button class="chip" data-tier="L1b" aria-pressed="false" type="button">L1b</button>
      <button class="chip" data-tier="L2" aria-pressed="false" type="button">L2</button>
      <button class="chip" data-tier="L3" aria-pressed="false" type="button">L3</button>
    </div>
    <div class="grp"><label>Verdict</label>
      <button class="chip" data-verdict="ok" aria-pressed="false" type="button">correct</button>
      <button class="chip" data-verdict="bad" aria-pressed="false" type="button">wrong</button>
    </div>
    <div class="grp"><label>Task</label>
      <button class="chip" data-kind="output_prediction" aria-pressed="false" type="button">output</button>
      <button class="chip" data-kind="slice_prediction" aria-pressed="false" type="button">slicing</button>
      <button class="chip" data-kind="prompt_reads" aria-pressed="false" type="button">reads only</button>
    </div>
    <div class="grp"><label>Coverage</label>
      <button class="chip" data-dense="1" aria-pressed="false" type="button">deep (dense pass)</button>
    </div>
    <div class="grp"><label>Sort</label>
      <button class="chip on" data-sort="default" aria-pressed="true" type="button">default</button>
      <button class="chip" data-sort="faith_hi" aria-pressed="false" type="button">most faithful</button>
      <button class="chip" data-sort="faith_lo" aria-pressed="false" type="button">least faithful</button>
    </div>
    <input class="searchbox" id="q" type="search" placeholder="Search readings, code, answers…"
      aria-label="Search readings, code and answers">
    <button class="tbtn" id="reset" type="button">Reset</button>
    <span class="count" id="count"></span>
  </nav>
  <p class="sortnote"><b>About the sort.</b> It orders cases by mean round-trip faithfulness — how completely
    the reconstructor rebuilds each vector from the reading's text. That is a measure of <b>completeness, not
    truth, and not correctness</b>: the length-matched test above (HT12) found it does <b>not</b> predict
    whether the model got the answer right. Read it as “which readings carry the most of the original state”,
    nothing more. A sort by <i>read↔reasoning alignment</i> was planned and is deliberately <b>not offered</b>:
    the judge that would produce it failed its reliability check (HT13), so it could not rank cases honestly.
    Cases with no scored readings always sort last.</p>

  <main id="feed"></main>

  <footer class="foot">
    Single greedy runs — each cell is a data point, not a conclusion. Readings are theme-reliable and
    specifics-unreliable: languages, libraries and digits are frequently confabulated.<br>
    Source: data/nla/overnight/2026-08-04/captures.jsonl · report: reports/2026-08-05_nla-out-of-box/REPORT.md
  </footer>
</div>
<script>window.__NLA__ = {payload};</script>
<script>{JS}</script>
"""

OUT.write_text(HTML)
print(f"wrote {OUT} · {OUT.stat().st_size/1_000_000:.2f} MB")
