"""Inline-SVG charts for the results browser.

No chart library: the page ships under a strict CSP with no external requests, and these are six
static figures, so hand-built SVG is both smaller and fully themeable through CSS custom properties.

COLOR IS NOT CHOSEN HERE BY EYE. Every hex below was run through the data-viz validator against
BOTH surfaces the page uses (light #FFFFFF, dark #141B1E) before being written down:

  correct / wrong      blue #2a78d6 / orange #eb6834   (dark #3987e5 / #d95926)  -- all checks pass
  judge / AR baseline  blue #2a78d6 / red    #e34948   (dark #3987e5 / #e66767)  -- all checks pass
  score 0-3            ordinal blue ramp, 4 steps                                -- all checks pass
  heatmap              sequential blue, same ramp

Rejected by the validator rather than by taste: the page's own semantic green/red (`--ok`/`--bad`)
failed the chroma floor and sat at CVD dE 6.8 -- the classic red/green trap; and blue+violet
collapsed in dark mode (CVD dE 1.9, normal-vision 9.8).

`correct` and `wrong` keep the SAME two hues in every chart they appear in -- color follows the
entity, never its position in a list.

Accessibility: every chart has a legend when it has >=2 series, values are direct-labelled rather
than hover-only, and each mark carries a <title> for a native tooltip. Text never wears a series
color; a swatch beside it carries identity.
"""
from __future__ import annotations

import html
import json
from pathlib import Path

# ---------------------------------------------------------------- palette (validated; see docstring)
CSS = """
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
.fig .vl.ondark{fill:#fff}
.fig .vl.onlight{fill:var(--ink)}          /* on a surface-derived pale fill (heatmap): follows theme */
.fig .vl.onpale{fill:#0f1720}              /* on a FIXED pale-blue fill (bar segments s0/s1, pale in
                                              both themes) -- must NOT follow theme or it washes out */
.fig .gl{stroke:var(--grid);stroke-width:1}
.fig .zl{stroke:var(--ink-3);stroke-width:1;stroke-dasharray:3 3}
.fig .ref{stroke:var(--ink-3);stroke-width:1;stroke-dasharray:4 4;opacity:.6}
.figlegend{display:flex;flex-wrap:wrap;gap:6px 15px;font-size:11px;color:var(--ink-2);
  margin:0 0 9px;align-items:center}
.figlegend span{display:inline-flex;align-items:center;gap:5px}
.figlegend i{width:11px;height:11px;border-radius:2px;display:inline-block;flex:none}
.figstat{font-size:11px;color:var(--ink-2);margin:8px 0 0;font-variant-numeric:tabular-nums}
.figcap{font-size:11.5px;color:var(--ink-3);line-height:1.5;margin:7px 0 0}
"""

W = 460          # viewBox width; the SVG scales to its container


def _esc(s) -> str:
    return html.escape(str(s), quote=True)


def legend(items: list[tuple[str, str]]) -> str:
    """items = [(css color expression, label)]"""
    return ('<p class="figlegend">'
            + "".join(f'<span><i style="background:{c}"></i>{_esc(l)}</span>' for c, l in items)
            + "</p>")


# ---------------------------------------------------------------- 1. stacked score distribution
def judge_dist(d: dict) -> str:
    """100% stacked bars: how the 0-3 scores redistribute as the pairing gets less related."""
    rows = [("real", "matched pair"), ("distant", "same trace, far away"),
            ("shuffled", "different case")]
    H, bar, gap, left = 0, 26, 16, 128
    H = len(rows) * (bar + gap) + 26
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Judge score distribution by pairing">']
    plot = W - left - 24      # breathing room at the right edge for the thin top segment
    for i, (key, lab) in enumerate(rows):
        y = i * (bar + gap)
        r = d[key]
        out.append(f'<text class="ax" x="{left-8}" y="{y+bar/2+3.5}" text-anchor="end">{_esc(lab)}</text>')
        x = left
        for s in range(4):
            pct = r["pct"][s]
            w = plot * pct / 100
            if w <= 0:
                continue
            # 2px surface gap between segments, per the mark spec
            out.append(
                f'<rect x="{x:.2f}" y="{y}" width="{max(w-2,0.6):.2f}" height="{bar}" rx="2" '
                f'fill="var(--s{s})"><title>{_esc(lab)} · score {s} · '
                f'{pct:.1f}% ({r["counts"][s]:,} of {r["n"]:,})</title></rect>')
            if w > 34:
                cls = "vl ondark" if s >= 2 else "vl onpale"
                out.append(f'<text class="{cls}" x="{x+w/2-1:.2f}" y="{y+bar/2+3.5}" '
                           f'text-anchor="middle">{pct:.0f}%</text>')
            x += w
        out.append(f'<text class="vl" x="{W-4}" y="{y+bar/2+3.5}" text-anchor="end" '
                   f'opacity="0">.</text>')
        out.append(f'<text class="ax" x="{left}" y="{y+bar+11}">mean {r["mean"]:.2f} / 3 '
                   f'· n={r["n"]:,}</text>')
    out.append("</svg>")
    return (legend([(f"var(--s{s})", f"score {s}") for s in range(4)]) + '<div class="fig">'
            + "".join(out) + "</div>")


# ---------------------------------------------------------------- 2. ROC
def roc(d: dict) -> str:
    S, pad = 250, 34
    VBW = S + pad + 16          # square plot in a tailored viewBox, so CSS scales it up to the
                                # card width instead of padding it out with dead space
    def X(v): return pad + v * S
    def Y(v): return pad + (1 - v) * S
    out = [f'<svg viewBox="0 0 {VBW} {S+pad+36}" role="img" '
           f'aria-label="ROC curves for the judge and the activation-space baseline">']
    for v in (0, .5, 1):        # axis ticks were missing entirely on the first render
        out.append(f'<text class="ax" x="{X(v)}" y="{Y(0)+15}" text-anchor="middle">{v:g}</text>')
        out.append(f'<text class="ax" x="{pad-7}" y="{Y(v)+3.5}" text-anchor="end">{v:g}</text>')
    for g in (0, .25, .5, .75, 1):
        out.append(f'<line class="gl" x1="{X(0)}" y1="{Y(g)}" x2="{X(1)}" y2="{Y(g)}"/>')
        out.append(f'<line class="gl" x1="{X(g)}" y1="{Y(0)}" x2="{X(g)}" y2="{Y(1)}"/>')
    out.append(f'<line class="ref" x1="{X(0)}" y1="{Y(0)}" x2="{X(1)}" y2="{Y(1)}"/>')
    out.append(f'<text class="ax" x="{X(1)-2}" y="{Y(1)-6}" text-anchor="end">chance</text>')
    for key, colour, lab in (("ar", "var(--c-ar)", "activation-space baseline"),
                             ("judge", "var(--c-judge)", "Llama-3.1-8B judge")):
        pts = d[key]["points"]
        path = " ".join(f'{"M" if i==0 else "L"}{X(p["fpr"]):.2f},{Y(p["tpr"]):.2f}'
                        for i, p in enumerate(pts))
        out.append(f'<path d="{path}" fill="none" stroke="{colour}" stroke-width="2" '
                   f'stroke-linejoin="round"><title>{_esc(lab)} · AUC {d[key]["auc"]:.3f}</title></path>')
        if key == "judge":       # the judge's 4-level scale gives a handful of operating points
            for p in pts[1:-1]:
                out.append(f'<circle cx="{X(p["fpr"]):.2f}" cy="{Y(p["tpr"]):.2f}" r="4" '
                           f'fill="{colour}" stroke="var(--figsurface)" stroke-width="2"/>')
    out.append(f'<text class="ax" x="{X(.5)}" y="{Y(0)+30}" text-anchor="middle">'
               f'false positives — unrelated pairs called a match</text>')
    out.append(f'<text class="ax" transform="translate({pad-24},{Y(.5)}) rotate(-90)" '
               f'text-anchor="middle">true positives</text>')
    # AUCs called out on the plot rather than hidden in a tooltip
    out.append(f'<text class="vl" x="{X(.42)}" y="{Y(.30)}" fill="var(--ink)">'
               f'judge AUC {d["judge"]["auc"]:.2f}</text>')
    out.append(f'<text class="vl" x="{X(.42)}" y="{Y(.30)+14}" fill="var(--ink)">'
               f'baseline AUC {d["ar"]["auc"]:.2f}</text>')
    out.append("</svg>")
    return (legend([("var(--c-judge)", "Llama-3.1-8B judge"),
                    ("var(--c-ar)", "activation-space baseline (free)")])
            + '<div class="fig">' + "".join(out) + "</div>")


# ---------------------------------------------------------------- 3. agreement heatmap
def confusion(d: dict) -> str:
    m = d["matrix"]
    n = max(max(r) for r in m) or 1
    cell, left, top = 52, 78, 34
    H = top + cell * 4 + 8
    VBW = left + cell * 4 + 14   # tailored viewBox: the matrix fills its card
    out = [f'<svg viewBox="0 0 {VBW} {H}" role="img" '
           f'aria-label="Agreement matrix between the two judges">']
    out.append(f'<text class="ax" x="{left}" y="12">second judge (Phi-3.5) said</text>')
    for c in range(4):
        out.append(f'<text class="ax" x="{left+c*cell+cell/2}" y="{top-7}" '
                   f'text-anchor="middle">{c}</text>')
    out.append(f'<text class="ax" transform="translate({left-52},{top+cell*2}) rotate(-90)" '
               f'text-anchor="middle">Llama said</text>')
    for r in range(4):
        out.append(f'<text class="ax" x="{left-10}" y="{top+r*cell+cell/2+3.5}" '
                   f'text-anchor="end">{r}</text>')
        for c in range(4):
            v = m[r][c]
            # sequential single hue: opacity carries magnitude on one validated blue
            op = 0.06 + 0.94 * (v / n) ** 0.65 if v else 0.03
            cls = "vl ondark" if op > 0.55 else "vl onlight"
            out.append(
                f'<rect x="{left+c*cell+1}" y="{top+r*cell+1}" width="{cell-2}" height="{cell-2}" '
                f'rx="3" fill="var(--s2)" fill-opacity="{op:.3f}"/>'
                f'<rect x="{left+c*cell+1}" y="{top+r*cell+1}" width="{cell-2}" height="{cell-2}" '
                f'rx="3" fill="none" stroke="var(--grid)"><title>Llama {r} · Phi {c} · '
                f'{v} of {d["n"]}</title></rect>')
            if v:
                out.append(f'<text class="{cls}" x="{left+c*cell+cell/2}" '
                           f'y="{top+r*cell+cell/2+3.5}" text-anchor="middle">{v}</text>')
    out.append("</svg>")
    # kept OUT of the SVG: at this tailored viewBox width the line was clipped
    return ('<div class="fig">' + "".join(out) + "</div>"
            + f'<p class="figstat">exact agreement on the diagonal: <b>{d["agree_pct"]:.0f}%</b>'
              f' · mean score {d["llama_mean"]} (Llama) vs {d["phi_mean"]} (Phi-3.5)</p>')


# ---------------------------------------------------------------- 4 & 6. line charts over position
def _line_chart(series: dict, lo: float, hi: float, ylab: str, aria: str,
                zero: bool = False, pct: bool = False, bands: bool = True) -> str:
    S, padl, padt, padb = 178, 44, 12, 30
    plot = W - padl - 12
    def X(b): return padl + (b + 0.5) / 10 * plot
    def Y(v): return padt + (1 - (v - lo) / (hi - lo)) * S
    out = [f'<svg viewBox="0 0 {W} {S+padt+padb}" role="img" aria-label="{_esc(aria)}">']
    steps = 5
    for k in range(steps):
        v = lo + (hi - lo) * k / (steps - 1)
        out.append(f'<line class="gl" x1="{padl}" y1="{Y(v):.2f}" x2="{padl+plot}" y2="{Y(v):.2f}"/>')
        lab = f"{v*100:.0f}%" if pct else f"{v:+.2f}" if zero else f"{v:.2f}"
        out.append(f'<text class="ax" x="{padl-6}" y="{Y(v)+3.5:.2f}" text-anchor="end">{lab}</text>')
    if zero and lo < 0 < hi:
        out.append(f'<line class="zl" x1="{padl}" y1="{Y(0):.2f}" x2="{padl+plot}" y2="{Y(0):.2f}"/>')
    for name, colour in (("wrong", "var(--c-wrong)"), ("correct", "var(--c-correct)")):
        pts = [p for p in series[name] if p]
        if not pts:
            continue
        if bands and "lo" in pts[0]:
            up = " ".join(f'{X(p["bin"]):.2f},{Y(p["hi"]):.2f}' for p in pts)
            dn = " ".join(f'{X(p["bin"]):.2f},{Y(p["lo"]):.2f}' for p in reversed(pts))
            out.append(f'<polygon points="{up} {dn}" fill="{colour}" fill-opacity=".13"/>')
        d = " ".join(f'{"M" if i==0 else "L"}{X(p["bin"]):.2f},{Y(p["mean"]):.2f}'
                     for i, p in enumerate(pts))
        out.append(f'<path d="{d}" fill="none" stroke="{colour}" stroke-width="2" '
                   f'stroke-linejoin="round" stroke-linecap="round"/>')
        for p in pts:
            t = (f'{name} · {int(p["bin"]*10)}–{int(p["bin"]*10+10)}% through the trace · '
                 f'{p["mean"]:+.4f}' if zero else
                 f'{name} · {int(p["bin"]*10)}–{int(p["bin"]*10+10)}% through the trace · '
                 f'{p["mean"]*100:.1f}%' + (f' (n={p["n"]})' if "n" in p else ""))
            out.append(f'<circle cx="{X(p["bin"]):.2f}" cy="{Y(p["mean"]):.2f}" r="4" fill="{colour}" '
                       f'stroke="var(--figsurface)" stroke-width="2"><title>{_esc(t)}</title></circle>')
    for b in (0, 2, 4, 6, 8, 9):
        out.append(f'<text class="ax" x="{X(b):.2f}" y="{S+padt+15}" text-anchor="middle">'
                   f'{b*10}%</text>')
    out.append(f'<text class="ax" x="{padl+plot/2}" y="{S+padt+28}" text-anchor="middle">'
               f'position through the reasoning trace</text>')
    out.append("</svg>")
    return ('<div class="fig">' + "".join(out) + "</div>")


def align_by_pos(d: dict) -> str:
    vals = [p["lo"] for s in d.values() for p in s if p] + [p["hi"] for s in d.values() for p in s if p]
    lo, hi = min(vals) - .01, max(vals) + .01
    return (legend([("var(--c-correct)", "runs the model got right"),
                    ("var(--c-wrong)", "runs it got wrong")])
            + _line_chart(d, lo, hi, "alignment", "Judge alignment across the reasoning trace",
                          pct=True))


def onset_excess(d: dict) -> str:
    ser = {k: [{"bin": i, "mean": v} for i, v in enumerate(d[k])] for k in ("correct", "wrong")}
    vals = [v for k in ("correct", "wrong") for v in d[k]]
    m = max(abs(min(vals)), abs(max(vals))) * 1.25
    return (legend([("var(--c-correct)", "runs the model got right"),
                    ("var(--c-wrong)", "runs it got wrong")])
            + _line_chart(ser, -m, m, "excess", "Answer mentions above the decoy floor",
                          zero=True, bands=False))


# ---------------------------------------------------------------- 5. forest
def forest(rows: list[dict]) -> str:
    left, rowh, padt = 210, 30, 20
    H = padt + len(rows) * rowh + 30
    lo = min(min(r["lo"] for r in rows), 0) - .003
    hi = max(max(r["hi"] for r in rows), 0) + .003
    plot = W - left - 58      # reserve a right column so value labels cannot hit the whiskers
    def X(v): return left + (v - lo) / (hi - lo) * plot
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" '
           f'aria-label="Faithfulness gap between right and wrong runs, by analysis">']
    out.append(f'<line class="zl" x1="{X(0):.2f}" y1="{padt-8}" x2="{X(0):.2f}" y2="{padt+len(rows)*rowh}"/>')
    out.append(f'<text class="ax" x="{X(0):.2f}" y="{padt-12}" text-anchor="middle">no difference</text>')
    for i, r in enumerate(rows):
        y = padt + i * rowh + rowh / 2
        crosses = r["lo"] <= 0 <= r["hi"]
        colour = "var(--ink-3)" if crosses else "var(--c-correct)"
        out.append(f'<text class="ax" x="{left-10}" y="{y+3.5}" text-anchor="end">'
                   f'{_esc(r["label"])}</text>')
        out.append(f'<line x1="{X(r["lo"]):.2f}" y1="{y}" x2="{X(r["hi"]):.2f}" y2="{y}" '
                   f'stroke="{colour}" stroke-width="2" stroke-linecap="round"/>')
        for e in ("lo", "hi"):
            out.append(f'<line x1="{X(r[e]):.2f}" y1="{y-4}" x2="{X(r[e]):.2f}" y2="{y+4}" '
                       f'stroke="{colour}" stroke-width="2"/>')
        out.append(f'<circle cx="{X(r["diff"]):.2f}" cy="{y}" r="5" fill="{colour}" '
                   f'stroke="var(--figsurface)" stroke-width="2"><title>'
                   f'{_esc(r["label"])} · {r["diff"]:+.4f} (95% CI {r["lo"]:+.4f} to {r["hi"]:+.4f}) · '
                   f'{r["n_cases"]} cases · p={r["p"]:.3g}</title></circle>')
        out.append(f'<text class="vl" x="{W-6}" y="{y+3.5}" text-anchor="end" '
                   f'fill="{"var(--ink-3)" if crosses else "var(--ink)"}">{r["diff"]:+.4f}</text>')
    out.append(f'<text class="ax" x="{left+plot/2}" y="{H-8}" text-anchor="middle">'
               f'faithfulness gap, right minus wrong (95% CI)</text>')
    out.append("</svg>")
    return ('<p class="figlegend"><span><i style="background:var(--c-correct)"></i>'
            'interval excludes zero</span><span><i style="background:var(--ink-3)"></i>'
            'interval includes zero — no effect</span></p>'
            + '<div class="fig">' + "".join(out) + "</div>")


def load(path: Path) -> dict:
    return json.loads(Path(path).read_text())


# ---------------------------------------------------------------- 7. confabulation rate by context
def confab_by_kind(d: dict) -> str:
    """Single series, so no legend: the title names it. Sorted, with direct value labels."""
    lab = {"code": "reading a token of the code", "dispatcher": "reading a dispatcher variable",
           "answer": "reading the answer line", "cot": "reading the model's own reasoning"}
    rows = sorted(((lab.get(k, k), v["wrong_pct"], v["named"]) for k, v in d.items()),
                  key=lambda r: r[1])
    bar, gap, left = 22, 14, 208
    H = len(rows) * (bar + gap) + 24
    plot = W - left - 52
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" '
           f'aria-label="Wrong-language rate by what the reading was taken from">']
    for g in (0, 25, 50):
        x = left + plot * g / 60
        out.append(f'<line class="gl" x1="{x:.1f}" y1="0" x2="{x:.1f}" y2="{len(rows)*(bar+gap)-gap}"/>')
        out.append(f'<text class="ax" x="{x:.1f}" y="{len(rows)*(bar+gap)+2}" '
                   f'text-anchor="middle">{g}%</text>')
    for i, (name, pct, n) in enumerate(rows):
        y = i * (bar + gap)
        w = plot * pct / 60
        out.append(f'<text class="ax" x="{left-8}" y="{y+bar/2+3.5}" text-anchor="end">{_esc(name)}</text>')
        out.append(f'<rect x="{left}" y="{y}" width="{max(w,1.5):.1f}" height="{bar}" rx="3" '
                   f'fill="var(--s2)"><title>{_esc(name)} · {pct:.1f}% of {n:,} readings that named '
                   f'a language named the wrong one</title></rect>')
        out.append(f'<text class="vl" x="{left+w+7:.1f}" y="{y+bar/2+3.5}">{pct:.0f}%</text>')
    out.append("</svg>")
    return '<div class="fig">' + "".join(out) + "</div>"


# ---------------------------------------------------------------- 8. does faithfulness catch it?
def confab_by_faith(q: list[dict]) -> str:
    S, padl, padt, padb = 130, 46, 10, 40
    plot = W - padl - 16
    lo, hi = 0, max(r["wrong_pct"] for r in q) * 1.25
    def X(i): return padl + (i + 0.5) / len(q) * plot
    def Y(v): return padt + (1 - (v - lo) / (hi - lo)) * S
    out = [f'<svg viewBox="0 0 {W} {S+padt+padb}" role="img" '
           f'aria-label="Wrong-language rate by reconstruction faithfulness">']
    for g in (0, 20, 40):
        out.append(f'<line class="gl" x1="{padl}" y1="{Y(g):.1f}" x2="{padl+plot}" y2="{Y(g):.1f}"/>')
        out.append(f'<text class="ax" x="{padl-6}" y="{Y(g)+3.5:.1f}" text-anchor="end">{g}%</text>')
    d = " ".join(f'{"M" if i==0 else "L"}{X(i):.1f},{Y(r["wrong_pct"]):.1f}' for i, r in enumerate(q))
    out.append(f'<path d="{d}" fill="none" stroke="var(--c-ar)" stroke-width="2" '
               f'stroke-linejoin="round"/>')
    for i, r in enumerate(q):
        out.append(f'<circle cx="{X(i):.1f}" cy="{Y(r["wrong_pct"]):.1f}" r="4.5" fill="var(--c-ar)" '
                   f'stroke="var(--figsurface)" stroke-width="2"><title>faithfulness '
                   f'{r["rt_cos_lo"]:.3f}–{r["rt_cos_hi"]:.3f} · {r["wrong_pct"]:.1f}% wrong '
                   f'({r["wrong"]} of {r["n"]})</title></circle>')
        out.append(f'<text class="vl" x="{X(i):.1f}" y="{Y(r["wrong_pct"])-11:.1f}" '
                   f'text-anchor="middle">{r["wrong_pct"]:.0f}%</text>')
        out.append(f'<text class="ax" x="{X(i):.1f}" y="{S+padt+15}" text-anchor="middle">'
                   f'{r["rt_cos_lo"]:.2f}–{r["rt_cos_hi"]:.2f}</text>')
    out.append(f'<text class="ax" x="{padl+plot/2}" y="{S+padt+30}" text-anchor="middle">'
               f'round-trip faithfulness of the reading (quintiles, least → most faithful)</text>')
    out.append("</svg>")
    return '<div class="fig">' + "".join(out) + "</div>"


# ---------------------------------------------------------------- 9. the look-ahead null
def lookahead_null(d: dict) -> str:
    """Two bars and an interval. The whole point is that they are the same height."""
    rows = [("the case's own reading", d["own_mean"], "var(--c-judge)"),
            ("a reading from another case", d["foreign_mean"], "var(--ink-3)")]
    bar, gap, left = 24, 16, 176
    H = len(rows) * (bar + gap) + 4
    plot = W - left - 66
    hi = max(r[1] for r in rows) * 1.35
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" '
           f'aria-label="Words anticipated by a reading, own versus a foreign reading">']
    for i, (lab, v, col) in enumerate(rows):
        y = i * (bar + gap)
        w = plot * v / hi
        out.append(f'<text class="ax" x="{left-8}" y="{y+bar/2+3.5}" text-anchor="end">{_esc(lab)}</text>')
        out.append(f'<rect x="{left}" y="{y}" width="{w:.1f}" height="{bar}" rx="3" fill="{col}">'
                   f'<title>{_esc(lab)} · {v:.3f} words named ≥400 characters ahead</title></rect>')
        out.append(f'<text class="vl" x="{left+w+7:.1f}" y="{y+bar/2+3.5}">{v:.3f}</text>')
    out.append("</svg>")
    # same lesson as the heatmap footer: a long line inside the viewBox gets clipped, so it lives in HTML
    return ('<div class="fig">' + "".join(out) + "</div>"
            + f'<p class="figstat">excess <b>{d["excess"]:+.3f}</b> '
              f'(95% CI {d["ci95"][0]:+.3f} to {d["ci95"][1]:+.3f}) — the interval contains zero</p>')
