// Runs the SHIPPED cotHTML (extracted verbatim from the built page) over every real case and
// asserts that no read is lost to overlap collapse -- the bug F2 was meant to fix.
import fs from "fs";
const html = fs.readFileSync("nla_results.html", "utf8");

// 1. the data payload
const mk = "window.__NLA__ = ";
const s = html.indexOf(mk) + mk.length;
const e = html.indexOf(";</script>", s);
const DATA = JSON.parse(html.slice(s, e));

// 2. the shipped functions, taken straight out of the page (no re-implementation)
const jsAll = html.slice(html.lastIndexOf("<script>", html.indexOf("const DATA = window.__NLA__")));
const grab = name => {
  const i = jsAll.indexOf("function " + name + "(");
  let d = 0, j = jsAll.indexOf("{", i);
  for (let k = j; k < jsAll.length; k++){
    if (jsAll[k] === "{") d++;
    else if (jsAll[k] === "}"){ d--; if (!d) return jsAll.slice(i, k+1); }
  }
};
const esc = s => (s??"").replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const MARK_CAP = 120;
const cotHTML = eval("(" + grab("cotHTML") + ")");

let cases=0, lost=0, multi=0, marks=0, capped=0, textBad=0;
for (const c of DATA.cases){
  const want = c.reads.map((r,i)=>({i,a:r.anchor}))
    .filter(x=>x.a && x.a.in_reply && Number.isFinite(x.a.rs) && Number.isFinite(x.a.re)
               && x.a.re > x.a.rs && x.a.rs < (c.model_reply||"").length)
    .map(x=>x.i);
  if (!want.length) continue;
  cases++;
  const out = cotHTML(c);
  const got = new Set();
  for (const m of out.matchAll(/data-r="([\d,]+)"/g)) m[1].split(",").forEach(n=>got.add(+n));
  for (const m of out.matchAll(/data-r="([\d,]+)"/g)){ marks++; if (m[1].includes(",")) multi++; }
  if (want.length > MARK_CAP){ capped++; continue; }        // thinning is intentional above the cap
  const missing = want.filter(i => !got.has(i));
  if (missing.length){ lost++; if (lost<=3) console.log(`  LOST ${c.task_key}: reads ${missing}`); }
  // the visible text must still equal the reply exactly (no dropped or duplicated characters)
  const plain = out.replace(/<[^>]+>/g,"")
    .replace(/&amp;/g,"&").replace(/&lt;/g,"<").replace(/&gt;/g,">").replace(/&quot;/g,'"');
  if (plain !== (c.model_reply||"")) { textBad++; if(textBad<=2) console.log("  TEXT MISMATCH", c.task_key); }
}
console.log(`cases with in-reply reads : ${cases}`);
console.log(`cases over the ${MARK_CAP}-mark cap  : ${capped} (thinned by design, excluded)`);
console.log(`marks emitted             : ${marks}  (${multi} carry >1 reading)`);
console.log(`cases LOSING a reading    : ${lost}`);
console.log(`cases with text corruption: ${textBad}`);
console.log(lost===0 && textBad===0 ? "\nPASS — no reading lost to overlap, transcript text byte-exact"
                                    : "\nFAIL");
