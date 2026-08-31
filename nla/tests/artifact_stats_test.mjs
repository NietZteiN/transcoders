// Every headline number on the page must be recomputable from the payload it ships with.
//
// This exists because the page merged a second corpus (N5 dense) into an existing one, and the
// failure mode there is silent: a count that still reflects the old corpus reads as correct, and
// a section whose analysis covers a SUBSET reads as if it covered everything. Both are wrong in a
// way no rendering check would catch. Run from the directory holding nla_results.html:
//
//     node nla/tests/artifact_stats_test.mjs
import fs from "fs";

const html = fs.readFileSync("nla_results.html", "utf8");
const mk = "window.__NLA__ = ";
const s = html.indexOf(mk) + mk.length;
const D = JSON.parse(html.slice(s, html.indexOf(";</script>", s)));
const C = D.cases, ST = D.stats;

let fail = 0;
const ok = (cond, label, detail = "") => {
  if (!cond) fail++;
  console.log(`  ${cond ? "ok  " : "FAIL"} ${label}${detail ? "  — " + detail : ""}`);
};

// ---- counts the header prints ----
const reads = C.reduce((a, c) => a + c.reads.length, 0);
const bySrc = {};
for (const c of C) for (const r of c.reads) bySrc[r.src || "banked"] = (bySrc[r.src || "banked"] || 0) + 1;
console.log("counts");
ok(ST.cases === C.length, "stats.cases matches the payload", `${ST.cases} vs ${C.length}`);
ok(ST.reads === reads, "stats.reads matches the payload", `${ST.reads} vs ${reads}`);
ok(ST.dense_cases === C.filter(c => c.dense).length, "stats.dense_cases matches",
   `${ST.dense_cases} vs ${C.filter(c => c.dense).length}`);
ok(ST.dense_added === (bySrc.dense || 0), "stats.dense_added matches readings tagged dense",
   `${ST.dense_added} vs ${bySrc.dense || 0}`);
console.log(`       breakdown: ${JSON.stringify(bySrc)}`);

// ---- the faithfulness section covers a SUBSET; its stated n must equal that subset ----
// The page prints FAITH.overall.n. If the dense merge ever changes what "first pass" means, or the
// analysis is regenerated over a different corpus, this catches the drift instead of letting the
// page claim a scope it does not have.
const mN = html.match(/([\d,]+) readings of the first pass/);
console.log("\nfaithfulness section scope");
ok(!!mN, "the section states its own n");
if (mN) {
  const stated = +mN[1].replace(/,/g, "");
  ok(stated === (bySrc.banked || 0), "stated n equals the first-pass readings actually present",
     `${stated} vs ${bySrc.banked || 0}`);
  ok(stated < reads, "the section is scoped BELOW the page total (so the note is warranted)",
     `${stated} < ${reads}`);
}

// ---- per-reading integrity, for both corpora ----
let noText = 0, noScore = 0, badAnchor = 0, dupPos = 0;
for (const c of C) {
  const seen = new Set();
  for (const r of c.reads) {
    if (!r.read || !r.read.trim()) noText++;
    if (typeof r.rt_cos !== "number") noScore++;
    const a = r.anchor || {};
    // an in-reply anchor drives BOTH the transcript highlight and the token-context strip
    if (a.in_reply && (a.tok == null || a.before == null || a.after == null
                       || !Number.isFinite(a.rs) || !Number.isFinite(a.re) || a.re <= a.rs)) badAnchor++;
    if (r.position != null) { if (seen.has(r.position)) dupPos++; seen.add(r.position); }
  }
}
console.log("\nper-reading integrity");
ok(noText === 0, "every reading has text", `${noText} empty`);
ok(noScore === 0, "every reading has a round-trip score", `${noScore} missing`);
ok(badAnchor === 0, "every in-reply anchor is complete and well-ordered", `${badAnchor} broken`);
ok(dupPos === 0, "no case reads the same token position twice", `${dupPos} duplicates`);

// ---- the dense merge must not have disturbed the first pass ----
const denseCases = C.filter(c => c.dense);
const perCase = denseCases.map(c => c.reads.length).sort((a, b) => a - b);
console.log("\ndense merge");
ok(denseCases.every(c => c.reads.some(r => (r.src || "banked") === "banked")),
   "every dense case still carries its first-pass readings");
ok(denseCases.every(c => c.reads.length >= 10), "dense cases are actually denser",
   `min ${perCase[0]}, median ${perCase[Math.floor(perCase.length / 2)]}, max ${perCase[perCase.length - 1]}`);

console.log(fail === 0 ? "\nPASS — every printed number is recomputable from the shipped payload"
                       : `\nFAIL — ${fail} check(s) failed`);
process.exit(fail === 0 ? 0 : 1);
