# Artifact regression tests

Both extract the **shipped** `cotHTML` out of the built page and run it — they never
re-implement it, so they cannot drift from what actually ships.

    node nla/tests/artifact_overlap_test.mjs   # run from the dir holding nla_results.html
    node nla/tests/artifact_span_test.mjs
    node nla/tests/artifact_stats_test.mjs

- `artifact_overlap_test.mjs` — over all 330 real cases: no reading is dropped, and the
  transcript text is byte-identical to `model_reply` after tags are stripped.
  *(2026-08-07: 330 cases, 2,828 marks, 0 lost, 0 corrupted.)*
- `artifact_span_test.mjs` — synthetic overlapping anchors (5 readings, 6-char spans at
  stride 2) compared against the pre-2026-08-07 algorithm.
  *(2026-08-07: OLD misplaces 4/5 highlights, NEW 0/5.)*

- `artifact_stats_test.mjs` — every printed number must be recomputable from the payload the page
  ships with, and any section covering a **subset** must state an n equal to that subset. Added when
  the N5 dense corpus was merged in, because that failure mode is silent: a stale count still reads
  as correct, and a subset analysis reads as if it covered everything.
  *(2026-08-07: 380 cases / 9,511 readings — banked 5,090 + dense 4,421; faithfulness section
  correctly scoped at 5,090; 0 broken anchors, 0 duplicate positions.)*

The builder lives at `nla/tools/build_page.py`.
