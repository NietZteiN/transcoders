#!/usr/bin/env bash
# Smoke-test the activation-extraction harness on a tiny model + toy stimuli (../CLAUDE.md §4).
# CPU by default — validates plumbing INCLUDING the identifier-span→token resolution path
# (config merge, seed, forward, span resolution, cache, manifests, provenance) without
# downloading an 8B model or touching a GPU. Run from anywhere:
#   bash scripts/smoke.sh
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="${1:-configs/experiments/e1_semantic_capture.yaml}"

echo "== transcoders smoke =="
echo "config: $CONFIG"
python -m src.extract_activations --config "$CONFIG" --smoke

# Newest *_smoke* run dir, robust under set -e when the glob is empty (nullglob, no ls-pipe).
shopt -s nullglob
runs=(data/activations/*_smoke*/)
RUN_DIR=""
if ((${#runs[@]})); then
  RUN_DIR=$(ls -dt -- "${runs[@]}" | head -1)
fi

echo
echo "latest smoke run: ${RUN_DIR:-<none>}"
if [[ -n "$RUN_DIR" ]]; then
  echo "-- manifest.jsonl (first line) --"; head -1 "$RUN_DIR/manifest.jsonl"
  echo "-- run_manifest.json: finished + resolution rate --"
  python -c "import json;m=json.load(open('$RUN_DIR/run_manifest.json'));print('finished:',m['finished_utc'],'| mean_resolution_rate:',m['extra'].get('mean_resolution_rate'))"
fi
echo "OK"
