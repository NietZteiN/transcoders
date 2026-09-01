#!/usr/bin/env bash
# Concatenate draw-shards into the draws.jsonl every scorer already reads.
# Shards hold disjoint draw ranges, so a plain append is correct and idempotent-by-rebuild:
# draws.jsonl is rebuilt from shard 0 plus the rest each time rather than appended to twice.
set -euo pipefail
ROOT="${1:?usage: merge_ladder_shards.sh <ladder-dir>}"
for d in "$ROOT"/*/; do
  t=$(basename "$d")
  shards=("$d"draws_d*.jsonl)
  [ -e "${shards[0]}" ] || { echo "$t: no shards, nothing to merge"; continue; }
  [ -f "$d/draws.jsonl.base" ] || cp "$d/draws.jsonl" "$d/draws.jsonl.base" 2>/dev/null || true
  cat "$d/draws.jsonl.base" "${shards[@]}" > "$d/draws.jsonl.tmp" 2>/dev/null \
    || cat "${shards[@]}" > "$d/draws.jsonl.tmp"
  mv "$d/draws.jsonl.tmp" "$d/draws.jsonl"
  n=$(wc -l < "$d/draws.jsonl")
  echo "$t: merged $(( ${#shards[@]} )) shard(s) -> $n rows"
done
