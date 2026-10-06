#!/usr/bin/env sh
# Memory sweep (benchmark-plan.md §4): trace decode speed from fully resident
# to mostly streamed, with and without prefetch.
#
#   benchmarks/run_memory_sweep.sh model.gguf [runs] [tokens]
#
# Writes benchmarks/results/<model>-sweep-<fraction>.json per fraction.
set -e
model="$1"; runs="${2:-3}"; tokens="${3:-32}"
[ -n "$model" ] || { echo "usage: $0 model.gguf [runs] [tokens]"; exit 2; }
kestrel="${KESTREL:-$(dirname "$0")/../target/release/kestrel}"
name="$(basename "$model" .gguf)"
out="$(dirname "$0")/results"
mkdir -p "$out"
for f in 0.25 0.5 0.75 0.9 1.0; do
  echo "== stream fraction $f"
  "$kestrel" benchmark "$model" --arms kestrel,no-prefetch --stream-fraction "$f" \
    --runs "$runs" --tokens "$tokens" --json "$out/$name-sweep-$f.json"
done
