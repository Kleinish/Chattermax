#!/usr/bin/env bash
set -euo pipefail

APP_ROOT="${APP_ROOT:-/app}"
DATA_ROOT="${DATA_ROOT:-/data}"

mkdir -p \
  "$DATA_ROOT/input" \
  "$DATA_ROOT/corpus_sources" \
  "$DATA_ROOT/corpus_cache" \
  "$DATA_ROOT/dataset_raw" \
  "$DATA_ROOT/dataset" \
  "$DATA_ROOT/rejected" \
  "$DATA_ROOT/training" \
  "$DATA_ROOT/output" \
  "$DATA_ROOT/logs" \
  "$DATA_ROOT/reports" \
  "$DATA_ROOT/review"

for name in \
  input corpus_sources corpus_cache dataset_raw dataset rejected \
  training output logs reports review
do
  rm -rf "$APP_ROOT/$name"
  ln -s "$DATA_ROOT/$name" "$APP_ROOT/$name"
done

rm -f "$APP_ROOT/corpus.txt"
ln -s "$DATA_ROOT/corpus.txt" "$APP_ROOT/corpus.txt"

# Compatibility for scripts that still use the historical /workspace path.
rm -rf /workspace
ln -s "$APP_ROOT" /workspace

cd "$APP_ROOT"
exec "$@"
