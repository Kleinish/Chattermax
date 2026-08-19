#!/usr/bin/env bash
set -euo pipefail

VOICE_NAME="${VOICE_NAME:-my_voice}"
CKPT="${1:-}"

if [[ -z "$CKPT" ]]; then
  CKPT=$(find /workspace/training/runs -type f -name '*.ckpt' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2- || true)
fi

if [[ -z "$CKPT" || ! -f "$CKPT" ]]; then
  echo "No checkpoint found. Pass one explicitly: scripts/export.sh /workspace/training/.../epoch=...ckpt" >&2
  exit 1
fi

mkdir -p /workspace/output
OUT="/workspace/output/${VOICE_NAME}.onnx"
CONFIG="/workspace/training/config/${VOICE_NAME}.onnx.json"

python3 -m piper.train.export_onnx \
  --checkpoint "$CKPT" \
  --output-file "$OUT"

if [[ ! -f "$CONFIG" ]]; then
  echo "Training config missing: $CONFIG" >&2
  exit 1
fi
cp "$CONFIG" "${OUT}.json"

echo "Exported:"
echo "  $OUT"
echo "  ${OUT}.json"
