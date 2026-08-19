#!/usr/bin/env bash
set -euo pipefail

VOICE_NAME="${VOICE_NAME:-my_voice}"
ESPEAK_VOICE="${ESPEAK_VOICE:-en-us}"
SAMPLE_RATE="${SAMPLE_RATE:-22050}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-4}"
VALIDATION_SPLIT="${VALIDATION_SPLIT:-0.03}"
NUM_TEST_EXAMPLES="${NUM_TEST_EXAMPLES:-100}"
MAX_EPOCHS="${MAX_EPOCHS:-2000}"
PRECISION="${PRECISION:-32-true}"

mkdir -p training/cache training/config training/runs

if [[ ! -f dataset/metadata.csv ]]; then
  echo "dataset/metadata.csv is missing. Run prepare first." >&2
  exit 1
fi

# TRUE FROM-SCRATCH TRAINING:
# Intentionally NO --ckpt_path and NO --model.vocoder_warmstart_ckpt.
python3 -m piper.train fit \
  --data.voice_name "$VOICE_NAME" \
  --data.csv_path /workspace/dataset/metadata.csv \
  --data.audio_dir /workspace/dataset/wavs \
  --model.sample_rate "$SAMPLE_RATE" \
  --data.espeak_voice "$ESPEAK_VOICE" \
  --data.cache_dir /workspace/training/cache \
  --data.config_path /workspace/training/config/${VOICE_NAME}.onnx.json \
  --data.batch_size "$BATCH_SIZE" \
  --data.validation_split "$VALIDATION_SPLIT" \
  --data.num_test_examples "$NUM_TEST_EXAMPLES" \
  --data.num_workers "$NUM_WORKERS" \
  --data.trim_silence true \
  --trainer.accelerator gpu \
  --trainer.devices 1 \
  --trainer.max_epochs "$MAX_EPOCHS" \
  --trainer.precision "$PRECISION" \
  --trainer.default_root_dir /workspace/training/runs \
  --trainer.enable_checkpointing true \
  --trainer.log_every_n_steps 20
