# Piper From-Scratch Voice Pipeline

Docker Compose pipeline for Windows + Docker Desktop + WSL2:

1. Assemble a high-coverage text corpus.
2. Clone a permitted reference voice with Chatterbox.
3. Validate every generated clip with Whisper plus audio checks.
4. Standardize accepted audio to 22.05 kHz mono PCM WAV.
5. Train the current OHF-Voice Piper model from random initialization.
6. Export the resulting voice to ONNX.

## Important

Only clone a voice you have permission to use.

This project is configured for **true from-scratch Piper training**. `scripts/train.sh` intentionally provides neither `--ckpt_path` nor `--model.vocoder_warmstart_ckpt`.

## Folder layout

```text
.
├── compose.yaml
├── .env
├── input/
│   └── reference.wav
├── corpus_sources/
│   ├── source1.txt
│   └── source2.txt
├── corpus.txt                 # generated
├── dataset_raw/
│   ├── metadata.csv           # generated/accepted clips
│   └── wavs/
├── rejected/                  # failed Chatterbox attempts
├── dataset/
│   ├── metadata.csv           # Piper-ready
│   └── wavs/                  # 22050 Hz mono PCM16
├── training/
├── output/
└── scripts/
```

## 1. Configure

```bash
cp .env.example .env
```

Edit `.env`. Start with `BATCH_SIZE=16`; lower it if CUDA runs out of VRAM.

Put the clean voice reference at:

```text
input/reference.wav
```

A clean 5-20 second single-speaker clip is preferred.

## 2. Build the corpus automatically

Build the generator image, then run the dedicated corpus service:

```bash
docker compose build corpus
docker compose run --rm corpus
```

Or simply:

```bash
make corpus
```

This pipeline:

1. Downloads LJSpeech normalized transcripts.
2. Downloads CMU ARCTIC prompts.
3. Downloads the 720 Harvard sentences.
4. Clones Piper Recording Studio and extracts English prompts.
5. Generates `numbers.txt` with numbers, dates, time, percentages, units, IP-style addresses, and technical phrases.
6. Generates a conversational prompt set.
7. Preserves `corpus_sources/custom.txt` for your own application-specific phrases.
8. Normalizes and deduplicates all source text.
9. Shuffles/selects up to `CORPUS_TARGET` phrases.
10. Uses eSpeak NG (`en-us`) to calculate phoneme/diphone coverage reports.

Generated files include:

```text
corpus_sources/
├── ljspeech.txt
├── arctic.txt
├── harvard.txt
├── piper.txt
├── numbers.txt
├── conversational.txt
└── custom.txt
corpus.txt
reports/phoneme_coverage.json
reports/phoneme_counts.csv
```

Edit `corpus_sources/custom.txt` at any time and rerun `docker compose run --rm corpus`; the combined corpus is rebuilt and deduplicated. Lines beginning with `#` are ignored.

The downloader caches the Piper Recording Studio clone in `corpus_cache/`. Public remote sources can change, so source downloads are isolated and report warnings rather than deleting previously generated files.

## 3. Build containers

```bash
docker compose build
```

Test GPU passthrough:

```bash
docker compose run --rm trainer nvidia-smi
```

## 4. Generate synthetic training speech

Small smoke test first:

```bash
docker compose run --rm generator \
  python scripts/generate.py --limit 10
```

Then run the full corpus:

```bash
docker compose run --rm generator python scripts/generate.py
```

Generation is resumable. Accepted files already listed in `dataset_raw/metadata.csv` are skipped on later runs.

The default QA gates are configurable in `.env`:

- Whisper transcript similarity >= `MIN_ASR_SCORE`
- duration boundaries
- silence fraction
- clipping fraction
- up to `MAX_GENERATION_ATTEMPTS` attempts per phrase

Failed attempts are retained under `rejected/`; each attempt is written to `logs/generation.jsonl`.

Check progress:

```bash
docker compose run --rm generator python scripts/status.py
```

## 5. Validate and manually spot-check

```bash
docker compose run --rm generator python scripts/validate.py
```

This writes `logs/manual_review.csv`. Listen to the sampled clips and rate voice identity/prosody manually. ASR correctness alone does not guarantee that Chatterbox preserved the target voice consistently.

## 6. Prepare Piper dataset

```bash
docker compose run --rm generator python scripts/prepare_dataset.py
```

This converts accepted clips to mono 22,050 Hz PCM16 WAV files and writes `dataset/metadata.csv` in Piper's `filename.wav|text` format.

## 7. Train Piper from scratch

```bash
docker compose run --rm trainer bash scripts/train.sh
```

This is intentionally from scratch:

```text
NO --ckpt_path
NO --model.vocoder_warmstart_ckpt
```

Training output/checkpoints go under `training/runs/`.

For GPU OOM, lower `BATCH_SIZE` in `.env`. For a first end-to-end test, temporarily set `MAX_EPOCHS=2`, then restore the desired value before the real run.

## 8. Export ONNX

Automatically use the newest checkpoint:

```bash
docker compose run --rm trainer bash scripts/export.sh
```

Or pass a specific checkpoint:

```bash
docker compose run --rm trainer \
  bash scripts/export.sh /workspace/training/runs/.../checkpoints/epoch=....ckpt
```

Outputs:

```text
output/my_voice.onnx
output/my_voice.onnx.json
```

## Windows / WSL notes

Run these commands inside WSL. Docker Desktop must have WSL integration enabled for your distro and be using Linux containers. Keep shell scripts with LF line endings.

The project can live under `/mnt/d/...`, but heavy dataset/training I/O is generally faster in the WSL ext4 filesystem (for example `~/piper-from-scratch`). If you keep it on `D:`, expect some overhead when reading/writing many thousands of small WAV/cache files.

## Configurable data/share mounts

Every persistent data folder can be placed on a different host path or mounted share without editing `compose.yaml`. Configure the paths in `.env`:

```dotenv
INPUT_MOUNT=./input
CORPUS_SOURCES_MOUNT=./corpus_sources
CORPUS_CACHE_MOUNT=./corpus_cache
DATASET_RAW_MOUNT=./dataset_raw
REJECTED_MOUNT=./rejected
DATASET_MOUNT=./dataset
TRAINING_MOUNT=./training
OUTPUT_MOUNT=./output
LOGS_MOUNT=./logs
REPORTS_MOUNT=./reports
SPEECH_INPUT_MOUNT=./speech-isolator/input
SPEECH_OUTPUT_MOUNT=./speech-isolator/output
SPEECH_CACHE_MOUNT=./speech-isolator/cache
```

The default `./...` values behave exactly like the original project. To move selected folders, change only those variables. For example, when running Docker Compose from Windows PowerShell:

```dotenv
INPUT_MOUNT=D:/ChattyPiper/reference
DATASET_RAW_MOUNT=E:/ChattyPiper/dataset_raw
TRAINING_MOUNT=F:/PiperTraining
OUTPUT_MOUNT=Z:/PiperVoices
```

When running Compose from WSL, use paths visible to WSL instead:

```dotenv
INPUT_MOUNT=/mnt/d/ChattyPiper/reference
DATASET_RAW_MOUNT=/mnt/e/ChattyPiper/dataset_raw
TRAINING_MOUNT=/mnt/f/PiperTraining
OUTPUT_MOUNT=/mnt/z/PiperVoices
```

You can mix local disks and mounted network shares. The source directories must be accessible to Docker Desktop and should exist before starting the stack. The paths are mounted at the same internal container locations (`/workspace/input`, `/workspace/dataset_raw`, etc.), so the Python scripts and Web UI do not need to know where the host data actually lives.

After changing mounts, recreate the containers:

```bash
docker compose down
docker compose up -d --build
```

Use this to verify the resolved mount configuration before starting:

```bash
docker compose config
```

## Web UI

The project includes a lightweight FastAPI/Jinja web interface. It intentionally
uses the same files and scripts as the CLI workflow; there is no separate
application database.

Start it with:

```bash
docker compose up -d --build webui
```

Open `http://localhost:8080` (or change `WEBUI_PORT` in `.env`). The dashboard
supports:

- Start/resume the entire corpus with automatic fresh workers every
  `GENERATION_CHUNK_SIZE` prompts (default 2000).
- Retry crashed worker chunks and resume from persistent supervisor state.
- Stop a running web-launched job.
- Live generation/job logs in the browser.
- Build/refresh the corpus, validate accepted audio, and prepare the Piper
  dataset.
- Upload/replace the Chatterbox reference audio.
- Clean reference audio with DeepFilterNet directly from the same dashboard, preview original vs. cleaned audio, and choose which version Chatterbox uses.
- Change common generation/validation settings in `.env`.
- Review rejected attempts with an HTML audio player and manually promote good
  samples into `dataset_raw/<model>/wavs` + metadata.

The low-level CLI remains available:

```bash
docker compose run --rm generator python scripts/generate.py --start 0 --limit 100
docker compose run --rm generator python scripts/run_generation.py
```

### Long-run generation model

`run_generation.py` is the default full-corpus runner. The lightweight
supervisor stays alive while `generate.py` is launched as a fresh child process
for each chunk. When a child exits, Chatterbox, Whisper, and its CUDA context are
released before the next worker starts. Supervisor progress is stored under
`logs/generation-supervisor-<model>.json`.

## Unraid

The generator image is self-contained for Unraid/published-image use. Mount a
persistent host directory at `/workspace`; on first start the image seeds the
application scripts/web UI and creates the runtime directories without
replacing `.env`, datasets, logs, reference audio, or training data.

Recommended mappings:

- Web port: `8080/tcp`
- Persistent workspace: `/mnt/user/appdata/chatterbox-piper` -> `/workspace`
- GPU: NVIDIA GPU exposed to the container (`--gpus=all` or the equivalent
  Unraid NVIDIA configuration)
- Optional variable: `HF_TOKEN`

An `unraid-template.xml` starter is included. Before publishing it, replace the
`YOUR_GITHUB_USERNAME` placeholders with the GitHub owner that publishes the
container image.

The web UI image handles Chatterbox generation and dataset preparation. Piper
training remains in the dedicated `trainer` image so Chatterbox and Piper's
training dependency stacks do not collide.

## Speech Isolator / Reference Audio Cleaner

The Compose stack includes a `speech-isolator` backend service powered by DeepFilterNet. It is intentionally **not exposed as a second web interface**. The main Chatterbox dashboard at `http://localhost:8080` now controls reference cleanup directly.

Start the complete stack:

```bash
docker compose up -d --build
```

In **Reference voice + cleanup**, upload the original reference, choose Light/Medium/Strong cleanup, preview the original and cleaned recordings, and click **Use Original** or **Use Cleaned**. Cleaning automatically selects the cleaned WAV for the next Chatterbox generation job.

Recommended starting settings for a Chatterbox reference recording:

- Noise reduction: **Medium**
- Normalize reference volume: **On**
- Remove low-frequency rumble: **On**

The cleaned reference used by Chatterbox is stored at `input/reference_clean.wav`. DeepFilterNet's persistent model cache and temporary backend outputs remain under `speech-isolator/cache` and `speech-isolator/output`. The backend is reachable only inside the Compose network as `http://speech-isolator:7860`.
