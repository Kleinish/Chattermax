# ChatterMax

ChatterMax is a Docker-based workflow for generating a synthetic speech dataset with Chatterbox, validating it with Whisper, preparing it for Piper, and training/exporting a Piper voice.

The repository keeps application source in Git and keeps generated/runtime data in persistent Docker-mounted storage.

## Repository layout

```text
Chattermax/
├── app/
│   ├── scripts/
│   └── web/
├── assets/
│   └── chattypiper.png
├── config/
│   └── corpus/
│       └── custom.txt
├── docker/
│   ├── Dockerfile.generator
│   ├── Dockerfile.trainer
│   ├── Dockerfile.speech-isolator
│   └── entrypoint.sh
├── speech-isolator/
│   └── app/
│       ├── app.py
│       └── requirements.txt
├── compose.yaml
├── .env.example
├── .gitignore
├── .dockerignore
├── Makefile
└── README.md
```

Runtime folders are not stored in Git.

Default host storage:

```text
data/
├── input/
├── corpus_sources/
├── corpus_cache/
├── dataset_raw/
├── dataset/
├── rejected/
├── training/
├── output/
├── logs/
├── reports/
├── review/
└── speech-isolator/
    └── cache/
```

Inside the generator and trainer containers:

```text
/app     application source
/data    persistent runtime data
```

The entrypoint also provides `/workspace` as a compatibility link for older scripts.

## Speech Isolator → Generator flow

The Speech Isolator output and the generator input intentionally use the **same host folder**:

```text
data/input/
```

The generator sees it as:

```text
/data/input
```

The Speech Isolator writes cleaned results to:

```text
/data/output
```

but Compose maps `/data/output` to the same host `INPUT_MOUNT`.

This means cleaned reference audio is immediately available in the generator input storage without maintaining a separate speech-isolator output share.

The Speech Isolator model cache remains separate:

```text
data/speech-isolator/cache/
```

## Setup

Copy the environment template:

```bash
cp .env.example .env
```

PowerShell:

```powershell
Copy-Item .env.example .env
```

Validate Compose:

```bash
docker compose config
```

Start:

```bash
docker compose up -d --build
```

Open:

```text
http://localhost:8080
```

## Storage recommendations

| Folder | Recommended storage | Notes |
|---|---|---|
| `input` | Any reliable disk / SSD preferred | Reference and cleaned reference audio; small |
| `corpus_sources` | HDD/array is fine | Small text files |
| `corpus_cache` | SSD preferred | Downloads/clones; HDD works |
| `dataset_raw` | SSD / NVMe | Heavy WAV creation/read activity |
| `dataset` | SSD / NVMe | Repeatedly read during training |
| `training` | SSD / NVMe | Checkpoints, cache, active training I/O |
| `rejected` | HDD / array | Can become large but not latency-sensitive |
| `output` | HDD / array / share | Final exported models |
| `logs` | HDD / array | Low throughput |
| `reports` | HDD / array | Small/low throughput |
| Speech Isolator cache | SSD preferred | Model/cache activity |

For active training, avoid SMB/NFS for `dataset_raw`, `dataset`, and `training` when local SSD/NVMe is available.

## Split-storage example

```env
INPUT_MOUNT=/fast/chattermax/input
DATASET_RAW_MOUNT=/fast/chattermax/dataset_raw
DATASET_MOUNT=/fast/chattermax/dataset
TRAINING_MOUNT=/fast/chattermax/training

REJECTED_MOUNT=/bulk/chattermax/rejected
OUTPUT_MOUNT=/bulk/chattermax/output
LOGS_MOUNT=/bulk/chattermax/logs
REPORTS_MOUNT=/bulk/chattermax/reports
```

## Unraid / Compose Manager Plus

Use `compose.yaml` and `.env`. This layout does not use `unraid-template.xml`.

Example:

```env
INPUT_MOUNT=/mnt/cache/chattermax/input
DATASET_RAW_MOUNT=/mnt/cache/chattermax/dataset_raw
DATASET_MOUNT=/mnt/cache/chattermax/dataset
TRAINING_MOUNT=/mnt/cache/chattermax/training

REJECTED_MOUNT=/mnt/user/chattermax/rejected
OUTPUT_MOUNT=/mnt/user/chattermax/output
LOGS_MOUNT=/mnt/user/appdata/chattermax/logs
REPORTS_MOUNT=/mnt/user/chattermax/reports

SPEECH_CACHE_MOUNT=/mnt/cache/chattermax/speech-isolator-cache
```

Replace `/mnt/cache` with the actual pool path used by your Unraid system.

## Common commands

```bash
docker compose build
docker compose up -d --build
docker compose run --rm corpus
docker compose run --rm generator python scripts/generate.py --limit 100
docker compose run --rm generator python scripts/run_generation.py
docker compose run --rm generator python scripts/status.py
docker compose run --rm generator python scripts/validate.py
docker compose run --rm generator python scripts/prepare_dataset.py
docker compose run --rm trainer bash scripts/train.sh
docker compose run --rm trainer bash scripts/export.sh
```
