#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio as ta
import whisper
from rapidfuzz.fuzz import ratio
from tqdm import tqdm
from chatterbox.tts import ChatterboxTTS
from chatterbox.tts_turbo import ChatterboxTurboTTS

def env_float(name, default):
    return float(os.getenv(name, str(default)))


def env_int(name, default):
    return int(os.getenv(name, str(default)))


def norm_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def audio_metrics(path: Path):
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    duration = len(audio) / float(sr)
    abs_audio = np.abs(audio)
    clipped_fraction = float(np.mean(abs_audio >= 0.999)) if len(audio) else 1.0
    # Conservative digital-silence proxy. Synthetic TTS normally remains well above this while speaking.
    silence_fraction = float(np.mean(abs_audio < 0.004)) if len(audio) else 1.0
    peak = float(abs_audio.max()) if len(audio) else 0.0
    return {
        "duration": duration,
        "clipped_fraction": clipped_fraction,
        "silence_fraction": silence_fraction,
        "peak": peak,
        "sample_rate": sr,
    }


def load_existing(metadata_path: Path):
    accepted = {}
    if not metadata_path.exists():
        return accepted
    with metadata_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.reader(f, delimiter="|"):
            if len(row) >= 2:
                accepted[row[0]] = row[1]
    return accepted


def append_metadata(metadata_path: Path, filename: str, text: str):
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("a", encoding="utf-8", newline="") as f:
        csv.writer(f, delimiter="|", lineterminator="\n").writerow([filename, text])
        f.flush()
        os.fsync(f.fileno())


def append_jsonl(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()


def main():
    ap = argparse.ArgumentParser(description="Resumable Chatterbox dataset generator with Whisper QA")
    ap.add_argument("--corpus", default="corpus.txt")
    ap.add_argument("--reference", default=os.getenv("REFERENCE_AUDIO", "input/reference.wav"))
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="0 = no limit")
    ap.add_argument("--force", action="store_true", help="Regenerate even if already accepted")
    args = ap.parse_args()

    corpus_path = Path(args.corpus)
    reference = Path(args.reference)
    if not corpus_path.exists():
        raise SystemExit(f"Missing corpus: {corpus_path}")
    if not reference.exists():
        raise SystemExit(f"Missing reference audio: {reference}")

    lines = [x.strip() for x in corpus_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    indexed = list(enumerate(lines))[args.start:]
    if args.limit:
        indexed = indexed[:args.limit]

    raw_dir = Path("dataset_raw/wavs")
    rejected_dir = Path("rejected")
    tmp_dir = Path("dataset_raw/.tmp")
    metadata = Path("dataset_raw/metadata.csv")
    audit = Path("logs/generation.jsonl")
    raw_dir.mkdir(parents=True, exist_ok=True)
    rejected_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    existing = load_existing(metadata)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    whisper_device = os.getenv("WHISPER_DEVICE", device)
    if whisper_device == "cuda" and not torch.cuda.is_available():
        whisper_device = "cpu"

    print(f"Chatterbox device: {device}")
    print(f"Whisper device: {whisper_device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    tts_model = os.getenv("TTS_MODEL", "original").strip().lower()

    print(f"TTS model: {tts_model}")

    if tts_model == "original":
        model = ChatterboxTTS.from_pretrained(device=device)
    elif tts_model == "turbo":
        model = ChatterboxTurboTTS.from_pretrained(device=device)
    else:
        raise SystemExit(
            f"Invalid TTS_MODEL={tts_model!r}. Expected 'original' or 'turbo'."
        )
    asr = whisper.load_model(os.getenv("WHISPER_MODEL", "medium.en"), device=whisper_device)

    min_score = env_float("MIN_ASR_SCORE", 92)
    min_duration = env_float("MIN_DURATION_SEC", 0.60)
    max_duration = env_float("MAX_DURATION_SEC", 18.0)
    max_silence = env_float("MAX_SILENCE_FRACTION", 0.35)
    max_clip = env_float("MAX_CLIPPED_FRACTION", 0.0005)
    max_attempts = env_int("MAX_GENERATION_ATTEMPTS", 4)
    exaggeration = env_float("CHATTERBOX_EXAGGERATION", 0.20)
    cfg_weight = env_float("CHATTERBOX_CFG_WEIGHT", 0.50)

    accepted_count = rejected_count = skipped_count = 0

    for i, text in tqdm(indexed, desc="Generating"):
        filename = f"utt_{i:06d}.wav"
        if filename in existing and not args.force:
            skipped_count += 1
            continue

        ok = False
        for attempt in range(1, max_attempts + 1):
            temp = tmp_dir / f"{i:06d}_{attempt}.wav"
            try:
                wav = model.generate(
                    text,
                    audio_prompt_path=str(reference),
                    exaggeration=exaggeration,
                    cfg_weight=cfg_weight,
                )
                ta.save(str(temp), wav.cpu(), model.sr)
                metrics = audio_metrics(temp)
                result = asr.transcribe(str(temp), language="en", fp16=(whisper_device == "cuda"))
                heard = result.get("text", "").strip()
                score = ratio(norm_text(text), norm_text(heard))

                reasons = []
                if score < min_score:
                    reasons.append(f"asr_score={score:.1f}<{min_score}")
                if metrics["duration"] < min_duration:
                    reasons.append("too_short")
                if metrics["duration"] > max_duration:
                    reasons.append("too_long")
                if metrics["silence_fraction"] > max_silence:
                    reasons.append("too_much_silence")
                if metrics["clipped_fraction"] > max_clip:
                    reasons.append("clipping")

                record = {
                    "id": i,
                    "filename": filename,
                    "attempt": attempt,
                    "text": text,
                    "asr": heard,
                    "asr_score": score,
                    **metrics,
                    "accepted": not reasons,
                    "reasons": reasons,
                }
                append_jsonl(audit, record)

                if not reasons:
                    dest = raw_dir / filename
                    shutil.move(str(temp), dest)
                    if filename not in existing:
                        append_metadata(metadata, filename, text)
                        existing[filename] = text
                    ok = True
                    accepted_count += 1
                    break

                reject_path = rejected_dir / f"utt_{i:06d}_attempt{attempt}.wav"
                shutil.move(str(temp), reject_path)
            except Exception as exc:
                append_jsonl(audit, {
                    "id": i, "filename": filename, "attempt": attempt,
                    "text": text, "accepted": False, "reasons": [f"exception: {exc!r}"]
                })
                if temp.exists():
                    shutil.move(str(temp), rejected_dir / f"utt_{i:06d}_attempt{attempt}_error.wav")

        if not ok:
            rejected_count += 1

    print("\nGeneration summary")
    print(f"  Accepted this run: {accepted_count}")
    print(f"  Already accepted:  {skipped_count}")
    print(f"  Failed all tries:   {rejected_count}")
    print(f"  Total accepted:     {len(existing)}")
    print(f"  Metadata:           {metadata}")
    print(f"  Audit log:          {audit}")


if __name__ == "__main__":
    main()
