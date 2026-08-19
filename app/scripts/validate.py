#!/usr/bin/env python3
import argparse
import csv
import json
import random
import os
from pathlib import Path

import numpy as np
import soundfile as sf


def metrics(path: Path):
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if not len(audio):
        return 0.0, 1.0, 1.0
    a = np.abs(audio)
    return len(audio) / sr, float(np.mean(a < 0.004)), float(np.mean(a >= 0.999))


def main():
    ap = argparse.ArgumentParser()
    model = os.getenv("TTS_MODEL", "turbo").lower()
    ap.add_argument("--metadata", default=f"dataset_raw/{model}/metadata.csv")
    ap.add_argument("--audio-dir", default=f"dataset_raw/{model}/wavs")
    ap.add_argument("--sample", type=int, default=50, help="Create manual-review list of N random accepted files")
    args = ap.parse_args()

    metadata = Path(args.metadata)
    audio_dir = Path(args.audio_dir)
    if not metadata.exists():
        raise SystemExit(f"Missing {metadata}")

    rows = []
    missing = []
    durations = []
    with metadata.open("r", encoding="utf-8", newline="") as f:
        for row in csv.reader(f, delimiter="|"):
            if len(row) < 2:
                continue
            wav = audio_dir / row[0]
            if not wav.exists():
                missing.append(row[0])
                continue
            duration, silence, clipped = metrics(wav)
            durations.append(duration)
            rows.append((row[0], row[1], duration, silence, clipped))

    hours = sum(durations) / 3600
    print(f"Accepted clips: {len(rows)}")
    print(f"Missing clips:  {len(missing)}")
    print(f"Audio hours:    {hours:.2f}")
    if durations:
        print(f"Duration sec:   min={min(durations):.2f} avg={np.mean(durations):.2f} max={max(durations):.2f}")

    report = {
        "accepted_clips": len(rows),
        "missing_clips": missing,
        "hours": hours,
        "duration_min": min(durations) if durations else None,
        "duration_mean": float(np.mean(durations)) if durations else None,
        "duration_max": max(durations) if durations else None,
    }
    Path("logs").mkdir(exist_ok=True)
    Path(f"logs/validation_summary_{model}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    if rows and args.sample:
        picked = random.Random(1337).sample(rows, min(args.sample, len(rows)))
        out = Path(f"logs/manual_review_{model}.csv")
        with out.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["file", "text", "duration", "silence_fraction", "clipped_fraction", "rating", "notes"])
            for row in picked:
                w.writerow(row + ("", ""))
        print(f"Manual review sheet: {out}")


if __name__ == "__main__":
    main()
