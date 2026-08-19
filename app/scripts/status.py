#!/usr/bin/env python3
import csv
import json
import os
from pathlib import Path
import soundfile as sf

model = os.getenv("TTS_MODEL", "turbo").lower()
corpus = Path("corpus.txt")
meta = Path("dataset_raw") / model / "metadata.csv"

corpus_n = len([x for x in corpus.read_text(encoding="utf-8").splitlines() if x.strip()]) if corpus.exists() else 0
accepted = []
seconds = 0.0
if meta.exists():
    with meta.open("r", encoding="utf-8", newline="") as f:
        accepted = [r for r in csv.reader(f, delimiter="|") if len(r) >= 2]
    for r in accepted:
        p = Path("dataset_raw") / model / "wavs" / r[0]
        if p.exists():
            info = sf.info(p)
            seconds += info.frames / info.samplerate

attempts = rejects = 0
log = Path("logs") / f"generation-{model}.jsonl"
if log.exists():
    for line in log.read_text(encoding="utf-8", errors="ignore").splitlines():
        try:
            obj = json.loads(line)
            attempts += 1
            if not obj.get("accepted"):
                rejects += 1
        except Exception:
            pass

print(f"Corpus phrases : {corpus_n:,}")
print(f"Accepted clips : {len(accepted):,}")
print(f"Remaining      : {max(0, corpus_n-len(accepted)):,}")
print(f"Audio hours    : {seconds/3600:.2f}")
print(f"Attempts logged: {attempts:,}")
print(f"Rejected tries : {rejects:,}")
