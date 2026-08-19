#!/usr/bin/env python3
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


def read_jsonl(path: Path):
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def append_jsonl(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()


def load_metadata(path: Path):
    rows = {}
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.reader(f, delimiter="|"):
            if len(row) >= 2:
                rows[row[0]] = row[1]
    return rows


def append_metadata(path: Path, filename: str, text: str):
    existing = load_metadata(path)
    if filename in existing:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="") as f:
        csv.writer(f, delimiter="|", lineterminator="\n").writerow([filename, text])
        f.flush()
        os.fsync(f.fileno())
    return True


def load_corpus(path: Path):
    if not path.exists():
        raise SystemExit(f"Missing corpus: {path}")
    return [x.strip() for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def group_attempts(audit_rows, rejected_dir: Path):
    grouped = defaultdict(list)
    for row in audit_rows:
        if row.get("accepted"):
            continue
        utt_id = row.get("id")
        attempt = row.get("attempt")
        if utt_id is None or attempt is None:
            continue

        wav = rejected_dir / f"utt_{int(utt_id):06d}_attempt{int(attempt)}.wav"
        wav_error = rejected_dir / f"utt_{int(utt_id):06d}_attempt{int(attempt)}_error.wav"

        if wav.exists():
            candidate = wav
        elif wav_error.exists():
            candidate = wav_error
        else:
            continue

        item = dict(row)
        item["_wav"] = candidate
        grouped[int(utt_id)].append(item)

    # Best candidate first: accepted-ish audio generally has higher ASR and lower silence.
    for utt_id, items in grouped.items():
        items.sort(
            key=lambda x: (
                float(x.get("asr_score", 0.0)),
                -float(x.get("silence_fraction", 1.0)),
                -float(x.get("duration", 0.0)),
            ),
            reverse=True,
        )
    return grouped


def load_review_state(path: Path):
    state = {}
    for row in read_jsonl(path):
        utt_id = row.get("id")
        action = row.get("action")
        if utt_id is not None and action:
            state[int(utt_id)] = action
    return state


def stage_current(candidate: Path, text: str, record: dict, review_dir: Path):
    review_dir.mkdir(parents=True, exist_ok=True)
    current_wav = review_dir / "current.wav"
    current_txt = review_dir / "current.txt"
    current_json = review_dir / "current.json"

    shutil.copy2(candidate, current_wav)
    current_txt.write_text(text + "\n", encoding="utf-8")
    current_json.write_text(
        json.dumps(
            {
                "id": record.get("id"),
                "attempt": record.get("attempt"),
                "expected": text,
                "heard": record.get("asr", ""),
                "asr_score": record.get("asr_score"),
                "silence_fraction": record.get("silence_fraction"),
                "duration": record.get("duration"),
                "source": str(candidate),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return current_wav


def try_play(path: Path):
    # Best effort only. Inside Docker, direct host playback may not work.
    players = [
        ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", str(path)],
        ["aplay", str(path)],
        ["paplay", str(path)],
    ]
    for cmd in players:
        try:
            result = subprocess.run(cmd, check=False)
            if result.returncode == 0:
                return True
        except FileNotFoundError:
            continue
    return False


def print_candidate(utt_id, idx, items, text, item, current_wav, done_count, total_count):
    silence = float(item.get("silence_fraction", 0.0))
    duration = float(item.get("duration", 0.0))
    asr_score = float(item.get("asr_score", 0.0))

    print("\n" + "─" * 72)
    print(f"Rejected sample review   [{done_count + 1}/{total_count}]")
    print("─" * 72)
    print(f"ID:       {utt_id}")
    print(f"Attempt:  {item.get('attempt')}   ({idx + 1}/{len(items)} candidates)")
    print(f"Source:   {item['_wav']}")
    print(f"Review:   {current_wav}")
    print()
    print("Expected:")
    print(f"  {text}")
    print()
    print("Whisper:")
    print(f"  {item.get('asr', '')}")
    print()
    print(f"ASR score:   {asr_score:.1f}")
    print(f"Silence:     {silence:.1%}")
    print(f"Duration:    {duration:.2f} sec")
    reasons = item.get("reasons", [])
    if reasons:
        print(f"Reasons:     {', '.join(str(x) for x in reasons)}")
    print()
    print("[A] Accept  [N] Next attempt  [S] Skip  [D] Reject ID  [P] Play  [Q] Quit")


def promote(
    utt_id,
    item,
    text,
    raw_dir: Path,
    metadata: Path,
    audit_path: Path,
    rejected_dir: Path,
):
    filename = f"utt_{utt_id:06d}.wav"
    dest = raw_dir / filename
    raw_dir.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        raise RuntimeError(f"Destination already exists: {dest}")

    shutil.copy2(item["_wav"], dest)
    append_metadata(metadata, filename, text)

    append_jsonl(
        audit_path,
        {
            "id": utt_id,
            "attempt": item.get("attempt"),
            "action": "manual_accept",
            "filename": filename,
            "text": text,
            "source": str(item["_wav"]),
            "asr": item.get("asr", ""),
            "asr_score": item.get("asr_score"),
            "silence_fraction": item.get("silence_fraction"),
            "duration": item.get("duration"),
        },
    )

    archive = rejected_dir / "reviewed"
    archive.mkdir(parents=True, exist_ok=True)
    for other in rejected_dir.glob(f"utt_{utt_id:06d}_attempt*.wav"):
        target = archive / other.name
        if target.exists():
            target.unlink()
        shutil.move(str(other), target)

    return dest


def main():
    ap = argparse.ArgumentParser(description="Interactive manual review tool for rejected Chatterbox samples")
    ap.add_argument("--model", default=os.getenv("TTS_MODEL", "turbo"), choices=["original", "turbo"])
    ap.add_argument("--corpus", default="corpus.txt")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=-1)
    ap.add_argument("--include-reviewed", action="store_true")
    args = ap.parse_args()

    model = args.model
    corpus_path = Path(args.corpus)
    corpus = load_corpus(corpus_path)

    dataset_root = Path("dataset_raw") / model
    raw_dir = dataset_root / "wavs"
    metadata = dataset_root / "metadata.csv"
    rejected_dir = Path("rejected") / model
    generation_log = Path("logs") / f"generation-{model}.jsonl"
    review_log = Path("logs") / f"manual-review-{model}.jsonl"
    review_dir = Path("review")

    generation_rows = read_jsonl(generation_log)
    grouped = group_attempts(generation_rows, rejected_dir)
    review_state = load_review_state(review_log)
    accepted_metadata = load_metadata(metadata)

    ids = []
    for utt_id in sorted(grouped):
        if utt_id < args.start:
            continue
        if args.end >= 0 and utt_id > args.end:
            continue
        filename = f"utt_{utt_id:06d}.wav"
        if filename in accepted_metadata:
            continue
        if not args.include_reviewed and review_state.get(utt_id) in {"manual_accept", "manual_reject"}:
            continue
        ids.append(utt_id)

    if not ids:
        print("No rejected samples need review.")
        return

    print(f"Model: {model}")
    print(f"Rejected IDs available for review: {len(ids)}")
    print(f"Current WAV will be staged at: {review_dir / 'current.wav'}")

    position = 0
    while position < len(ids):
        utt_id = ids[position]
        items = grouped[utt_id]
        if not (0 <= utt_id < len(corpus)):
            append_jsonl(
                review_log,
                {"id": utt_id, "action": "skip_missing_corpus_text"},
            )
            position += 1
            continue

        text = corpus[utt_id]
        attempt_idx = 0

        while attempt_idx < len(items):
            item = items[attempt_idx]
            current_wav = stage_current(item["_wav"], text, item, review_dir)
            print_candidate(utt_id, attempt_idx, items, text, item, current_wav, position, len(ids))

            choice = input("Choice: ").strip().lower()

            if choice == "a":
                try:
                    dest = promote(
                        utt_id,
                        item,
                        text,
                        raw_dir,
                        metadata,
                        review_log,
                        rejected_dir,
                    )
                    print(f"Accepted -> {dest}")
                    review_state[utt_id] = "manual_accept"
                    position += 1
                    break
                except Exception as exc:
                    print(f"Could not accept sample: {exc}")

            elif choice == "n":
                attempt_idx += 1
                if attempt_idx >= len(items):
                    print("No more attempts for this ID. Leaving it for later.")
                    append_jsonl(
                        review_log,
                        {"id": utt_id, "action": "skip", "reason": "no_more_attempts"},
                    )
                    position += 1
                    break

            elif choice == "s":
                append_jsonl(
                    review_log,
                    {"id": utt_id, "action": "skip"},
                )
                print("Skipped for now.")
                position += 1
                break

            elif choice == "d":
                append_jsonl(
                    review_log,
                    {
                        "id": utt_id,
                        "action": "manual_reject",
                        "text": text,
                    },
                )
                review_state[utt_id] = "manual_reject"
                print("Marked as manually rejected.")
                position += 1
                break

            elif choice == "p":
                if not try_play(current_wav):
                    print(f"Could not play inside the container. Open this on the host: {current_wav}")

            elif choice == "q":
                print("Review state saved. Exiting.")
                return

            else:
                print("Unknown choice. Use A, N, S, D, P, or Q.")

    print("\nReview complete.")
    print(f"Accepted metadata: {metadata}")
    print(f"Manual review log: {review_log}")


if __name__ == "__main__":
    main()
