#!/usr/bin/env python3
import argparse
import os
import random
import re
import unicodedata
from pathlib import Path

SPACE_RE = re.compile(r"\s+")


def clean(text: str) -> str:
    text = unicodedata.normalize("NFKC", text.strip())
    text = SPACE_RE.sub(" ", text)
    text = text.replace("|", " ")
    return text.strip()


def key(text: str) -> str:
    return re.sub(r"[^a-z0-9']+", " ", text.lower()).strip()


def main():
    p = argparse.ArgumentParser(description="Build a deduplicated Piper corpus from corpus_sources/*.txt")
    p.add_argument("--source-dir", default="corpus_sources")
    p.add_argument("--output", default="corpus.txt")
    p.add_argument("--target", type=int, default=int(os.getenv("CORPUS_TARGET", "20000")))
    p.add_argument("--min-words", type=int, default=int(os.getenv("MIN_WORDS", "2")))
    p.add_argument("--max-words", type=int, default=int(os.getenv("MAX_WORDS", "35")))
    p.add_argument("--shuffle", action="store_true")
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()

    files = sorted(Path(args.source_dir).glob("*.txt"))
    if not files:
        raise SystemExit(f"No .txt files found in {args.source_dir}")

    seen = set()
    rows = []
    skipped = 0
    for path in files:
        for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            text = clean(raw)
            if not text or text.startswith("#"):
                continue
            words = text.split()
            k = key(text)
            if not k or len(words) < args.min_words or len(words) > args.max_words or k in seen:
                skipped += 1
                continue
            seen.add(k)
            rows.append(text)

    if args.shuffle:
        random.Random(args.seed).shuffle(rows)
    if args.target > 0:
        rows = rows[: args.target]

    Path(args.output).write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"Sources: {len(files)}")
    print(f"Accepted unique phrases: {len(rows)}")
    print(f"Filtered/duplicate lines: {skipped}")
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
