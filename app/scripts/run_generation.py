#!/usr/bin/env python3
"""
Cross-platform supervisor for long Chatterbox dataset generation jobs.

Runs scripts/generate.py in fresh child processes using fixed-size chunks.
Each child process loads Chatterbox/Whisper/CUDA, processes one chunk, and exits,
so GPU/process memory is fully reset between chunks.

Designed to run inside the existing Docker generator container:

    python scripts/run_generation.py

Configuration can be supplied through .env / Docker Compose:

    GENERATION_CHUNK_SIZE=2000
    GENERATION_MAX_RETRIES=5
    GENERATION_RETRY_DELAY=10
    TTS_MODEL=turbo

The supervisor persists progress under logs/ so an interrupted full-corpus run
can be resumed with the same command.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def read_corpus(path: Path):
    if not path.exists():
        raise SystemExit(f"Missing corpus: {path}")
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def corpus_fingerprint(lines) -> str:
    payload = "\n".join(lines).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def write_json_atomic(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def save_state(
    state_path: Path,
    *,
    model: str,
    corpus_path: Path,
    fingerprint: str,
    total: int,
    next_start: int,
    requested_end: int,
    chunk_size: int,
    status: str,
    last_chunk=None,
):
    state = {
        "version": 1,
        "model": model,
        "corpus": str(corpus_path),
        "corpus_sha256": fingerprint,
        "corpus_lines": total,
        "next_start": next_start,
        "requested_end": requested_end,
        "chunk_size": chunk_size,
        "status": status,
        "updated_at": now_iso(),
    }
    if last_chunk is not None:
        state["last_chunk"] = last_chunk
    write_json_atomic(state_path, state)


def build_worker_command(worker: Path, corpus: Path, start: int, limit: int, reference):
    cmd = [
        sys.executable,
        str(worker),
        "--corpus",
        str(corpus),
        "--start",
        str(start),
        "--limit",
        str(limit),
    ]
    if reference:
        cmd.extend(["--reference", reference])
    return cmd


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Generate an entire corpus in fresh subprocess chunks, automatically "
            "restarting failed chunks and persisting progress."
        )
    )
    ap.add_argument("--corpus", default=os.getenv("CORPUS_FILE", "corpus.txt"))
    ap.add_argument("--worker", default="scripts/generate.py")
    ap.add_argument("--reference", default=os.getenv("REFERENCE_AUDIO"))
    ap.add_argument(
        "--chunk-size",
        type=int,
        default=env_int("GENERATION_CHUNK_SIZE", 2000),
        help="Prompts per fresh worker process (default: 2000)",
    )
    ap.add_argument(
        "--max-retries",
        type=int,
        default=env_int("GENERATION_MAX_RETRIES", 5),
        help="Retries for a crashed/non-zero worker chunk (default: 5)",
    )
    ap.add_argument(
        "--retry-delay",
        type=int,
        default=env_int("GENERATION_RETRY_DELAY", 10),
        help="Seconds before retrying a crashed chunk (default: 10)",
    )
    ap.add_argument(
        "--start",
        type=int,
        default=None,
        help="Explicit corpus index to start at; overrides saved progress",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional total number of corpus prompts to cover; 0 = through end",
    )
    ap.add_argument(
        "--reset-state",
        action="store_true",
        help="Discard saved supervisor progress and start from --start or 0",
    )
    ap.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore saved progress for this invocation without deleting it",
    )
    args = ap.parse_args()

    if args.chunk_size <= 0:
        raise SystemExit("--chunk-size must be greater than zero")
    if args.max_retries < 0:
        raise SystemExit("--max-retries cannot be negative")

    model = os.getenv("TTS_MODEL", "original").lower()
    if model not in {"original", "turbo"}:
        raise SystemExit(f"Invalid TTS_MODEL '{model}'. Use 'original' or 'turbo'.")

    corpus_path = Path(args.corpus)
    worker = Path(args.worker)

    if not worker.exists():
        raise SystemExit(f"Missing worker generator: {worker}")

    lines = read_corpus(corpus_path)
    total = len(lines)
    fingerprint = corpus_fingerprint(lines)

    if total == 0:
        raise SystemExit("Corpus contains no non-empty prompts.")

    state_path = Path("logs") / f"generation-supervisor-{model}.json"

    if args.reset_state and state_path.exists():
        state_path.unlink()

    state = None if args.no_resume else read_json(state_path)

    # Validate saved state before using it.
    resume_start = 0
    if state and args.start is None:
        same_model = state.get("model") == model
        same_corpus = state.get("corpus_sha256") == fingerprint
        same_total = state.get("corpus_lines") == total

        if same_model and same_corpus and same_total:
            resume_start = int(state.get("next_start", 0))
            print(f"Resuming saved generation state at corpus index {resume_start}.")
        else:
            print("Saved generation state does not match the current model/corpus.")
            print("Starting from 0. Use --reset-state to replace the old state explicitly.")

    start = args.start if args.start is not None else resume_start
    start = max(0, min(start, total))

    if args.limit > 0:
        requested_end = min(total, start + args.limit)
    else:
        requested_end = total

    if start >= requested_end:
        print("Nothing to generate: requested range is already complete.")
        return

    print()
    print("=" * 72)
    print("Full corpus generation supervisor")
    print("=" * 72)
    print(f"TTS model:       {model}")
    print(f"Corpus:          {corpus_path}")
    print(f"Corpus prompts:  {total}")
    print(f"Start index:     {start}")
    print(f"End index:       {requested_end - 1}")
    print(f"Chunk size:      {args.chunk_size}")
    print(f"Max retries:     {args.max_retries}")
    print(f"State file:      {state_path}")
    print("=" * 72)

    current = start

    save_state(
        state_path,
        model=model,
        corpus_path=corpus_path,
        fingerprint=fingerprint,
        total=total,
        next_start=current,
        requested_end=requested_end,
        chunk_size=args.chunk_size,
        status="running",
    )

    try:
        while current < requested_end:
            limit = min(args.chunk_size, requested_end - current)
            chunk_end = current + limit - 1

            print()
            print("#" * 72)
            print(f"Starting fresh worker for corpus {current}–{chunk_end}")
            print("#" * 72)

            cmd = build_worker_command(
                worker,
                corpus_path,
                current,
                limit,
                args.reference,
            )

            success = False
            final_exit_code = None

            # First run + N retries.
            for attempt in range(1, args.max_retries + 2):
                if attempt > 1:
                    print(
                        f"\nRetrying chunk {current}–{chunk_end} "
                        f"(retry {attempt - 1}/{args.max_retries})..."
                    )
                    if args.retry_delay > 0:
                        time.sleep(args.retry_delay)

                result = subprocess.run(cmd)
                final_exit_code = result.returncode

                if result.returncode == 0:
                    success = True
                    break

                print(
                    f"Worker exited with code {result.returncode} "
                    f"for chunk {current}–{chunk_end}."
                )

            if not success:
                save_state(
                    state_path,
                    model=model,
                    corpus_path=corpus_path,
                    fingerprint=fingerprint,
                    total=total,
                    next_start=current,
                    requested_end=requested_end,
                    chunk_size=args.chunk_size,
                    status="failed",
                    last_chunk={
                        "start": current,
                        "end": chunk_end,
                        "exit_code": final_exit_code,
                        "completed": False,
                    },
                )
                raise SystemExit(
                    f"Chunk {current}–{chunk_end} failed after "
                    f"{args.max_retries + 1} total attempts. "
                    f"Run the same supervisor command again to resume here."
                )

            next_start = current + limit

            save_state(
                state_path,
                model=model,
                corpus_path=corpus_path,
                fingerprint=fingerprint,
                total=total,
                next_start=next_start,
                requested_end=requested_end,
                chunk_size=args.chunk_size,
                status="running" if next_start < requested_end else "complete",
                last_chunk={
                    "start": current,
                    "end": chunk_end,
                    "exit_code": 0,
                    "completed": True,
                },
            )

            print()
            print(
                f"Chunk {current}–{chunk_end} complete. "
                "Worker exited; Chatterbox/Whisper/CUDA state has been released."
            )

            current = next_start

    except KeyboardInterrupt:
        save_state(
            state_path,
            model=model,
            corpus_path=corpus_path,
            fingerprint=fingerprint,
            total=total,
            next_start=current,
            requested_end=requested_end,
            chunk_size=args.chunk_size,
            status="interrupted",
        )
        print()
        print(
            f"Interrupted. Progress saved at {state_path}. "
            "Run the same command again to resume."
        )
        raise SystemExit(130)

    print()
    print("=" * 72)
    print("FULL CORPUS RANGE COMPLETE")
    print("=" * 72)
    print(f"Processed range: {start}–{requested_end - 1}")
    print(f"Supervisor state: {state_path}")
    print()
    print(
        "Note: prompts that failed all validator attempts remain under rejected/ "
        "for manual review; they do not make a successfully completed chunk fail."
    )


if __name__ == "__main__":
    main()
