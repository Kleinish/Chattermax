from __future__ import annotations

import docker
import asyncio
import csv
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import httpx

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

ROOT = Path(os.getenv("WORKSPACE_ROOT", "/workspace"))
LOGS = ROOT / "logs"
WEB_LOG = LOGS / "webui-current.log"
WEB_STATE = LOGS / "webui-job.json"
ENV_FILE = ROOT / ".env"
SPEECH_ISOLATOR_URL = os.getenv("SPEECH_ISOLATOR_URL", "http://speech-isolator:7860").rstrip("/")

DOCKER_PROJECT = os.getenv("COMPOSE_PROJECT_NAME", "chattermax")
SPEECH_ISOLATOR_SERVICE = "speech-isolator"

def docker_client():
    return docker.from_env()

def find_compose_container(service: str):
    """Find a container belonging to this Compose service."""
    client = docker_client()

    containers = client.containers.list(
        all=True,
        filters={
            "label": f"com.docker.compose.service={service}"
        },
    )

    if not containers:
        return None

    return containers[0]

def start_compose_service(service: str):
    container = find_compose_container(service)

    if container is None:
        raise RuntimeError(
            f"No Compose container exists for {service}. "
            f"Create it once with: docker compose --profile jobs create {service}"
        )

    container.reload()

    if container.status != "running":
        container.start()

    return container


def stop_compose_service(service: str) -> None:
    subprocess.run(
        [
            "docker",
            "compose",
            "stop",
            service,
        ],
        cwd=ROOT,
        check=False,
        env=process_env(),
    )

    subprocess.run(
        [
            "docker",
            "compose",
            "rm",
            "-f",
            service,
        ],
        cwd=ROOT,
        check=False,
        env=process_env(),
    )

app = FastAPI(title="ChatterMAX", version="0.1.0")
app.mount("/static", StaticFiles(directory=ROOT / "web" / "static"), name="static")
templates = Jinja2Templates(directory=ROOT / "web" / "templates")

SETTINGS = [
    "TTS_MODEL",
    "REFERENCE_AUDIO",
    "WHISPER_MODEL",
    "WHISPER_DEVICE",
    "MIN_ASR_SCORE",
    "MIN_DURATION_SEC",
    "MAX_DURATION_SEC",
    "MAX_SILENCE_FRACTION",
    "MAX_CLIPPED_FRACTION",
    "MAX_GENERATION_ATTEMPTS",
    "GENERATION_CHUNK_SIZE",
    "GENERATION_MAX_RETRIES",
    "GENERATION_RETRY_DELAY",
    "CORPUS_TARGET",
]

DEFAULTS = {
    "TTS_MODEL": "turbo",
    "REFERENCE_AUDIO": "input/reference.wav",
    "WHISPER_MODEL": "medium.en",
    "WHISPER_DEVICE": "cuda",
    "MIN_ASR_SCORE": "92",
    "MIN_DURATION_SEC": "0.60",
    "MAX_DURATION_SEC": "18.0",
    "MAX_SILENCE_FRACTION": "0.50",
    "MAX_CLIPPED_FRACTION": "0.0005",
    "MAX_GENERATION_ATTEMPTS": "4",
    "GENERATION_CHUNK_SIZE": "2000",
    "GENERATION_MAX_RETRIES": "5",
    "GENERATION_RETRY_DELAY": "10",
    "CORPUS_TARGET": "20000",
}


def parse_env(path: Path = ENV_FILE) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def current_settings() -> dict[str, str]:
    file_env = parse_env()
    return {k: file_env.get(k, os.getenv(k, DEFAULTS.get(k, ""))) for k in SETTINGS}


def write_env_updates(updates: dict[str, str]) -> None:
    existing = ENV_FILE.read_text(encoding="utf-8").splitlines() if ENV_FILE.exists() else []
    wanted = {k: str(v) for k, v in updates.items() if k in SETTINGS}
    seen: set[str] = set()
    output: list[str] = []
    for raw in existing:
        stripped = raw.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key = stripped.split("=", 1)[0].strip()
            if key in wanted:
                output.append(f"{key}={wanted[key]}")
                seen.add(key)
                continue
        output.append(raw)
    for key, value in wanted.items():
        if key not in seen:
            output.append(f"{key}={value}")
    ENV_FILE.write_text("\n".join(output).rstrip() + "\n", encoding="utf-8")


def process_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update(parse_env())
    env["PYTHONUNBUFFERED"] = "1"
    return env


def reference_path(settings: dict[str, str] | None = None) -> Path:
    settings = settings or current_settings()
    configured = settings.get("REFERENCE_AUDIO", "input/reference.wav").strip() or "input/reference.wav"
    path = Path(configured)
    return path if path.is_absolute() else ROOT / path

def custom_corpus_path() -> Path:
    return ROOT / "corpus_sources" / "custom.txt"

def reference_status(settings: dict[str, str] | None = None) -> dict[str, Any]:
    settings = settings or current_settings()
    path = reference_path(settings)
    exists = path.is_file() and path.stat().st_size > 0
    return {
        "exists": exists,
        "path": settings.get("REFERENCE_AUDIO", "input/reference.wav"),
        "size": path.stat().st_size if exists else 0,
    }

def original_reference_path() -> Path | None:
    input_dir = ROOT / "input"
    candidates = [
        p for p in input_dir.glob("reference.*")
        if p.is_file() and p.stat().st_size > 0
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime_ns)


def cleaned_reference_path() -> Path:
    return ROOT / "input" / "reference_clean.wav"


def reference_cleaner_status() -> dict[str, Any]:
    original = original_reference_path()
    cleaned = cleaned_reference_path()
    active = reference_path()
    return {
        "original_exists": bool(original),
        "original_path": str(original.relative_to(ROOT)) if original else None,
        "cleaned_exists": cleaned.is_file() and cleaned.stat().st_size > 0,
        "cleaned_path": str(cleaned.relative_to(ROOT)),
        "using_cleaned": active.resolve() == cleaned.resolve() if active.exists() or cleaned.exists() else False,
    }


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


class JobManager:
    def __init__(self):
        self.lock = threading.RLock()
        self.proc: subprocess.Popen | None = None
        self.log_handle = None
        self.state = read_json(WEB_STATE, {}) or {}
        if self.state.get("status") == "running":
            self.state["status"] = "interrupted"
            self.state["message"] = "Web UI restarted while a job was marked running."
            write_json(WEB_STATE, self.state)

    def status(self) -> dict[str, Any]:
        with self.lock:
            running = self.proc is not None and self.proc.poll() is None
            state = dict(self.state)
            state["running"] = running
            if self.proc is not None and not running:
                state["exit_code"] = self.proc.returncode
            return state

    def start(self, kind: str, cmd: list[str]) -> dict[str, Any]:
        with self.lock:
            if self.proc is not None and self.proc.poll() is None:
                raise RuntimeError("Another job is already running")
            LOGS.mkdir(parents=True, exist_ok=True)
            WEB_LOG.write_text("", encoding="utf-8")
            self.log_handle = WEB_LOG.open("a", encoding="utf-8", buffering=1)
            self.state = {
                "kind": kind,
                "status": "running",
                "started_at": time.time(),
                "command": cmd,
            }
            write_json(WEB_STATE, self.state)
            self.proc = subprocess.Popen(
                cmd,
                cwd=ROOT,
                env=process_env(),
                stdout=self.log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            threading.Thread(target=self._waiter, daemon=True).start()
            return self.status()

    def _waiter(self):
        proc = self.proc
        if proc is None:
            return
        code = proc.wait()
        with self.lock:
            self.state["status"] = "complete" if code == 0 else "failed"
            self.state["finished_at"] = time.time()
            self.state["exit_code"] = code
            write_json(WEB_STATE, self.state)
            if self.log_handle:
                self.log_handle.close()
                self.log_handle = None

    def stop(self):
        with self.lock:
            proc = self.proc
            if proc is None or proc.poll() is not None:
                return self.status()
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            self.state["status"] = "stopping"
            write_json(WEB_STATE, self.state)
            return dict(self.state, running=True)


jobs = JobManager()


class DatasetCache:
    def __init__(self):
        self.key = None
        self.value = None

    def stats(self, model: str) -> dict[str, Any]:
        metadata = ROOT / "dataset_raw" / model / "metadata.csv"
        audit = ROOT / "logs" / f"generation-{model}.jsonl"
        corpus = ROOT / "corpus.txt"
        key = tuple((p.stat().st_mtime_ns, p.stat().st_size) if p.exists() else (0, 0) for p in (metadata, audit, corpus))
        if key == self.key and self.value is not None:
            return self.value

        accepted_ids: set[int] = set()
        accepted_count = 0
        if metadata.exists():
            with metadata.open("r", encoding="utf-8", newline="") as f:
                for row in csv.reader(f, delimiter="|"):
                    if not row:
                        continue
                    accepted_count += 1
                    stem = Path(row[0]).stem
                    try:
                        accepted_ids.add(int(stem.split("_")[-1]))
                    except Exception:
                        pass

        rejected_ids: set[int] = set()
        if audit.exists():
            for line in audit.open("r", encoding="utf-8"):
                try:
                    r = json.loads(line)
                    if not r.get("accepted") and r.get("id") is not None:
                        rejected_ids.add(int(r["id"]))
                except Exception:
                    continue
        rejected_ids -= accepted_ids

        total = 0
        if corpus.exists():
            total = sum(1 for x in corpus.read_text(encoding="utf-8").splitlines() if x.strip())

        self.key = key
        self.value = {
            "accepted": accepted_count,
            "rejected_ids": len(rejected_ids),
            "corpus_total": total,
            "remaining": max(total - accepted_count, 0),
            "percent": round((accepted_count / total * 100.0), 1) if total else 0.0,
        }
        return self.value


dataset_cache = DatasetCache()


def supervisor_state(model: str):
    return read_json(ROOT / "logs" / f"generation-supervisor-{model}.json", {}) or {}


def read_corpus() -> list[str]:
    path = ROOT / "corpus.txt"
    if not path.exists():
        return []
    return [x.strip() for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def review_state(model: str) -> dict[int, str]:
    path = ROOT / "logs" / f"manual-review-{model}.jsonl"
    result: dict[int, str] = {}
    if not path.exists():
        return result
    for line in path.open("r", encoding="utf-8"):
        try:
            row = json.loads(line)
            if row.get("id") is not None and row.get("action"):
                result[int(row["id"])] = row["action"]
        except Exception:
            pass
    return result


def rejection_groups(model: str):
    audit = ROOT / "logs" / f"generation-{model}.jsonl"
    rejected_dir = ROOT / "rejected" / model
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    if not audit.exists():
        return groups
    for line in audit.open("r", encoding="utf-8"):
        try:
            row = json.loads(line)
        except Exception:
            continue
        if row.get("accepted") or row.get("id") is None or row.get("attempt") is None:
            continue
        uid = int(row["id"])
        attempt = int(row["attempt"])
        candidates = [
            rejected_dir / f"utt_{uid:06d}_attempt{attempt}.wav",
            rejected_dir / f"utt_{uid:06d}_attempt{attempt}_error.wav",
        ]
        wav = next((p for p in candidates if p.exists()), None)
        if wav is None:
            continue
        item = dict(row)
        item["wav_name"] = wav.name
        groups[uid].append(item)
    for uid in groups:
        groups[uid].sort(key=lambda r: (float(r.get("asr_score", 0)), -float(r.get("silence_fraction", 1))), reverse=True)
    return groups


def append_jsonl(path: Path, obj: dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def append_metadata(path: Path, filename: str, text: str):
    existing = set()
    if path.exists():
        with path.open("r", encoding="utf-8", newline="") as f:
            existing = {row[0] for row in csv.reader(f, delimiter="|") if row}
    if filename in existing:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="") as f:
        csv.writer(f, delimiter="|", lineterminator="\n").writerow([filename, text])


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html", context={})


@app.get("/api/status")
def api_status():
    settings = current_settings()
    model = settings.get("TTS_MODEL", "turbo")
    ref = reference_status(settings)
    return {
        "job": jobs.status(),
        "dataset": dataset_cache.stats(model),
        "supervisor": supervisor_state(model),
        "settings": settings,
        "reference_exists": ref["exists"],
        "reference": ref,
        "reference_cleaner": reference_cleaner_status(),
        "corpus_exists": (ROOT / "corpus.txt").exists(),
    }


@app.get("/api/settings")
def get_settings():
    return current_settings()


@app.post("/api/settings")
async def save_settings(request: Request):
    payload = await request.json()
    updates = {k: str(v) for k, v in payload.items() if k in SETTINGS}
    write_env_updates(updates)
    return current_settings()

@app.get("/api/corpus/custom")
def custom_corpus_status():
    path = custom_corpus_path()

    if not path.exists():
        return {
            "exists": False,
            "filename": None,
            "size": 0,
            "phrases": 0,
        }

    lines = path.read_text(
        encoding="utf-8",
        errors="ignore",
    ).splitlines()

    phrases = sum(
        1
        for line in lines
        if line.strip() and not line.strip().startswith("#")
    )

    return {
        "exists": True,
        "filename": "custom.txt",
        "size": path.stat().st_size,
        "phrases": phrases,
    }


@app.post("/api/corpus/custom")
async def upload_custom_corpus(file: UploadFile = File(...)):
    filename = Path(file.filename or "").name

    if Path(filename).suffix.lower() != ".txt":
        raise HTTPException(
            400,
            "Custom phrases must be uploaded as a .txt file",
        )

    target = custom_corpus_path()
    target.parent.mkdir(parents=True, exist_ok=True)

    temp = target.with_suffix(".txt.uploading")

    try:
        with temp.open("wb") as output:
            shutil.copyfileobj(file.file, output)
            output.flush()
            os.fsync(output.fileno())

        if not temp.exists() or temp.stat().st_size == 0:
            raise HTTPException(
                400,
                "Uploaded custom phrase file is empty",
            )

        # Validate that it is readable text before replacing the current file.
        try:
            text = temp.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            raise HTTPException(
                400,
                "Custom phrase file must be UTF-8 text",
            )

        phrases = [
            line.strip()
            for line in text.splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]

        if not phrases:
            raise HTTPException(
                400,
                "The file does not contain any usable phrases",
            )

        os.replace(temp, target)

        return {
            "ok": True,
            "filename": "custom.txt",
            "size": target.stat().st_size,
            "phrases": len(phrases),
        }

    finally:
        temp.unlink(missing_ok=True)

        try:
            await file.close()
        except Exception:
            pass

@app.post("/api/reference")
async def upload_reference(file: UploadFile = File(...)):
    suffix = Path(file.filename or "reference.wav").suffix.lower()
    if suffix not in {".wav", ".mp3", ".flac", ".ogg", ".m4a"}:
        raise HTTPException(400, "Use WAV, MP3, FLAC, OGG, or M4A audio")

    target = ROOT / "input" / ("reference" + suffix)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(target.suffix + ".uploading")

    try:
        with temp.open("wb") as f:
            shutil.copyfileobj(file.file, f)
            f.flush()
            os.fsync(f.fileno())

        if not temp.exists() or temp.stat().st_size == 0:
            raise HTTPException(400, "Uploaded reference audio is empty")

        os.replace(temp, target)

        # Keep only the newest uploaded original. A new upload also invalidates
        # any previously cleaned reference so the UI cannot accidentally reuse it.
        for old_ref in (ROOT / "input").glob("reference.*"):
            if old_ref != target and old_ref.is_file():
                old_ref.unlink(missing_ok=True)
        cleaned_reference_path().unlink(missing_ok=True)

        relative = str(target.relative_to(ROOT))
        write_env_updates({"REFERENCE_AUDIO": relative})

        return {
            "ok": True,
            "path": relative,
            "size": target.stat().st_size,
        }
    finally:
        try:
            await file.close()
        except Exception:
            pass
        if temp.exists():
            temp.unlink(missing_ok=True)


@app.get("/audio/reference/{kind}")
def reference_audio(kind: str):
    if kind == "original":
        path = original_reference_path()
    elif kind == "cleaned":
        path = cleaned_reference_path()
    elif kind == "active":
        path = reference_path()
    else:
        raise HTTPException(404, "Unknown reference audio type")

    if path is None or not path.is_file() or path.stat().st_size == 0:
        raise HTTPException(404, "Reference audio not found")
    return FileResponse(path, media_type="audio/wav", filename=path.name)


@app.post("/api/reference/clean")
async def clean_reference(request: Request):
    if jobs.status().get("running"):
        raise HTTPException(
            409,
            "Stop the current generation job before cleaning the reference voice."
        )

    source = original_reference_path() or reference_path()
    if not source.is_file() or source.stat().st_size == 0:
        raise HTTPException(
            400,
            "Upload a reference voice before cleaning it."
        )

    payload = await request.json()
    strength = str(payload.get("strength", "medium")).lower()

    if strength not in {"light", "medium", "strong"}:
        raise HTTPException(
            400,
            "Strength must be light, medium, or strong."
        )

    normalize = bool(payload.get("normalize", True))
    remove_rumble = bool(payload.get("remove_rumble", True))

    container = None

    try:
        container = start_compose_service(SPEECH_ISOLATOR_SERVICE)

        # Wait up to 60 seconds for the service to become ready.
        deadline = time.time() + 60

        async with httpx.AsyncClient(
            timeout=httpx.Timeout(600.0, connect=5.0)
        ) as client:

            while True:
                try:
                    health = await client.get(
                        f"{SPEECH_ISOLATOR_URL}/api/health"
                    )

                    if health.status_code == 200:
                        break

                except httpx.HTTPError:
                    pass

                if time.time() >= deadline:
                    raise RuntimeError(
                        "Speech Isolator did not become ready within 60 seconds."
                    )

                await asyncio.sleep(1)

            with source.open("rb") as audio_file:
                response = await client.post(
                    f"{SPEECH_ISOLATOR_URL}/api/clean",
                    files={
                        "file": (
                            source.name,
                            audio_file,
                            "application/octet-stream"
                        )
                    },
                    data={
                        "strength": strength,
                        "normalize": str(normalize).lower(),
                        "remove_rumble": str(remove_rumble).lower(),
                    },
                )

            response.raise_for_status()

            result = response.json()
            cleaned_url = result.get("cleaned_url")

            if not cleaned_url:
                raise RuntimeError(
                    "Speech Isolator did not return a cleaned file URL."
                )

            cleaned_response = await client.get(
                f"{SPEECH_ISOLATOR_URL}{cleaned_url}"
            )

            cleaned_response.raise_for_status()
            cleaned_bytes = cleaned_response.content
            

    except httpx.HTTPStatusError as exc:
        detail = (
            exc.response.text[-4000:]
            if exc.response is not None
            else str(exc)
        )
        raise HTTPException(
            502,
            f"Speech Isolator failed: {detail}"
        ) from exc

    except (httpx.HTTPError, RuntimeError) as exc:
        raise HTTPException(
            502,
            f"Speech Isolator unavailable: {exc}"
        ) from exc

    finally:
        if container is not None:
            try:
                container.stop(timeout=10)
            except Exception:
                pass
    
    target = cleaned_reference_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(".wav.downloading")
    temp.write_bytes(cleaned_bytes)
    if temp.stat().st_size == 0:
        temp.unlink(missing_ok=True)
        raise HTTPException(502, "Speech Isolator returned an empty file.")
    os.replace(temp, target)

    relative = str(target.relative_to(ROOT))
    write_env_updates({"REFERENCE_AUDIO": relative})
    return {
        "ok": True,
        "job_id": result.get("job_id"),
        "path": relative,
        "strength": strength,
        "size": target.stat().st_size,
    }


@app.post("/api/reference/use/{kind}")
def use_reference(kind: str):
    if jobs.status().get("running"):
        raise HTTPException(409, "Stop the current generation job before changing the reference voice.")

    if kind == "original":
        path = original_reference_path()
    elif kind == "cleaned":
        path = cleaned_reference_path()
    else:
        raise HTTPException(400, "Reference type must be original or cleaned.")

    if path is None or not path.is_file() or path.stat().st_size == 0:
        raise HTTPException(404, f"{kind.capitalize()} reference audio is not available.")

    relative = str(path.relative_to(ROOT))
    write_env_updates({"REFERENCE_AUDIO": relative})
    return {"ok": True, "path": relative, "kind": kind}


@app.post("/api/jobs/generate")
async def start_generation(request: Request):
    settings = current_settings()
    ref = reference_status(settings)
    if not ref["exists"]:
        raise HTTPException(
            400,
            f"Reference audio is missing: {ref['path']}. Upload a reference voice before starting generation.",
        )
    if not (ROOT / "corpus.txt").is_file():
        raise HTTPException(400, "corpus.txt is missing. Build or upload the corpus before starting generation.")

    payload = await request.json()
    cmd = [sys.executable, "scripts/run_generation.py"]
    if payload.get("reset_state"):
        cmd.append("--reset-state")
    if payload.get("start") not in (None, ""):
        cmd.extend(["--start", str(int(payload["start"]))])
    try:
        return jobs.start("generation", cmd)
    except RuntimeError as e:
        raise HTTPException(409, str(e))


@app.post("/api/jobs/corpus")
def start_corpus():
    try:
        return jobs.start("corpus", [sys.executable, "scripts/prepare_corpus.py"])
    except RuntimeError as e:
        raise HTTPException(409, str(e))


@app.post("/api/jobs/validate")
def start_validate():
    try:
        return jobs.start("validate", [sys.executable, "scripts/validate.py"])
    except RuntimeError as e:
        raise HTTPException(409, str(e))


@app.post("/api/jobs/prepare")
def start_prepare():
    try:
        return jobs.start("prepare", [sys.executable, "scripts/prepare_dataset.py"])
    except RuntimeError as e:
        raise HTTPException(409, str(e))


@app.post("/api/jobs/stop")
def stop_job():
    return jobs.stop()


@app.get("/api/logs")
def get_logs(lines: int = 200):
    if not WEB_LOG.exists():
        return {"lines": []}
    data = WEB_LOG.read_text(encoding="utf-8", errors="replace").splitlines()
    return {"lines": data[-max(1, min(lines, 2000)):]}


@app.get("/api/logs/stream")
async def stream_logs():
    async def events():
        pos = 0
        while True:
            if WEB_LOG.exists():
                try:
                    with WEB_LOG.open("r", encoding="utf-8", errors="replace") as f:
                        f.seek(pos)
                        chunk = f.read()
                        pos = f.tell()
                    if chunk:
                        for line in chunk.splitlines():
                            yield f"data: {json.dumps({'line': line})}\n\n"
                except Exception:
                    pass
            yield f"event: heartbeat\ndata: {json.dumps({'ts': time.time()})}\n\n"
            await asyncio.sleep(1)
    return StreamingResponse(events(), media_type="text/event-stream")


@app.get("/api/rejects")
def list_rejects():
    model = current_settings().get("TTS_MODEL", "turbo")
    groups = rejection_groups(model)
    corpus = read_corpus()
    reviewed = review_state(model)
    accepted = set()
    metadata = ROOT / "dataset_raw" / model / "metadata.csv"
    if metadata.exists():
        with metadata.open("r", encoding="utf-8", newline="") as f:
            accepted = {Path(row[0]).stem for row in csv.reader(f, delimiter="|") if row}

    items = []
    for uid in sorted(groups):
        if f"utt_{uid:06d}" in accepted:
            continue
        if reviewed.get(uid) in {"manual_accept", "manual_reject"}:
            continue
        if uid >= len(corpus):
            continue
        candidates = groups[uid]
        if not candidates:
            continue
        best = candidates[0]
        items.append({
            "id": uid,
            "text": corpus[uid],
            "attempts": len(candidates),
            "best": best,
        })
    return {"model": model, "count": len(items), "items": items[:1000]}


@app.get("/api/rejects/{uid}")
def reject_detail(uid: int):
    model = current_settings().get("TTS_MODEL", "turbo")
    groups = rejection_groups(model)
    corpus = read_corpus()
    if uid not in groups or uid >= len(corpus):
        raise HTTPException(404, "Rejected utterance not found")
    attempts = []
    for item in groups[uid]:
        item = dict(item)
        item["audio_url"] = f"/audio/rejected/{model}/{item['wav_name']}"
        attempts.append(item)
    return {"id": uid, "text": corpus[uid], "attempts": attempts}


@app.get("/audio/rejected/{model}/{filename}")
def rejected_audio(model: str, filename: str):
    if model not in {"original", "turbo"} or Path(filename).name != filename:
        raise HTTPException(400, "Invalid path")
    path = ROOT / "rejected" / model / filename
    if not path.exists():
        raise HTTPException(404, "Audio not found")
    return FileResponse(path, media_type="audio/wav")


@app.post("/api/rejects/{uid}/accept")
async def accept_reject(uid: int, request: Request):
    payload = await request.json()
    attempt = int(payload.get("attempt", 0))
    model = current_settings().get("TTS_MODEL", "turbo")
    groups = rejection_groups(model)
    corpus = read_corpus()
    if uid not in groups or uid >= len(corpus):
        raise HTTPException(404, "Rejected utterance not found")
    selected = next((x for x in groups[uid] if int(x.get("attempt", -1)) == attempt), None)
    if selected is None:
        raise HTTPException(404, "Attempt not found")

    source = ROOT / "rejected" / model / selected["wav_name"]
    filename = f"utt_{uid:06d}.wav"
    dest = ROOT / "dataset_raw" / model / "wavs" / filename
    if dest.exists():
        raise HTTPException(409, "Accepted WAV already exists")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)
    append_metadata(ROOT / "dataset_raw" / model / "metadata.csv", filename, corpus[uid])

    review_log = ROOT / "logs" / f"manual-review-{model}.jsonl"
    append_jsonl(review_log, {
        "id": uid,
        "attempt": attempt,
        "action": "manual_accept",
        "filename": filename,
        "text": corpus[uid],
        "source": str(source.relative_to(ROOT)),
        "asr": selected.get("asr", ""),
        "asr_score": selected.get("asr_score"),
    })

    archive = ROOT / "rejected" / model / "reviewed"
    archive.mkdir(parents=True, exist_ok=True)
    for candidate in (ROOT / "rejected" / model).glob(f"utt_{uid:06d}_attempt*.wav"):
        target = archive / candidate.name
        if target.exists():
            target.unlink()
        shutil.move(str(candidate), target)
    return {"ok": True, "destination": str(dest.relative_to(ROOT))}


@app.post("/api/rejects/{uid}/reject")
def manually_reject(uid: int):
    model = current_settings().get("TTS_MODEL", "turbo")
    corpus = read_corpus()
    if uid >= len(corpus):
        raise HTTPException(404, "Utterance not found")
    append_jsonl(ROOT / "logs" / f"manual-review-{model}.jsonl", {
        "id": uid,
        "action": "manual_reject",
        "text": corpus[uid],
    })
    return {"ok": True}


@app.get("/health")
def health():
    return {"ok": True}
