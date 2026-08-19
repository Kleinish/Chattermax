from __future__ import annotations

import os
import re
import shutil
import subprocess
import threading
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

# DeepFilterNet is called through its Python API on purpose.  We do not invoke
# the installed `deepFilter` command-line wrapper, which was the source of the
# earlier TorchAudio compatibility failures.
from df.enhance import enhance, init_df, load_audio, save_audio

DATA_ROOT = Path(os.getenv("DATA_ROOT", "/data"))
INPUT_DIR = DATA_ROOT / "input"
OUTPUT_DIR = DATA_ROOT / "output"
WORK_DIR = DATA_ROOT / "work"
DEFAULT_STRENGTH = os.getenv("DEFAULT_STRENGTH", "medium").lower()
VALID_STRENGTHS = {"light", "medium", "strong"}

for folder in (INPUT_DIR, OUTPUT_DIR, WORK_DIR):
    folder.mkdir(parents=True, exist_ok=True)

PROCESS_LOCK = threading.Lock()
MODEL_LOCK = threading.Lock()
_MODELS: dict[str, tuple] = {}


def safe_name(name: str) -> str:
    name = Path(name).name
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    return stem or "audio"


def run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "Command failed").strip()
        raise RuntimeError(detail[-6000:])


def prepare_audio(source: Path, destination: Path, remove_rumble: bool) -> Path:
    """Convert any FFmpeg-readable source to mono 48 kHz PCM WAV."""
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(source), "-vn", "-ac", "1", "-ar", "48000",
    ]
    if remove_rumble:
        cmd += ["-af", "highpass=f=70"]
    cmd += ["-c:a", "pcm_f32le", str(destination)]
    run(cmd)
    return destination


def _get_model(strength: str):
    """Load and cache normal or post-filter DeepFilterNet3."""
    key = "strong" if strength == "strong" else "normal"
    with MODEL_LOCK:
        if key not in _MODELS:
            model, df_state, _suffix = init_df(
                post_filter=(key == "strong"),
                log_level="INFO",
                log_file=None,
            )
            _MODELS[key] = (model, df_state)
        return _MODELS[key]


def deepfilter_python(prepared: Path, destination: Path, strength: str) -> Path:
    model, df_state = _get_model(strength)
    audio, _meta = load_audio(str(prepared), sr=df_state.sr())

    # Light mode limits maximum attenuation to 12 dB so more of the natural
    # reference recording is preserved. Medium uses full enhancement. Strong
    # uses DeepFilterNet's documented post-filter.
    atten_lim_db = 12.0 if strength == "light" else None

    enhanced = enhance(
        model,
        df_state,
        audio,
        pad=True,
        atten_lim_db=atten_lim_db,
    )
    save_audio(str(destination), enhanced, df_state.sr())
    return destination


def finalize_audio(source: Path, destination: Path, normalize: bool) -> Path:
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(source),
    ]
    if normalize:
        # Conservative normalization for reference voice audio.
        cmd += ["-af", "loudnorm=I=-18:LRA=7:TP=-1.5"]
    cmd += ["-ac", "1", "-ar", "48000", "-c:a", "pcm_s16le", str(destination)]
    run(cmd)
    return destination


def clean_file(
    input_path: Path,
    strength: str = DEFAULT_STRENGTH,
    normalize: bool = True,
    remove_rumble: bool = True,
) -> dict:
    strength = strength.lower().strip()
    if strength not in VALID_STRENGTHS:
        raise ValueError(f"strength must be one of: {', '.join(sorted(VALID_STRENGTHS))}")

    job_id = uuid.uuid4().hex[:12]
    base = f"{job_id}_{safe_name(input_path.stem)}"
    job_work = WORK_DIR / job_id
    job_work.mkdir(parents=True, exist_ok=True)

    prepared = job_work / "prepared_48k.wav"
    enhanced = job_work / "enhanced.wav"
    final = OUTPUT_DIR / f"{base}_clean.wav"

    try:
        prepare_audio(input_path, prepared, remove_rumble)
        # DFState/model use is serialized because the model holds state between calls.
        with PROCESS_LOCK:
            deepfilter_python(prepared, enhanced, strength)
        finalize_audio(enhanced, final, normalize)
    finally:
        shutil.rmtree(job_work, ignore_errors=True)

    return {
        "job_id": job_id,
        "cleaned": final,
        "strength": strength,
        "normalize": normalize,
        "remove_rumble": remove_rumble,
    }



api = FastAPI(title="Speech Isolator API", version="0.4.0")


@api.get("/api/health")
def health():
    return {
        "status": "ok",
        "mode": "reference-audio-cleaner",
        "engine": "DeepFilterNet 0.5.6 Python API",
        "default_strength": DEFAULT_STRENGTH,
        "sample_rate": 48000,
    }


@api.get("/api/model-status")
def model_status():
    return {
        "normal_loaded": "normal" in _MODELS,
        "strong_loaded": "strong" in _MODELS,
    }


@api.post("/api/clean")
async def clean_api(
    file: UploadFile = File(...),
    strength: str = Form(DEFAULT_STRENGTH),
    normalize: bool = Form(True),
    remove_rumble: bool = Form(True),
):
    original_name = safe_name(file.filename or "reference.wav")
    upload_name = f"{uuid.uuid4().hex[:12]}_{original_name}"
    upload_path = INPUT_DIR / upload_name

    try:
        with upload_path.open("wb") as out:
            shutil.copyfileobj(file.file, out)
        result = clean_file(upload_path, strength, normalize, remove_rumble)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        await file.close()
        upload_path.unlink(missing_ok=True)

    return {
        "job_id": result["job_id"],
        "strength": result["strength"],
        "cleaned_url": f"/files/{result['cleaned'].name}",
    }


@api.get("/files/{filename}")
def get_output(filename: str):
    path = OUTPUT_DIR / safe_name(filename)
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path, media_type="audio/wav", filename=path.name)


app = api
