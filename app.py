import os
import shutil
import tempfile
import time
import threading
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Any, Tuple

from fastapi import FastAPI, File, UploadFile, Form, Header, HTTPException, status
from fastapi.responses import JSONResponse
from faster_whisper import WhisperModel
import uvicorn

# Environment configuration
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
CONFIGURED_API_KEY = os.getenv("API_KEY", "").strip()

# Rate limiting configuration
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "10"))
RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))

# Global model instance
model: Optional[WhisperModel] = None


class SlidingWindowRateLimiter:
    """
    Thread-safe in-memory sliding-window rate limiter per client key.
    Tracks timestamp history per key without external dependencies.
    """
    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._lock = threading.Lock()
        self._requests: Dict[str, deque] = defaultdict(deque)

    def check_rate_limit(self, key: str) -> Tuple[bool, int]:
        """
        Check and record a request for the given key.
        Returns:
            (is_limited: bool, retry_after_seconds: int)
        """
        if self.max_requests <= 0:
            return False, 0

        now = time.time()
        with self._lock:
            timestamps = self._requests[key]
            cutoff = now - self.window_seconds

            # Evict timestamps older than the active window
            while timestamps and timestamps[0] <= cutoff:
                timestamps.popleft()

            if len(timestamps) >= self.max_requests:
                oldest_timestamp = timestamps[0]
                retry_after = max(1, int(oldest_timestamp + self.window_seconds - now + 0.999))
                return True, retry_after

            timestamps.append(now)
            return False, 0

    def reset(self):
        """Reset all tracked rate limit buckets."""
        with self._lock:
            self._requests.clear()


rate_limiter = SlidingWindowRateLimiter(
    max_requests=RATE_LIMIT_REQUESTS,
    window_seconds=RATE_LIMIT_WINDOW_SECONDS
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    print(f"Loading Whisper model '{WHISPER_MODEL}' on device '{WHISPER_DEVICE}' with compute_type '{WHISPER_COMPUTE_TYPE}'...")
    model = WhisperModel(
        WHISPER_MODEL,
        device=WHISPER_DEVICE,
        compute_type=WHISPER_COMPUTE_TYPE
    )
    print("Whisper model loaded successfully and ready for inference.")
    yield
    print("Shutting down Whisper service...")


app = FastAPI(
    title="Whisper Transcription API",
    description="FastAPI service for audio transcription using faster-whisper on CPU with rate limiting.",
    version="1.0.0",
    lifespan=lifespan
)


def extract_and_verify_auth(api_key: Optional[str], authorization: Optional[str], x_api_key: Optional[str]) -> str:
    """
    Validate API key if CONFIGURED_API_KEY is defined.
    Returns the key identifier to use for rate limiting.
    """
    token = None
    if api_key:
        token = api_key.strip()
    elif x_api_key:
        token = x_api_key.strip()
    elif authorization and authorization.startswith("Bearer "):
        token = authorization.split("Bearer ", 1)[1].strip()

    if CONFIGURED_API_KEY:
        if token != CONFIGURED_API_KEY:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing API key."
            )
        return token
    else:
        # If API_KEY is unset on the server, auth is disabled.
        # Group unauthenticated requests under a shared bucket, or track by supplied key if present.
        return token if token else "__anonymous__"


@app.get("/", tags=["Health"])
def health_check():
    """Health check endpoint to verify service, model availability, and rate limiting config."""
    return {
        "status": "ok",
        "model": WHISPER_MODEL,
        "device": WHISPER_DEVICE,
        "compute_type": WHISPER_COMPUTE_TYPE,
        "auth_enabled": bool(CONFIGURED_API_KEY),
        "rate_limit": {
            "max_requests": RATE_LIMIT_REQUESTS,
            "window_seconds": RATE_LIMIT_WINDOW_SECONDS
        },
        "model_loaded": model is not None
    }


@app.post("/transcribe", tags=["Transcription"])
def transcribe(
    file: UploadFile = File(..., description="Audio file to transcribe (e.g. mp3, wav, m4a, ogg)"),
    language: Optional[str] = Form(None, description="Optional ISO-639-1 language code (e.g. 'en', 'hi', 'es')"),
    beam_size: int = Form(1, description="Beam size for decoding (1=greedy fastest for CPU, 5=high accuracy)"),
    vad_filter: bool = Form(True, description="Enable Voice Activity Detection to filter music/silence"),
    api_key: Optional[str] = Form(None, description="Optional API key via form field"),
    authorization: Optional[str] = Header(None, description="Optional Bearer token"),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key", description="Optional API key header")
):
    """
    Transcribe an uploaded audio file.
    Runs in thread pool to prevent blocking ASGI event loop.
    Enforces authentication followed by sliding-window rate limiting per API key.
    Returns detected language, language probability, duration, full transcript, and segment timestamps.
    """
    # 1. Validate Authentication (401 if invalid)
    client_key = extract_and_verify_auth(api_key, authorization, x_api_key)

    # 2. Check Rate Limit (429 if limit exceeded)
    is_limited, retry_after = rate_limiter.check_rate_limit(client_key)
    if is_limited:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Maximum {RATE_LIMIT_REQUESTS} requests per {RATE_LIMIT_WINDOW_SECONDS} seconds.",
            headers={"Retry-After": str(retry_after)}
        )

    # 3. Ensure Whisper model is ready
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Whisper model is not loaded yet."
        )

    # Save uploaded file to a temporary file
    suffix = os.path.splitext(file.filename or "")[1] or ".tmp"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = tmp.name
        shutil.copyfileobj(file.file, tmp)

    try:
        segments, info = model.transcribe(
            tmp_path,
            language=language if language else None,
            beam_size=beam_size,
            vad_filter=vad_filter
        )

        segment_list: List[Dict[str, Any]] = []
        transcript_parts: List[str] = []

        for segment in segments:
            text = segment.text.strip()
            if text:
                transcript_parts.append(text)
                segment_list.append({
                    "id": segment.id,
                    "start": round(segment.start, 2),
                    "end": round(segment.end, 2),
                    "text": text
                })

        full_text = " ".join(transcript_parts).strip()

        return JSONResponse(
            content={
                "text": full_text,
                "language": info.language,
                "language_probability": round(info.language_probability, 4),
                "duration": round(info.duration, 2),
                "segments": segment_list
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Transcription failed: {str(e)}"
        )
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
