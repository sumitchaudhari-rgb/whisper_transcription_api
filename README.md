# Whisper Transcription API on Render

A fast, lightweight, CPU-optimized audio transcription web service powered by [FastAPI](https://fastapi.tiangolo.com/) and [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CTranslate2). Ready for one-click deployment to [Render](https://render.com).

---

## Features

- **Fast & Efficient**: Uses `faster-whisper` with `int8` quantization for up to 4x faster CPU transcription than original openai-whisper.
- **Single Model Load**: Loaded once during service startup lifecycle (`lifespan`), avoiding per-request initialization latency.
- **Rich Output**: Returns detected language, confidence score, total duration, full text, and word/phrase segment timestamps.
- **Configurable**: Model size (`tiny`, `base`, `small`, `medium`, `large-v3`), compute type, and optional API key configured via environment variables.
- **Dockerized**: Includes `ffmpeg` for wide audio format support (`.mp3`, `.wav`, `.m4a`, `.ogg`, `.flac`, etc.).
- **Render Ready**: Includes `render.yaml` Blueprint and dynamic `$PORT` handling.

---

## Endpoints

### `GET /` — Health Check
Confirms service status and active model configuration.

**Response:**
```json
{
  "status": "ok",
  "model": "base",
  "device": "cpu",
  "compute_type": "int8",
  "auth_enabled": false,
  "model_loaded": true
}
```

### `POST /transcribe` — Audio Transcription
Accepts `multipart/form-data` with an audio file.

**Parameters:**
- `file` (required): The audio file to transcribe.
- `language` (optional): 2-letter ISO code (e.g. `en`, `es`, `fr`). Whisper auto-detects if omitted.
- `api_key` (optional): Form-field API key if authentication is enabled.

**Headers (Optional):**
- `Authorization: Bearer <API_KEY>` or `X-API-Key: <API_KEY>`

**Response:**
```json
{
  "text": "Antigravity Whisper transcription test.",
  "language": "en",
  "language_probability": 0.9984,
  "duration": 2.5,
  "segments": [
    {
      "id": 1,
      "start": 0.0,
      "end": 2.5,
      "text": "Antigravity Whisper transcription test."
    }
  ]
}
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `WHISPER_MODEL` | `base` | Model size: `tiny`, `base`, `small`, `medium`, `large-v3` |
| `WHISPER_DEVICE` | `cpu` | Processing device: `cpu` or `cuda` |
| `WHISPER_COMPUTE_TYPE` | `int8` | Compute precision: `int8`, `float16`, `float32` |
| `RATE_LIMIT_REQUESTS` | `10` | Maximum allowed requests per window |
| `RATE_LIMIT_WINDOW_SECONDS` | `60` | Sliding window duration in seconds |
| `API_KEY` | `""` *(empty)* | Optional secret key for securing endpoints |
| `PORT` | `8000` | HTTP port (configured automatically by Render) |

---

## Rate Limiting

The service implements an in-memory sliding-window rate limiter:
- **Per-API Key Isolation**: Each authenticated API key has its own dedicated rate-limit bucket.
- **Shared Anonymous Bucket**: If `API_KEY` is not set (unauthenticated mode), all anonymous requests share a single rate-limit bucket to prevent server overload.
- **Order of Operations**: API key validation runs first (returning `401 Unauthorized` for invalid credentials), followed by the rate limit check.
- **HTTP 429 & Retry-After**: Exceeding the rate limit returns `HTTP 429 Too Many Requests` with a JSON error payload and a `Retry-After: <seconds>` header indicating when the client can retry.

### Rate Limit Configuration
Set these environment variables in your deployment or `.env`:
- `RATE_LIMIT_REQUESTS`: Number of requests allowed per window (default: `10`).
- `RATE_LIMIT_WINDOW_SECONDS`: Sliding window duration in seconds (default: `60`).

---

## Local Development

### 1. Prerequisites
- Python 3.10+
- `ffmpeg` installed on your system PATH

### 2. Setup
```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Run Locally
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

---

## Docker Local Build & Run

```bash
# Build the Docker image
docker build -t whisper-api .

# Run container on port 8000
docker run -p 8000:8000 -e WHISPER_MODEL=base whisper-api
```

---

## Deployment to Render

### Option A: Render Blueprint (Recommended)
1. Push this repository to GitHub.
2. In the [Render Dashboard](https://dashboard.render.com/), click **New +** → **Blueprint**.
3. Select your repository. Render will automatically read `render.yaml` and configure the service.
4. Set the `API_KEY` secret variable if desired, and click **Apply**.

### Option B: Manual Web Service Creation
1. Click **New +** → **Web Service**.
2. Select your repository.
3. Runtime: **Docker**.
4. Instance Type: **Starter** (512MB-1GB RAM works for `base`/`small` models).
5. Add Environment Variables under **Environment** tab:
   - `WHISPER_MODEL`: `base`
   - `WHISPER_DEVICE`: `cpu`
   - `WHISPER_COMPUTE_TYPE`: `int8`
