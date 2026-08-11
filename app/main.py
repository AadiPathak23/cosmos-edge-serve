"""FastAPI application: /infer, /health, /metrics.

The model is loaded once, in the lifespan handler, before uvicorn serves any
traffic. If the load fails the process exits non-zero — see `app.model` for why
there is deliberately no degraded mode.
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

import structlog
import torch
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app import metrics
from app.config import Settings, get_settings
from app.inference import InferenceRequest
from app.logging_conf import configure_logging, get_logger
from app.model import LoadedModel, ModelLoadError, load_model
from app.queue import InferenceQueue, QueueFullError, RequestTimeoutError, build_queue
from app.schemas import (
    ALLOWED_IMAGE_TYPES,
    ALLOWED_VIDEO_TYPES,
    EXTENSION_KINDS,
    HealthResponse,
    InferResponse,
    TimingMs,
    TokenCounts,
)

log = get_logger(__name__)

_UPLOAD_CHUNK = 1024 * 1024


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings: Settings = get_settings()
    configure_logging(settings.log_level)

    try:
        loaded = load_model(settings)
    except ModelLoadError as exc:
        # Loud and terminal. A half-working service that quietly serves the wrong
        # thing is worse than no service, so this is the one place we hard-exit.
        log.error("model.load_failed", error=str(exc))
        print(f"\nFATAL: {exc}\n", file=sys.stderr, flush=True)
        raise SystemExit(1) from exc

    app.state.loaded = loaded
    app.state.settings = settings
    app.state.queue = build_queue(loaded, settings)
    app.state.queue.start()

    metrics.MODEL_LOADED.set(1)
    _refresh_vram_gauge()

    try:
        yield
    finally:
        metrics.MODEL_LOADED.set(0)
        await app.state.queue.stop()
        # Clear the state so /health reports "loading" again rather than continuing
        # to advertise a model whose worker has already been shut down.
        app.state.loaded = None
        app.state.queue = None


app = FastAPI(
    title="cosmos-edge-serve",
    description=(
        "Inference service for nvidia/Cosmos-Reason2-2B. Send an image or short video "
        "plus a text prompt; get the model's reasoning and answer back."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


def _refresh_vram_gauge() -> None:
    if torch.cuda.is_available():
        metrics.VRAM_BYTES.set(torch.cuda.memory_allocated())


def _resolve_media_kind(file: UploadFile) -> Literal["image", "video"]:
    """Trust the declared content type, fall back to the extension.

    Clients (curl, k6) frequently send application/octet-stream, so rejecting on
    content type alone would break perfectly valid requests.
    """
    content_type = (file.content_type or "").split(";")[0].strip().lower()
    if content_type in ALLOWED_IMAGE_TYPES:
        return "image"
    if content_type in ALLOWED_VIDEO_TYPES:
        return "video"

    suffix = Path(file.filename or "").suffix.lower()
    if suffix in EXTENSION_KINDS:
        return EXTENSION_KINDS[suffix]

    raise HTTPException(
        status_code=415,
        detail=(
            f"Unsupported media type {content_type or 'unknown'!r} "
            f"(filename {file.filename!r}). Send jpg, png, webp, or mp4."
        ),
    )


async def _save_upload(file: UploadFile, max_bytes: int, suffix: str) -> Path:
    """Stream the upload to a temp file, aborting if it exceeds the cap.

    Streamed rather than `await file.read()` so a hostile or careless client
    cannot balloon the process's memory before the size check happens.
    """
    descriptor, name = tempfile.mkstemp(suffix=suffix)
    path = Path(name)
    written = 0
    try:
        with os.fdopen(descriptor, "wb") as handle:
            while chunk := await file.read(_UPLOAD_CHUNK):
                written += len(chunk)
                if written > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            f"Upload exceeds COSMOS_MAX_UPLOAD_MB "
                            f"({max_bytes // (1024 * 1024)} MB)."
                        ),
                    )
                handle.write(chunk)
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return path


@app.post(
    "/infer",
    response_model=InferResponse,
    responses={
        413: {"description": "Upload too large"},
        415: {"description": "Unsupported media type"},
        503: {"description": "Queue full — retry"},
        504: {"description": "Request timed out"},
    },
)
async def infer(
    request: Request,
    file: Annotated[UploadFile, File(description="jpg, png, webp, or mp4")],
    prompt: Annotated[str, Form(description="Question to ask about the media")],
    max_new_tokens: Annotated[int | None, Form()] = None,
    fps: Annotated[float | None, Form()] = None,
) -> InferResponse:
    settings: Settings = request.app.state.settings
    queue: InferenceQueue = request.app.state.queue

    request_id = uuid.uuid4().hex[:8]
    structlog.contextvars.bind_contextvars(request_id=request_id)
    started = time.perf_counter()

    if not prompt.strip():
        raise HTTPException(status_code=422, detail="prompt must not be empty.")

    kind = _resolve_media_kind(file)
    suffix = Path(file.filename or "").suffix.lower() or (".mp4" if kind == "video" else ".jpg")
    media_path = await _save_upload(file, settings.max_upload_mb * 1024 * 1024, suffix)

    inference_request = InferenceRequest(
        media_path=media_path,
        media_kind=kind,
        prompt=prompt,
        max_new_tokens=max_new_tokens or settings.max_new_tokens,
        fps=fps or settings.video_fps,
    )

    try:
        result, queue_wait_ms = await queue.submit(inference_request, request_id)
    except QueueFullError as exc:
        metrics.REQUESTS.labels(status="queue_full").inc()
        log.warning("request.rejected", reason="queue_full", queue_depth=queue.depth)
        return JSONResponse(
            status_code=503,
            content={"detail": str(exc), "request_id": request_id},
            headers={"Retry-After": "5"},
        )
    except RequestTimeoutError as exc:
        metrics.REQUESTS.labels(status="timeout").inc()
        log.warning("request.timeout", timeout_s=settings.request_timeout_s)
        return JSONResponse(
            status_code=504, content={"detail": str(exc), "request_id": request_id}
        )
    except Exception as exc:  # noqa: BLE001 - one place to turn worker faults into 500s
        metrics.REQUESTS.labels(status="error").inc()
        log.exception("request.failed", error=str(exc))
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc
    finally:
        media_path.unlink(missing_ok=True)
        structlog.contextvars.unbind_contextvars("request_id")

    total_ms = (time.perf_counter() - started) * 1000
    metrics.REQUESTS.labels(status="ok").inc()
    metrics.REQUEST_LATENCY.observe(total_ms / 1000)
    _refresh_vram_gauge()

    log.info(
        "request.completed",
        request_id=request_id,
        media_kind=kind,
        latency_ms=round(total_ms, 1),
        queue_wait_ms=round(queue_wait_ms, 1),
        preprocess_ms=round(result.preprocess_ms, 1),
        generate_ms=round(result.generate_ms, 1),
        input_tokens=result.input_tokens,
        text_tokens=result.text_tokens,
        visual_tokens=result.visual_tokens,
        output_tokens=result.output_tokens,
        tokens_per_second=round(result.tokens_per_second, 2),
        truncated=result.truncated,
        status="ok",
    )

    return InferResponse(
        answer=result.answer,
        reasoning=result.reasoning,
        truncated=result.truncated,
        tokens=TokenCounts(
            input=result.input_tokens,
            text=result.text_tokens,
            visual=result.visual_tokens,
            output=result.output_tokens,
        ),
        timing_ms=TimingMs(
            queue_wait=round(queue_wait_ms, 2),
            preprocess=round(result.preprocess_ms, 2),
            generate=round(result.generate_ms, 2),
            total=round(total_ms, 2),
        ),
        tokens_per_second=round(result.tokens_per_second, 2),
        request_id=request_id,
    )


@app.get("/health", response_model=HealthResponse)
async def health(request: Request) -> JSONResponse:
    loaded: LoadedModel | None = getattr(request.app.state, "loaded", None)
    queue: InferenceQueue | None = getattr(request.app.state, "queue", None)

    if loaded is None:
        # Reachable only in tests and during the brief window before lifespan
        # completes: in normal operation uvicorn does not accept connections until
        # the model is loaded and warmed up, so a refused connection (not a 503) is
        # the usual "still loading" signal. Hence the long start_period in compose.
        return JSONResponse(
            status_code=503,
            content=HealthResponse(
                status="loading", detail="Model is still loading."
            ).model_dump(),
        )

    _refresh_vram_gauge()
    return JSONResponse(
        status_code=200,
        content=HealthResponse(
            status="ok",
            detail="Model loaded and warmed up.",
            model=loaded.report.as_dict(),
            queue_depth=queue.depth if queue else 0,
        ).model_dump(),
    )


@app.get("/metrics", include_in_schema=False)
async def prometheus_metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/", include_in_schema=False)
async def root() -> dict[str, str]:
    return {
        "service": "cosmos-edge-serve",
        "model": get_settings().model_id,
        "docs": "/docs",
        "health": "/health",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=os.getenv("HOST", "0.0.0.0"),  # noqa: S104 - containerised, bound by compose
        port=int(os.getenv("PORT", "8000")),
        log_config=None,
    )
