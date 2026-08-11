"""Bounded request queue in front of a single GPU worker.

Why one worker: HuggingFace `generate()` holds the GPU for the whole request.
Running several concurrently multiplies KV-cache memory without buying real
parallelism, and on a 16 GB T4 that is how you turn a benchmark into an OOM.
Serialising through one worker also makes latency decompose cleanly into
queue wait plus compute, which is exactly the story the benchmark needs to tell.

Why bounded: an unbounded queue converts overload into unbounded latency, which
looks like a hang. A 503 with Retry-After is an honest answer.
"""

from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from app import metrics
from app.inference import InferenceRequest, InferenceResult, run_inference
from app.logging_conf import get_logger
from app.model import LoadedModel

log = get_logger(__name__)


class QueueFullError(Exception):
    """The queue is at capacity. Maps to 503 + Retry-After."""


class RequestTimeoutError(Exception):
    """The request exceeded COSMOS_REQUEST_TIMEOUT_S. Maps to 504."""


@dataclass
class _Job:
    request: InferenceRequest
    future: asyncio.Future
    request_id: str
    enqueued_at: float


class InferenceQueue:
    def __init__(self, loaded: LoadedModel, max_depth: int, timeout_s: float) -> None:
        self._loaded = loaded
        self._timeout_s = timeout_s
        self._queue: asyncio.Queue[_Job | None] = asyncio.Queue(maxsize=max_depth)
        # Exactly one thread. generate() releases the GIL inside torch kernels, so
        # running it off the event loop keeps /health and /metrics responsive while
        # the GPU is busy — which matters a lot when k6 is hammering the service.
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cosmos-gpu")
        self._worker: asyncio.Task | None = None

    @property
    def depth(self) -> int:
        return self._queue.qsize()

    def start(self) -> None:
        self._worker = asyncio.create_task(self._run(), name="cosmos-gpu-worker")

    async def stop(self) -> None:
        if self._worker is None:
            return
        await self._queue.put(None)
        await self._worker
        self._pool.shutdown(wait=True, cancel_futures=True)
        self._worker = None

    async def submit(
        self, request: InferenceRequest, request_id: str
    ) -> tuple[InferenceResult, float]:
        """Enqueue and await. Returns (result, queue_wait_ms)."""
        loop = asyncio.get_running_loop()
        job = _Job(
            request=request,
            future=loop.create_future(),
            request_id=request_id,
            enqueued_at=time.perf_counter(),
        )

        try:
            self._queue.put_nowait(job)
        except asyncio.QueueFull as exc:
            raise QueueFullError(
                f"Queue is at capacity ({self._queue.maxsize} requests). "
                "The GPU processes one request at a time; retry shortly."
            ) from exc

        metrics.QUEUE_DEPTH.set(self.depth)

        try:
            return await asyncio.wait_for(job.future, timeout=self._timeout_s)
        except TimeoutError as exc:
            # The worker checks for cancellation before starting, so a job that is
            # still queued gets dropped rather than burning GPU time on a caller
            # who has already given up. A job already running will finish and its
            # result is discarded.
            job.future.cancel()
            raise RequestTimeoutError(
                f"Request exceeded {self._timeout_s:.0f}s. Either the queue was deep or "
                "max_new_tokens was too high for this hardware."
            ) from exc

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        log.info("worker.started")

        while True:
            job = await self._queue.get()
            metrics.QUEUE_DEPTH.set(self.depth)

            if job is None:
                log.info("worker.stopping")
                return

            if job.future.cancelled():
                log.info("worker.skipped_cancelled", request_id=job.request_id)
                continue

            queue_wait_ms = (time.perf_counter() - job.enqueued_at) * 1000
            metrics.QUEUE_WAIT.observe(queue_wait_ms / 1000)

            try:
                result: InferenceResult = await loop.run_in_executor(
                    self._pool, run_inference, self._loaded, job.request
                )
            except Exception as exc:  # noqa: BLE001 - surfaced to the caller as a 500
                log.exception("worker.failed", request_id=job.request_id, error=str(exc))
                if not job.future.cancelled():
                    job.future.set_exception(exc)
                continue

            self._observe(result)
            if not job.future.cancelled():
                job.future.set_result((result, queue_wait_ms))

    @staticmethod
    def _observe(result: InferenceResult) -> None:
        metrics.PREPROCESS_SECONDS.observe(result.preprocess_ms / 1000)
        metrics.GENERATE_SECONDS.observe(result.generate_ms / 1000)
        metrics.INPUT_TOKENS.inc(result.input_tokens)
        metrics.VISUAL_TOKENS.inc(result.visual_tokens)
        metrics.OUTPUT_TOKENS.inc(result.output_tokens)
        if result.output_tokens:
            metrics.OUTPUT_TPS.observe(result.tokens_per_second)


def build_queue(loaded: LoadedModel, settings: Any) -> InferenceQueue:
    return InferenceQueue(
        loaded=loaded,
        max_depth=settings.max_queue_depth,
        timeout_s=settings.request_timeout_s,
    )
