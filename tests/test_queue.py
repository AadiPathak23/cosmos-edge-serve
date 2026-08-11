"""Queue behaviour under saturation.

These are the tests that describe what the service does when it is overloaded,
which is precisely the regime the Phase 2 benchmark runs it in.
"""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path

import pytest

import app.queue as queue_module
from app.inference import InferenceRequest
from app.model import LoadedModel
from app.queue import InferenceQueue, QueueFullError, RequestTimeoutError


def _request() -> InferenceRequest:
    return InferenceRequest(
        media_path=Path("unused.jpg"),
        media_kind="image",
        prompt="What is this?",
        max_new_tokens=16,
        fps=4.0,
    )


async def test_requests_are_served_one_at_a_time(
    monkeypatch: pytest.MonkeyPatch, fake_loaded: LoadedModel, fake_result
) -> None:
    """The single-worker guarantee. If this breaks, VRAM sizing breaks with it."""
    concurrent = 0
    peak = 0
    lock = threading.Lock()

    def tracked(loaded, request):
        nonlocal concurrent, peak
        with lock:
            concurrent += 1
            peak = max(peak, concurrent)
        threading.Event().wait(0.05)
        with lock:
            concurrent -= 1
        return fake_result

    monkeypatch.setattr(queue_module, "run_inference", tracked)
    queue = InferenceQueue(fake_loaded, max_depth=8, timeout_s=10)
    queue.start()
    try:
        await asyncio.gather(*(queue.submit(_request(), f"r{i}") for i in range(4)))
    finally:
        await queue.stop()

    assert peak == 1, f"expected serialised execution, saw {peak} concurrent generate() calls"


async def test_rejects_with_queue_full_past_capacity(
    monkeypatch: pytest.MonkeyPatch, fake_loaded: LoadedModel, fake_result
) -> None:
    release = threading.Event()

    def blocking(loaded, request):
        release.wait(timeout=5)
        return fake_result

    monkeypatch.setattr(queue_module, "run_inference", blocking)
    queue = InferenceQueue(fake_loaded, max_depth=1, timeout_s=10)
    queue.start()

    try:
        # Occupies the worker.
        busy = asyncio.create_task(queue.submit(_request(), "busy"))
        await asyncio.sleep(0.1)
        # Fills the single queue slot.
        waiting = asyncio.create_task(queue.submit(_request(), "waiting"))
        await asyncio.sleep(0.1)

        # Nowhere left to put it — must be refused rather than silently buffered.
        with pytest.raises(QueueFullError, match="capacity"):
            await queue.submit(_request(), "rejected")

        release.set()
        await asyncio.gather(busy, waiting)
    finally:
        release.set()
        await queue.stop()


async def test_times_out_rather_than_hanging(
    monkeypatch: pytest.MonkeyPatch, fake_loaded: LoadedModel, fake_result
) -> None:
    release = threading.Event()

    def blocking(loaded, request):
        release.wait(timeout=5)
        return fake_result

    monkeypatch.setattr(queue_module, "run_inference", blocking)
    queue = InferenceQueue(fake_loaded, max_depth=4, timeout_s=0.2)
    queue.start()

    try:
        with pytest.raises(RequestTimeoutError, match="exceeded"):
            await queue.submit(_request(), "slow")
    finally:
        release.set()
        await queue.stop()


async def test_worker_failures_surface_to_the_caller(
    monkeypatch: pytest.MonkeyPatch, fake_loaded: LoadedModel
) -> None:
    """A crash in generate() must propagate, not be swallowed into a hung future."""

    def boom(loaded, request):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(queue_module, "run_inference", boom)
    queue = InferenceQueue(fake_loaded, max_depth=4, timeout_s=5)
    queue.start()

    try:
        with pytest.raises(RuntimeError, match="CUDA out of memory"):
            await queue.submit(_request(), "doomed")
    finally:
        await queue.stop()


async def test_worker_survives_a_failed_request(
    monkeypatch: pytest.MonkeyPatch, fake_loaded: LoadedModel, fake_result
) -> None:
    """One bad request must not take the worker down for everyone after it."""
    calls = {"n": 0}

    def flaky(loaded, request):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("first one fails")
        return fake_result

    monkeypatch.setattr(queue_module, "run_inference", flaky)
    queue = InferenceQueue(fake_loaded, max_depth=4, timeout_s=5)
    queue.start()

    try:
        with pytest.raises(RuntimeError):
            await queue.submit(_request(), "bad")
        result, _ = await queue.submit(_request(), "good")
        assert result.output_tokens == 42
    finally:
        await queue.stop()
