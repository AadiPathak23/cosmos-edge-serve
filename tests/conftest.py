"""Shared test fixtures.

The whole suite runs without a GPU and without model weights: `load_model` and
`run_inference` are replaced with fakes. That keeps `pytest` a fast feedback loop
for the *service* — routing, validation, queueing, metrics, error mapping — while
anything that needs real weights lives in `scripts/smoke_test.py`.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.inference import InferenceResult
from app.model import LoadedModel, LoadReport


@pytest.fixture(autouse=True)
def _test_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin config for tests and drop the settings cache.

    `get_settings` is lru_cached, so without the clear the first test's config
    would leak into every later one.
    """
    monkeypatch.setenv("COSMOS_LOG_LEVEL", "WARNING")
    monkeypatch.setenv("COSMOS_MAX_UPLOAD_MB", "1")
    monkeypatch.setenv("COSMOS_MAX_QUEUE_DEPTH", "2")
    monkeypatch.setenv("COSMOS_REQUEST_TIMEOUT_S", "5")
    # Stop a developer's real .env from changing test outcomes.
    monkeypatch.setenv("COSMOS_QUANT", "none")
    monkeypatch.setenv("COSMOS_DEVICE", "cpu")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def fake_report() -> LoadReport:
    return LoadReport(
        model_id="nvidia/Cosmos-Reason2-2B",
        model_source="nvidia/Cosmos-Reason2-2B",
        model_class="Qwen3VLForConditionalGeneration",
        device="cpu",
        gpu_name=None,
        capability=None,
        bf16_supported=False,
        dtype="torch.float32",
        quantization="none (full precision weights)",
        attention="sdpa",
        adapter=None,
        total_params=2_438_696_960,
        trainable_params=0,
        adapter_params=0,
        weights_bytes=4_877_393_920,
        vram_allocated_bytes=0,
        vram_total_bytes=0,
        vision_token_min=256,
        vision_token_max=1024,
        warmup_seconds=1.5,
        warmup_tokens=8,
        load_seconds=12.0,
        notes=[],
    )


@pytest.fixture
def fake_loaded(fake_report: LoadReport) -> LoadedModel:
    return LoadedModel(
        model=SimpleNamespace(config=SimpleNamespace()),
        processor=SimpleNamespace(),
        report=fake_report,
        settings=get_settings(),
    )


@pytest.fixture
def fake_result() -> InferenceResult:
    return InferenceResult(
        answer="It is not safe to proceed; a vehicle is crossing.",
        reasoning="The red vehicle occupies the lane ahead.",
        truncated=False,
        input_tokens=1183,
        text_tokens=159,
        visual_tokens=1024,
        output_tokens=42,
        preprocess_ms=41.6,
        generate_ms=1400.0,
    )


@pytest.fixture
def client(
    monkeypatch: pytest.MonkeyPatch, fake_loaded: LoadedModel, fake_result: InferenceResult
) -> TestClient:
    """A client running the *real* lifespan, queue, and worker over fake weights.

    Faking at the `load_model` / `run_inference` seams rather than mocking the
    endpoint means the queue, the executor, metrics, and error mapping are all
    genuinely exercised.
    """
    import app.main as main_module
    import app.queue as queue_module

    monkeypatch.setattr(main_module, "load_model", lambda settings: fake_loaded)
    monkeypatch.setattr(queue_module, "run_inference", lambda loaded, request: fake_result)

    with TestClient(main_module.app) as test_client:
        yield test_client


@pytest.fixture
def jpeg_bytes() -> bytes:
    """A real, minimal JPEG. Pillow is a runtime dependency so this is free."""
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), (120, 140, 160)).save(buffer, format="JPEG")
    return buffer.getvalue()
