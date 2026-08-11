"""Endpoint contract tests: routing, validation, error mapping, metrics."""

from __future__ import annotations

from fastapi.testclient import TestClient

import app.main as main_module


def test_health_reports_the_load_report(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200

    body = response.json()
    assert body["status"] == "ok"
    # The point of /health is proving *what* loaded, not merely that something did.
    assert body["model"]["model_class"] == "Qwen3VLForConditionalGeneration"
    assert body["model"]["params"]["total"] == 2_438_696_960
    assert body["model"]["adapter"] is None


def test_health_is_503_before_the_model_loads() -> None:
    """Without the `with` block TestClient skips lifespan, so nothing is loaded."""
    bare = TestClient(main_module.app)
    response = bare.get("/health")
    assert response.status_code == 503
    assert response.json()["status"] == "loading"


def test_infer_returns_answer_reasoning_and_token_accounting(
    client: TestClient, jpeg_bytes: bytes
) -> None:
    response = client.post(
        "/infer",
        files={"file": ("sample.jpg", jpeg_bytes, "image/jpeg")},
        data={"prompt": "What is happening here?"},
    )
    assert response.status_code == 200

    body = response.json()
    assert body["answer"].startswith("It is not safe")
    assert body["reasoning"]
    assert body["truncated"] is False
    assert body["tokens"] == {"input": 1183, "text": 159, "visual": 1024, "output": 42}
    assert body["timing_ms"]["generate"] == 1400.0
    assert body["tokens_per_second"] == 30.0
    assert len(body["request_id"]) == 8


def test_infer_rejects_unsupported_media_type(client: TestClient) -> None:
    response = client.post(
        "/infer",
        files={"file": ("notes.txt", b"hello", "text/plain")},
        data={"prompt": "What is this?"},
    )
    assert response.status_code == 415
    assert "Unsupported media type" in response.json()["detail"]


def test_infer_falls_back_to_the_file_extension(client: TestClient, jpeg_bytes: bytes) -> None:
    """curl and k6 routinely send application/octet-stream; that must still work."""
    response = client.post(
        "/infer",
        files={"file": ("sample.jpg", jpeg_bytes, "application/octet-stream")},
        data={"prompt": "What is this?"},
    )
    assert response.status_code == 200


def test_infer_rejects_an_empty_prompt(client: TestClient, jpeg_bytes: bytes) -> None:
    response = client.post(
        "/infer",
        files={"file": ("sample.jpg", jpeg_bytes, "image/jpeg")},
        data={"prompt": "   "},
    )
    assert response.status_code == 422


def test_infer_rejects_an_oversized_upload(client: TestClient) -> None:
    """COSMOS_MAX_UPLOAD_MB is 1 in tests, so 2 MB must be refused."""
    response = client.post(
        "/infer",
        files={"file": ("big.mp4", b"\0" * (2 * 1024 * 1024), "video/mp4")},
        data={"prompt": "What is this?"},
    )
    assert response.status_code == 413
    assert "COSMOS_MAX_UPLOAD_MB" in response.json()["detail"]


def test_metrics_exposes_the_series_the_benchmark_depends_on(
    client: TestClient, jpeg_bytes: bytes
) -> None:
    client.post(
        "/infer",
        files={"file": ("sample.jpg", jpeg_bytes, "image/jpeg")},
        data={"prompt": "What is this?"},
    )
    body = client.get("/metrics").text

    for series in (
        "cosmos_requests_total",
        "cosmos_request_latency_seconds",
        "cosmos_queue_wait_seconds",
        "cosmos_generate_seconds",
        "cosmos_output_tokens_total",
        "cosmos_visual_tokens_total",
        "cosmos_queue_depth",
        "cosmos_model_loaded",
    ):
        assert series in body, f"{series} missing from /metrics"

    # Not an exact count: prometheus collectors are process-global, so successful
    # requests from earlier tests in this session accumulate here too.
    ok_samples = [line for line in body.splitlines() if 'cosmos_requests_total{status="ok"}' in line]
    assert ok_samples, 'no cosmos_requests_total{status="ok"} sample'
    assert float(ok_samples[0].rsplit(" ", 1)[1]) >= 1.0
    assert "cosmos_model_loaded 1.0" in body


def test_root_points_at_the_docs(client: TestClient) -> None:
    body = client.get("/").json()
    assert body["service"] == "cosmos-edge-serve"
    assert body["health"] == "/health"
