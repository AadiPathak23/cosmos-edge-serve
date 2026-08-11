"""Prometheus metrics.

Latency is deliberately split three ways — queue wait, preprocess, generate —
because under concurrency those tell completely different stories and a single
end-to-end histogram would hide which one is the bottleneck.
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

# Seconds. Buckets span sub-second (warm image, small output) through five
# minutes (long chain-of-thought on CPU), because this model's latency range
# really is that wide. The default prometheus buckets top out at 10s and would
# lump almost every real request into +Inf.
_LATENCY_BUCKETS = (0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 30, 60, 120, 300, float("inf"))

REQUESTS = Counter(
    "cosmos_requests_total",
    "Inference requests by terminal status.",
    ["status"],
)

REQUEST_LATENCY = Histogram(
    "cosmos_request_latency_seconds",
    "End-to-end latency: queue wait + preprocess + generate.",
    buckets=_LATENCY_BUCKETS,
)

QUEUE_WAIT = Histogram(
    "cosmos_queue_wait_seconds",
    "Time a request spent queued before a worker picked it up.",
    buckets=_LATENCY_BUCKETS,
)

PREPROCESS_SECONDS = Histogram(
    "cosmos_preprocess_seconds",
    "Media decode plus chat-template/tokenization time.",
    buckets=_LATENCY_BUCKETS,
)

GENERATE_SECONDS = Histogram(
    "cosmos_generate_seconds",
    "Time inside model.generate().",
    buckets=_LATENCY_BUCKETS,
)

OUTPUT_TPS = Histogram(
    "cosmos_output_tokens_per_second",
    "Decode throughput for a single request.",
    buckets=(1, 2, 5, 10, 15, 20, 25, 30, 40, 60, 100, float("inf")),
)

QUEUE_DEPTH = Gauge(
    "cosmos_queue_depth",
    "Requests currently waiting for the GPU worker.",
)

INPUT_TOKENS = Counter("cosmos_input_tokens_total", "Prompt tokens consumed (text + visual).")
VISUAL_TOKENS = Counter("cosmos_visual_tokens_total", "Visual tokens consumed.")
OUTPUT_TOKENS = Counter("cosmos_output_tokens_total", "Tokens generated.")

MODEL_LOADED = Gauge(
    "cosmos_model_loaded",
    "1 once the model is loaded and warmed up, 0 otherwise.",
)

VRAM_BYTES = Gauge(
    "cosmos_vram_allocated_bytes",
    "Torch-allocated VRAM. Excludes the caching allocator's reserved-but-free blocks.",
)
