# syntax=docker/dockerfile:1.7
#
# Multi-stage build for cosmos-edge-serve.
#
# Size expectations, stated honestly: torch 2.9.0+cu128 bundles roughly 4 GB of
# nvidia-* CUDA wheels, and there is no way around that for a GPU service. The
# realistic finished image is 6-8 GB. What multi-stage actually buys here is
# leaving pip, setuptools, the wheel cache, and apt lists behind in the builder.
#
# What keeps this from being a 12 GB image is the thing that matters most:
# **model weights are never baked in.** They live in the `cosmos-models` volume
# mounted at /models (HF_HOME), so the image stays the same size whether you are
# serving the 2B or nothing at all.

ARG CUDA_IMAGE=nvidia/cuda:12.8.1-runtime-ubuntu24.04

# ---------------------------------------------------------------------------
# Stage 1: builder — resolve and install dependencies into a venv
# ---------------------------------------------------------------------------
# Built on the *same* base as the runtime stage on purpose. A venv records an
# absolute path to its interpreter, so building it against a different Python
# (say python:3.12-slim) and copying it across would produce a venv pointing at
# a binary that does not exist in the final image.
FROM ${CUDA_IMAGE} AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-venv \
        python3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copied on its own so the (slow, ~4 GB) dependency layer is cached and only
# reinvalidated when the pins actually change, not on every source edit.
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel \
    && pip install -r requirements.txt

# Strip test suites and bytecode caches from site-packages. Saves a few hundred
# MB and nothing at runtime depends on them.
RUN find /opt/venv -type d -name "__pycache__" -prune -exec rm -rf {} + \
    && find /opt/venv -type d -name "tests" -prune -exec rm -rf {} + \
    && find /opt/venv -type d -name "test" -prune -exec rm -rf {} +

# ---------------------------------------------------------------------------
# Stage 2: runtime
# ---------------------------------------------------------------------------
FROM ${CUDA_IMAGE} AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    # Weights land here, and here is a mounted volume. This single line is what
    # keeps ~5 GB of safetensors out of the image.
    HF_HOME=/models

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        # torchcodec links against FFmpeg's shared libraries at import time; without
        # these, video requests fail with an opaque "could not load libtorchcodec"
        # rather than anything actionable. Ubuntu 24.04 ships FFmpeg 6.x, which
        # torchcodec 0.9 supports.
        ffmpeg \
        # Used by the container healthcheck.
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

# Run as a non-root user. /models must be writable by it because the first start
# downloads weights into the volume.
RUN useradd --create-home --uid 1000 cosmos \
    && mkdir -p /models \
    && chown -R cosmos:cosmos /models

WORKDIR /srv
COPY --chown=cosmos:cosmos app/ ./app/
COPY --chown=cosmos:cosmos scripts/ ./scripts/

USER cosmos
EXPOSE 8000

# start_period is generous because a cold start downloads ~5 GB and then runs a
# warmup inference pass. The container is correctly reported unhealthy until
# /health returns 200, which happens only after warmup succeeds.
HEALTHCHECK --interval=30s --timeout=10s --start-period=900s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

# --no-access-log because app/main.py already emits one structured JSON line per
# request with far more detail; uvicorn's access log would just duplicate it.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
