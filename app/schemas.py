"""Request and response models for the public API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

# Content types accepted by POST /infer. Anything else is a 415, never a 500.
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp"}
ALLOWED_VIDEO_TYPES = {"video/mp4"}

# Fallback for clients that send application/octet-stream or omit the type.
EXTENSION_KINDS: dict[str, Literal["image", "video"]] = {
    ".jpg": "image",
    ".jpeg": "image",
    ".png": "image",
    ".webp": "image",
    ".mp4": "video",
}


class TokenCounts(BaseModel):
    input: int = Field(description="Total prompt tokens fed to the model (text + visual).")
    text: int = Field(description="Prompt tokens that are not visual placeholders.")
    visual: int = Field(description="Tokens consumed by the image or video frames.")
    output: int = Field(description="Tokens generated, excluding the prompt.")


class TimingMs(BaseModel):
    queue_wait: float = Field(description="Milliseconds spent queued before a worker started.")
    preprocess: float = Field(description="Media decode plus chat-template/tokenization.")
    generate: float = Field(description="Milliseconds inside model.generate().")
    total: float = Field(description="Sum of the above, as observed by the endpoint.")


class InferResponse(BaseModel):
    answer: str = Field(description="The model's answer, with the <think> block stripped.")
    reasoning: str = Field(description="Contents of the <think> block. Empty if absent.")
    truncated: bool = Field(
        description=(
            "True if generation stopped at max_new_tokens rather than at an end-of-sequence "
            "token. A truncated response usually means the reasoning block ran long — raise "
            "max_new_tokens."
        )
    )
    tokens: TokenCounts
    timing_ms: TimingMs
    tokens_per_second: float = Field(description="Output tokens divided by generate time.")
    request_id: str


class HealthResponse(BaseModel):
    status: Literal["ok", "loading", "error"]
    detail: str
    model: dict[str, Any] | None = Field(
        default=None, description="The full load report, once the model is ready."
    )
    queue_depth: int = 0


class ErrorResponse(BaseModel):
    detail: str
    request_id: str | None = None
