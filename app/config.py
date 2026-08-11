"""Configuration, loaded from COSMOS_* environment variables.

Every knob the service has lives here. Defaults are tuned for the 6 GB dev laptop,
so an unconfigured `docker compose up` does the right thing locally and you only
override things when moving to a T4.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# One visual token covers a 32x32 pixel patch. The processor is sized in pixels,
# so token budgets get multiplied by this to become `shortest_edge`/`longest_edge`.
# Taken from NVIDIA's cosmos-reason2/scripts/inference_sample.py.
PIXELS_PER_TOKEN = 32**2

# The model class the service refuses to run without. Guards against a config typo
# silently loading some other (smaller, wrong) checkpoint.
EXPECTED_MODEL_CLASS = "Qwen3VLForConditionalGeneration"

# Cosmos-Reason2-2B reports 2,438,696,960 parameters. The guard uses a floor rather
# than equality so a LoRA adapter (which adds params) doesn't trip it.
MIN_EXPECTED_PARAMS = 2_000_000_000


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="COSMOS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        # pydantic v2 reserves the `model_` prefix for its own API and warns on any
        # field starting with it. We genuinely want `model_id` / `model_path`, so
        # the reservation is disabled rather than renaming the fields to something
        # less obvious.
        protected_namespaces=(),
    )

    # --- model source ------------------------------------------------------
    model_id: str = "nvidia/Cosmos-Reason2-2B"
    model_path: str | None = None
    """Local directory to load from instead of downloading. Used after the S3 pull."""

    # --- precision and placement -------------------------------------------
    quant: Literal["nf4", "none"] = "nf4"
    device: Literal["auto", "cuda", "cpu"] = "auto"
    dtype: Literal["float16", "bfloat16", "float32"] = "float16"
    attn_impl: str = "sdpa"

    # --- adapter ------------------------------------------------------------
    adapter_enabled: bool = False
    adapter_path: str | None = None

    # --- generation / vision budget ----------------------------------------
    max_new_tokens: int = Field(default=256, ge=1, le=8192)
    min_vision_tokens: int = Field(default=256, ge=4)
    max_vision_tokens: int = Field(default=1024, ge=4)
    video_fps: float = Field(default=4.0, gt=0, le=30)

    # --- serving ------------------------------------------------------------
    max_queue_depth: int = Field(default=32, ge=1)
    request_timeout_s: float = Field(default=300.0, gt=0)
    max_upload_mb: int = Field(default=64, ge=1)
    log_level: str = "INFO"

    @field_validator("model_path", "adapter_path", mode="before")
    @classmethod
    def _blank_to_none(cls, v: object) -> object:
        """Treat `COSMOS_MODEL_PATH=` in a .env file as unset rather than as "".

        Without this, an empty-but-present variable becomes an empty string and
        `from_pretrained("")` fails with a confusing error.
        """
        if isinstance(v, str) and not v.strip():
            return None
        return v

    @field_validator("log_level")
    @classmethod
    def _upper(cls, v: str) -> str:
        return v.upper()

    def model_source(self) -> str:
        """Where weights actually come from: local path wins over the HF repo id."""
        return self.model_path or self.model_id

    def vision_pixel_bounds(self) -> tuple[int, int]:
        """Vision token budget converted into the pixel bounds the processor wants."""
        lo = min(self.min_vision_tokens, self.max_vision_tokens)
        hi = max(self.min_vision_tokens, self.max_vision_tokens)
        return lo * PIXELS_PER_TOKEN, hi * PIXELS_PER_TOKEN


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
