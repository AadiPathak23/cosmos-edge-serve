"""Settings parsing."""

from __future__ import annotations

from app.config import PIXELS_PER_TOKEN, Settings


def test_blank_env_values_are_treated_as_unset() -> None:
    """`COSMOS_MODEL_PATH=` in a .env must not become `from_pretrained("")`."""
    settings = Settings(model_path="", adapter_path="   ")
    assert settings.model_path is None
    assert settings.adapter_path is None


def test_model_path_overrides_the_hub_id() -> None:
    """Phase 2 pulls weights from S3 to a local dir; that must win."""
    assert Settings().model_source() == "nvidia/Cosmos-Reason2-2B"
    assert Settings(model_path="/models/cosmos").model_source() == "/models/cosmos"


def test_vision_budget_converts_tokens_to_pixel_area() -> None:
    low, high = Settings(min_vision_tokens=256, max_vision_tokens=1024).vision_pixel_bounds()
    assert low == 256 * PIXELS_PER_TOKEN
    assert high == 1024 * PIXELS_PER_TOKEN
    assert PIXELS_PER_TOKEN == 1024


def test_swapped_vision_bounds_are_normalised() -> None:
    """A min above max would otherwise produce a processor size that never resolves."""
    low, high = Settings(min_vision_tokens=4096, max_vision_tokens=256).vision_pixel_bounds()
    assert low < high


def test_log_level_is_case_insensitive() -> None:
    assert Settings(log_level="debug").log_level == "DEBUG"
