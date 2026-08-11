"""Parameter counting and the placement guards.

No GPU required — these cover the decisions made *before* any weights are touched.
"""

from __future__ import annotations

import pytest
import torch

from app.config import Settings
from app.model import (
    ModelLoadError,
    _load_failure_message,
    count_parameters,
    resolve_placement,
)


class _Params4bit(torch.nn.Parameter):
    """Stands in for bitsandbytes' packed 4-bit parameter.

    `count_parameters` dispatches on the class *name*, so this reproduces the real
    behaviour without needing a CUDA build of bitsandbytes installed.
    """

    __name__ = "Params4bit"


_Params4bit.__name__ = "Params4bit"
_Params4bit.__qualname__ = "Params4bit"


def test_counts_plain_parameters() -> None:
    module = torch.nn.Linear(10, 20, bias=False)  # 200 params
    total, trainable = count_parameters(module)
    assert total == 200
    assert trainable == 200


def test_frozen_parameters_are_not_trainable() -> None:
    module = torch.nn.Linear(10, 20, bias=False)
    module.weight.requires_grad = False
    total, trainable = count_parameters(module)
    assert total == 200
    assert trainable == 0


def test_unpacks_4bit_parameter_counts() -> None:
    """The guard that stops NF4 from looking like the wrong checkpoint.

    bitsandbytes packs two 4-bit values per stored byte, so a raw numel() on a
    correctly loaded NF4 model reports roughly half the real parameter count —
    which would trip MIN_EXPECTED_PARAMS and abort startup on a healthy model.
    """
    module = torch.nn.Module()
    module.weight = _Params4bit(torch.zeros(100, 1, dtype=torch.uint8), requires_grad=False)
    total, _ = count_parameters(module)
    assert total == 200, "packed 4-bit params must be counted as 2 logical params per byte"


def test_nf4_on_cpu_is_refused() -> None:
    settings = Settings(device="cpu", quant="nf4")
    with pytest.raises(ModelLoadError, match="requires a CUDA device"):
        resolve_placement(settings)


def test_cpu_forces_float32_and_says_so() -> None:
    """Silently swapping dtype would make the banner a lie, so it becomes a note."""
    settings = Settings(device="cpu", quant="none", dtype="float16")
    placement = resolve_placement(settings)
    assert placement.torch_dtype is torch.float32
    assert any("float32" in note for note in placement.notes)


def test_gated_repo_errors_explain_the_licence_step() -> None:
    """A bare 401 reads like a network fault and sends people debugging the wrong thing."""
    message = _load_failure_message(
        "nvidia/Cosmos-Reason2-2B", OSError("401 Client Error: Unauthorized for url ...")
    )
    assert "GATED" in message
    assert "accept the NVIDIA Open Model License" in message
    assert "HF_TOKEN" in message


def test_unrelated_load_errors_are_passed_through_verbatim() -> None:
    message = _load_failure_message("/models/cosmos", OSError("No such file or directory"))
    assert "GATED" not in message
    assert "No such file or directory" in message


@pytest.mark.skipif(torch.cuda.is_available(), reason="requires a machine without CUDA")
def test_explicit_cuda_without_cuda_is_refused() -> None:
    """No silent CPU fallback — that is the failure mode this project exists to avoid."""
    settings = Settings(device="cuda", quant="none")
    with pytest.raises(ModelLoadError, match="torch.cuda.is_available"):
        resolve_placement(settings)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
def test_bfloat16_is_refused_on_turing() -> None:
    if torch.cuda.is_bf16_supported():
        pytest.skip("this GPU supports bf16")
    settings = Settings(device="cuda", quant="none", dtype="bfloat16")
    with pytest.raises(ModelLoadError, match="no bfloat16 support"):
        resolve_placement(settings)
