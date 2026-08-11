"""Model loading, verification, and warmup.

Design rule for this module: **it either loads the real model or it kills the
process.** There is no mock, no stub, and no silent fallback to a smaller model,
a different device, or a different dtype than the one that was asked for. Every
check below raises `ModelLoadError`, which the app's lifespan turns into a
non-zero exit. A service that starts successfully has, by construction, the real
weights on the requested device.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import torch
import transformers
from PIL import Image

from app.config import (
    EXPECTED_MODEL_CLASS,
    MIN_EXPECTED_PARAMS,
    Settings,
)
from app.logging_conf import get_logger, print_banner

log = get_logger(__name__)


class ModelLoadError(RuntimeError):
    """Fatal: the model could not be loaded as configured. Never recovered from."""


# ---------------------------------------------------------------------------
# Parameter counting
# ---------------------------------------------------------------------------


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    """Return (total, trainable) parameter counts, correcting for 4-bit packing.

    bitsandbytes stores 4-bit weights as packed uint8, so `Params4bit.numel()`
    reports half the logical parameter count. Without the correction a correctly
    loaded NF4 Cosmos-Reason2-2B reports ~1.2B params and trips the
    MIN_EXPECTED_PARAMS guard, which would look exactly like loading the wrong
    checkpoint. Two logical params per stored byte, hence the doubling.
    """
    total = trainable = 0
    for p in model.parameters():
        n = p.numel()
        if p.__class__.__name__ == "Params4bit":
            n *= 2
        total += n
        if p.requires_grad:
            trainable += n
    return total, trainable


# ---------------------------------------------------------------------------
# Device / dtype resolution
# ---------------------------------------------------------------------------


@dataclass
class Placement:
    device: str
    torch_dtype: torch.dtype
    gpu_name: str | None = None
    capability: str | None = None
    bf16_supported: bool = False
    notes: list[str] = field(default_factory=list)


def resolve_placement(settings: Settings) -> Placement:
    """Decide device and dtype, refusing combinations that cannot work.

    Refusals here are cheap; the same mistakes discovered 90 seconds into a model
    load, or worse, silently papered over, are not.
    """
    want = settings.device
    cuda_available = torch.cuda.is_available()

    if want == "cuda" and not cuda_available:
        raise ModelLoadError(
            "COSMOS_DEVICE=cuda but torch.cuda.is_available() is False. "
            "In Docker this almost always means the container was started without GPU "
            "access — check that the NVIDIA Container Toolkit is installed and that "
            "compose is passing the GPU through. Refusing to fall back to CPU silently; "
            "set COSMOS_DEVICE=cpu if that is genuinely what you want."
        )

    device = "cuda" if (want == "cuda" or (want == "auto" and cuda_available)) else "cpu"
    placement = Placement(device=device, torch_dtype=torch.float16)

    if device == "cuda":
        major, minor = torch.cuda.get_device_capability(0)
        placement.gpu_name = torch.cuda.get_device_name(0)
        placement.capability = f"sm_{major}{minor}"
        placement.bf16_supported = torch.cuda.is_bf16_supported()

        if settings.dtype == "bfloat16" and not placement.bf16_supported:
            raise ModelLoadError(
                f"COSMOS_DTYPE=bfloat16 but {placement.gpu_name} ({placement.capability}) "
                "has no bfloat16 support. T4 and every other Turing card are fp16-only. "
                "Set COSMOS_DTYPE=float16."
            )
        placement.torch_dtype = getattr(torch, settings.dtype)

        if settings.attn_impl == "flash_attention_2" and major < 8:
            raise ModelLoadError(
                f"COSMOS_ATTN_IMPL=flash_attention_2 requires sm_80 or newer; this GPU is "
                f"{placement.capability}. Use sdpa — it is what NVIDIA's own sample uses."
            )
    else:
        # fp16 matmul kernels largely do not exist on CPU ("addmm_impl_cpu_ not
        # implemented for 'Half'"). Silently running float32 would be a lie, so it
        # is logged as an explicit override and shown in the banner.
        if settings.dtype != "float32":
            placement.notes.append(
                f"dtype forced to float32 (requested {settings.dtype}; CPU has no fp16 kernels)"
            )
        placement.torch_dtype = torch.float32

        if settings.quant == "nf4":
            raise ModelLoadError(
                "COSMOS_QUANT=nf4 requires a CUDA device — bitsandbytes has no CPU 4-bit "
                "path. Set COSMOS_QUANT=none to run on CPU (expect ~5 GB of RAM and "
                "30-60 s per request)."
            )

    return placement


def _quantization_config(settings: Settings, placement: Placement) -> Any | None:
    if settings.quant != "nf4":
        return None
    return transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=placement.torch_dtype,
    )


# ---------------------------------------------------------------------------
# Load report
# ---------------------------------------------------------------------------


def _load_failure_message(source: str, exc: Exception) -> str:
    """Turn HuggingFace's auth errors into something actionable.

    nvidia/Cosmos-Reason2-* repos are gated (`gated: auto`). Without an accepted
    licence and a token the download fails with a bare 401, which reads like a
    network problem and sends people debugging the wrong thing entirely.
    """
    text = f"{type(exc).__name__}: {exc}"
    gated_markers = ("401", "gated", "GatedRepo", "Unauthorized", "authentication", "restricted")
    if any(marker.lower() in text.lower() for marker in gated_markers):
        return (
            f"Access denied loading {source!r}. The Cosmos-Reason2 repos are GATED on "
            "HuggingFace, so this is almost certainly a licence/token problem, not a "
            "network one. Two steps, both one-off:\n"
            f"  1. Visit https://huggingface.co/{source} while signed in and accept the "
            "NVIDIA Open Model License. Access is granted automatically — no waiting.\n"
            "  2. Create a read token at https://huggingface.co/settings/tokens and set "
            "HF_TOKEN in your .env.\n"
            f"Underlying error: {text}"
        )
    return f"Failed to load model from {source!r}: {text}"


@dataclass
class LoadReport:
    model_id: str
    model_source: str
    model_class: str
    device: str
    gpu_name: str | None
    capability: str | None
    bf16_supported: bool
    dtype: str
    quantization: str
    attention: str
    adapter: str | None
    total_params: int
    trainable_params: int
    adapter_params: int
    weights_bytes: int
    vram_allocated_bytes: int
    vram_total_bytes: int
    vision_token_min: int
    vision_token_max: int
    warmup_seconds: float
    warmup_tokens: int
    load_seconds: float
    notes: list[str]

    @staticmethod
    def _gib(n: int) -> str:
        return f"{n / 1024**3:.2f} GiB"

    def as_rows(self) -> list[tuple[str, str]]:
        pct = (self.trainable_params / self.total_params * 100) if self.total_params else 0.0
        device_str = self.device + (f"  ({self.gpu_name})" if self.gpu_name else "")
        cap_str = (
            f"{self.capability}   bf16 supported: {self.bf16_supported}"
            if self.capability
            else "n/a (CPU)"
        )
        vram = (
            f"{self._gib(self.vram_allocated_bytes)} allocated / "
            f"{self._gib(self.vram_total_bytes)} total"
            if self.vram_total_bytes
            else "n/a (CPU)"
        )
        rows = [
            ("model id", self.model_id),
            ("loaded from", self.model_source),
            ("model class", self.model_class),
            ("device", device_str),
            ("compute capability", cap_str),
            ("dtype", self.dtype),
            ("quantization", self.quantization),
            ("attention", self.attention),
            ("adapter", self.adapter or "NONE  (COSMOS_ADAPTER_ENABLED=false)"),
            ("total params", f"{self.total_params:,}"),
            ("trainable params", f"{self.trainable_params:,}  ({pct:.4f}%)"),
        ]
        if self.adapter_params:
            rows.append(("adapter params", f"{self.adapter_params:,}"))
        rows += [
            ("weights on device", self._gib(self.weights_bytes)),
            ("VRAM", vram),
            ("vision tokens", f"{self.vision_token_min} .. {self.vision_token_max}"),
            ("load time", f"{self.load_seconds:.1f} s"),
            ("warmup", f"OK — {self.warmup_tokens} tokens in {self.warmup_seconds:.2f} s"),
        ]
        rows += [("note", n) for n in self.notes]
        return rows

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_source": self.model_source,
            "model_class": self.model_class,
            "device": self.device,
            "gpu_name": self.gpu_name,
            "compute_capability": self.capability,
            "bf16_supported": self.bf16_supported,
            "dtype": self.dtype,
            "quantization": self.quantization,
            "attention": self.attention,
            "adapter": self.adapter,
            "params": {
                "total": self.total_params,
                "trainable": self.trainable_params,
                "adapter": self.adapter_params,
            },
            "memory": {
                "weights_bytes": self.weights_bytes,
                "vram_allocated_bytes": self.vram_allocated_bytes,
                "vram_total_bytes": self.vram_total_bytes,
            },
            "vision_tokens": {"min": self.vision_token_min, "max": self.vision_token_max},
            "load_seconds": round(self.load_seconds, 2),
            "warmup_seconds": round(self.warmup_seconds, 3),
            "notes": self.notes,
        }


@dataclass
class LoadedModel:
    model: Any
    processor: Any
    report: LoadReport
    settings: Settings


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _apply_vision_budget(processor: Any, settings: Settings) -> None:
    """Bound how many visual tokens an image or video frame may expand into.

    Despite the `shortest_edge`/`longest_edge` names these are *total pixel area*
    bounds (min_pixels/max_pixels), which is why the token budget is multiplied by
    32*32. Left unbounded a single 4K frame can produce tens of thousands of visual
    tokens and OOM a T4 on one request.
    """
    lo, hi = settings.vision_pixel_bounds()
    size = {"shortest_edge": lo, "longest_edge": hi}
    for attr in ("image_processor", "video_processor"):
        sub = getattr(processor, attr, None)
        if sub is not None:
            sub.size = dict(size)


def _load_adapter(model: Any, settings: Settings) -> tuple[Any, str, int]:
    """Attach a PEFT adapter. Any failure here is fatal, never a warning."""
    if not settings.adapter_path:
        raise ModelLoadError(
            "COSMOS_ADAPTER_ENABLED=true but COSMOS_ADAPTER_PATH is empty. Point it at a "
            "PEFT adapter directory or HF repo id, or set COSMOS_ADAPTER_ENABLED=false."
        )
    try:
        from peft import PeftModel
    except ImportError as exc:  # pragma: no cover - peft is a hard dependency
        raise ModelLoadError(f"peft is required to load an adapter: {exc}") from exc

    before, _ = count_parameters(model)
    try:
        model = PeftModel.from_pretrained(model, settings.adapter_path)
    except Exception as exc:
        raise ModelLoadError(
            f"Failed to load adapter from {settings.adapter_path!r}: {exc}"
        ) from exc

    if not hasattr(model, "peft_config"):
        raise ModelLoadError(
            "Adapter load returned a model with no peft_config — the adapter did not "
            "actually attach. Refusing to serve the base model while reporting an adapter."
        )

    after, _ = count_parameters(model)
    return model, settings.adapter_path, max(0, after - before)


def _warmup(model: Any, processor: Any, settings: Settings) -> tuple[float, int]:
    """Run one real inference before the service reports healthy.

    First-call CUDA kernel autotuning and lazy module init cost seconds. Paying
    that here keeps it out of the benchmark's first sample, where it would
    otherwise corrupt p95.
    """
    tokens = 8
    image = Image.new("RGB", (448, 448), color=(64, 96, 128))
    conversation = [
        {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
        # Media before text, matching the model's training inputs.
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Describe this image in one word."},
            ],
        },
    ]
    started = time.perf_counter()
    inputs = processor.apply_chat_template(
        conversation,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        model.generate(**inputs, max_new_tokens=tokens, do_sample=False)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter() - started, tokens


def load_model(settings: Settings) -> LoadedModel:
    """Load, verify, and warm up. Raises ModelLoadError on any problem."""
    started = time.perf_counter()
    placement = resolve_placement(settings)
    source = settings.model_source()

    log.info(
        "model.loading",
        source=source,
        device=placement.device,
        dtype=str(placement.torch_dtype),
        quant=settings.quant,
    )

    kwargs: dict[str, Any] = {
        "dtype": placement.torch_dtype,
        "attn_implementation": settings.attn_impl,
    }
    quant_config = _quantization_config(settings, placement)
    if quant_config is not None:
        kwargs["quantization_config"] = quant_config

    if placement.device == "cuda":
        # Pinned to a single device rather than "auto". "auto" would happily spill
        # layers to CPU RAM when VRAM is short, producing a service that starts
        # fine and then runs 20x slower for reasons nobody can see. An explicit
        # device map turns that case into an OOM at startup, which is the honest
        # failure.
        kwargs["device_map"] = {"": 0}

    try:
        model = transformers.Qwen3VLForConditionalGeneration.from_pretrained(source, **kwargs)
    except torch.cuda.OutOfMemoryError as exc:
        raise ModelLoadError(
            f"Out of VRAM loading {source}. fp16 weights are ~4.9 GB and do not fit in 6 GB "
            f"alongside the display buffer — set COSMOS_QUANT=nf4 for ~2 GB, or "
            f"COSMOS_DEVICE=cpu. Underlying error: {exc}"
        ) from exc
    except Exception as exc:
        raise ModelLoadError(_load_failure_message(source, exc)) from exc

    if placement.device == "cpu":
        model = model.to("cpu")
    model.eval()

    try:
        processor = transformers.AutoProcessor.from_pretrained(source)
    except Exception as exc:
        raise ModelLoadError(f"Failed to load processor from {source!r}: {exc}") from exc
    _apply_vision_budget(processor, settings)

    adapter_name: str | None = None
    adapter_params = 0
    if settings.adapter_enabled:
        model, adapter_name, adapter_params = _load_adapter(model, settings)
        model.eval()

    _verify(model, placement, settings)

    warmup_seconds, warmup_tokens = _warmup(model, processor, settings)

    lo_tok = min(settings.min_vision_tokens, settings.max_vision_tokens)
    hi_tok = max(settings.min_vision_tokens, settings.max_vision_tokens)
    total, trainable = count_parameters(model)
    report = LoadReport(
        model_id=settings.model_id,
        model_source=source,
        model_class=type(model).__name__,
        device=str(next(model.parameters()).device),
        gpu_name=placement.gpu_name,
        capability=placement.capability,
        bf16_supported=placement.bf16_supported,
        dtype=str(placement.torch_dtype),
        quantization=(
            "bitsandbytes-nf4 (double quant, "
            f"{str(placement.torch_dtype).replace('torch.', '')} compute)"
            if settings.quant == "nf4"
            else "none (full precision weights)"
        ),
        attention=settings.attn_impl,
        adapter=adapter_name,
        total_params=total,
        trainable_params=trainable,
        adapter_params=adapter_params,
        weights_bytes=model.get_memory_footprint(),
        vram_allocated_bytes=torch.cuda.memory_allocated() if placement.device == "cuda" else 0,
        vram_total_bytes=(
            torch.cuda.get_device_properties(0).total_memory if placement.device == "cuda" else 0
        ),
        vision_token_min=lo_tok,
        vision_token_max=hi_tok,
        warmup_seconds=warmup_seconds,
        warmup_tokens=warmup_tokens,
        load_seconds=time.perf_counter() - started,
        notes=placement.notes,
    )

    print_banner("COSMOS-EDGE-SERVE — model load report", report.as_rows())
    log.info("model.loaded", **report.as_dict())
    return LoadedModel(model=model, processor=processor, report=report, settings=settings)


def _verify(model: Any, placement: Placement, settings: Settings) -> None:
    """The anti-mock guards. Every one of these is fatal.

    These exist because a pipeline that silently falls back to a stub, a smaller
    checkpoint, or CPU is the single most expensive failure mode in this project —
    it produces plausible numbers that mean nothing.
    """
    # A PEFT-wrapped model reports PeftModel/PeftModelForCausalLM, so check the base.
    base = getattr(model, "base_model", None)
    inner = getattr(base, "model", base) if base is not None else model
    actual_class = type(inner).__name__
    if actual_class != EXPECTED_MODEL_CLASS:
        raise ModelLoadError(
            f"Loaded {actual_class}, expected {EXPECTED_MODEL_CLASS}. "
            f"{settings.model_source()!r} is not Cosmos-Reason2-2B (or a compatible "
            "Qwen3-VL checkpoint). Refusing to serve the wrong model."
        )

    total, _ = count_parameters(model)
    if total < MIN_EXPECTED_PARAMS:
        raise ModelLoadError(
            f"Loaded model has {total:,} parameters, expected at least "
            f"{MIN_EXPECTED_PARAMS:,}. Cosmos-Reason2-2B has 2,438,696,960. This is a "
            "different, smaller checkpoint — refusing to serve it."
        )

    actual_device = next(model.parameters()).device.type
    if actual_device != placement.device:
        raise ModelLoadError(
            f"Requested device {placement.device!r} but parameters landed on "
            f"{actual_device!r}. Refusing to serve from an unintended device."
        )

    devices = {p.device.type for p in model.parameters()}
    if len(devices) > 1:
        raise ModelLoadError(
            f"Model parameters are split across devices {sorted(devices)}. This means "
            "layers were offloaded to CPU because VRAM ran short, which would make every "
            "benchmark number meaningless. Use COSMOS_QUANT=nf4 or a larger GPU."
        )
