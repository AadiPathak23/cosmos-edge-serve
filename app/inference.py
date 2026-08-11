"""The inference path: media in, tokens counted, reasoning split out.

Everything here is synchronous and runs on the single GPU worker thread. It must
never touch the event loop.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from PIL import Image

from app.model import LoadedModel

# The model is trained to emit its chain of thought inside <think>...</think>
# before answering. DOTALL because the reasoning spans many lines.
_THINK_RE = re.compile(r"\s*<think>(.*?)</think>\s*(.*)\s*", re.DOTALL)
_OPEN_THINK_RE = re.compile(r"\s*<think>(.*)", re.DOTALL)

_SYSTEM_PROMPT = "You are a helpful assistant."


@dataclass
class InferenceRequest:
    media_path: Path
    media_kind: Literal["image", "video"]
    prompt: str
    max_new_tokens: int
    fps: float


@dataclass
class InferenceResult:
    answer: str
    reasoning: str
    truncated: bool
    input_tokens: int
    text_tokens: int
    visual_tokens: int
    output_tokens: int
    preprocess_ms: float
    generate_ms: float

    @property
    def tokens_per_second(self) -> float:
        return (self.output_tokens / (self.generate_ms / 1000)) if self.generate_ms > 0 else 0.0


def _visual_token_ids(loaded: LoadedModel) -> set[int]:
    """Token ids that stand in for image/video patches in the prompt.

    Read from the model config first because that is authoritative and survives
    tokenizer naming changes; the tokenizer lookup is a fallback. Qwen3-VL expands
    one placeholder per visual token, so counting these ids in `input_ids` gives
    the exact visual token count rather than an estimate.
    """
    ids: set[int] = set()
    configs = [loaded.model.config, getattr(loaded.model.config, "text_config", None)]
    for cfg in configs:
        if cfg is None:
            continue
        for attr in ("image_token_id", "video_token_id"):
            value = getattr(cfg, attr, None)
            if isinstance(value, int) and value >= 0:
                ids.add(value)

    if not ids:
        tokenizer = loaded.processor.tokenizer
        unk = getattr(tokenizer, "unk_token_id", None)
        for name in ("<|image_pad|>", "<|video_pad|>"):
            value = tokenizer.convert_tokens_to_ids(name)
            if isinstance(value, int) and value >= 0 and value != unk:
                ids.add(value)
    return ids


def _build_conversation(request: InferenceRequest) -> list[dict[str, Any]]:
    """Assemble the chat turns.

    Media is listed *before* text. NVIDIA's sample calls this out explicitly — it
    matches how the model was trained, and reversing it measurably degrades output.
    """
    if request.media_kind == "image":
        media: dict[str, Any] = {
            "type": "image",
            "image": Image.open(request.media_path).convert("RGB"),
        }
    else:
        media = {"type": "video", "video": str(request.media_path.resolve())}

    return [
        {"role": "system", "content": [{"type": "text", "text": _SYSTEM_PROMPT}]},
        {"role": "user", "content": [media, {"type": "text", "text": request.prompt}]},
    ]


def split_reasoning(text: str) -> tuple[str, str]:
    """Split raw output into (answer, reasoning).

    Three cases: a complete <think> block, an unterminated one (generation hit the
    token cap mid-reasoning, so there is no answer yet), or no block at all.
    """
    match = _THINK_RE.fullmatch(text)
    if match:
        return match.group(2).strip(), match.group(1).strip()

    open_match = _OPEN_THINK_RE.fullmatch(text)
    if open_match:
        return "", open_match.group(1).strip()

    return text.strip(), ""


def run_inference(loaded: LoadedModel, request: InferenceRequest) -> InferenceResult:
    model, processor = loaded.model, loaded.processor

    preprocess_started = time.perf_counter()
    conversation = _build_conversation(request)

    template_kwargs: dict[str, Any] = {
        "tokenize": True,
        "add_generation_prompt": True,
        "return_dict": True,
        "return_tensors": "pt",
    }
    if request.media_kind == "video":
        # Only meaningful for video; passing it for an image confuses the processor.
        template_kwargs["fps"] = request.fps

    inputs = processor.apply_chat_template(conversation, **template_kwargs).to(model.device)
    preprocess_ms = (time.perf_counter() - preprocess_started) * 1000

    input_ids = inputs["input_ids"]
    input_tokens = int(input_ids.shape[-1])

    visual_ids = _visual_token_ids(loaded)
    if visual_ids:
        mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for token_id in visual_ids:
            mask |= input_ids == token_id
        visual_tokens = int(mask.sum().item())
    else:
        visual_tokens = 0

    generate_started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=request.max_new_tokens,
            # Greedy: the benchmark needs runs to be comparable, and sampling would
            # add variance to both output length and latency for no benefit here.
            do_sample=False,
        )
    if model.device.type == "cuda":
        # generate() is async on CUDA; without this the timing measures kernel
        # launch, not execution, and every latency number comes out far too low.
        torch.cuda.synchronize()
    generate_ms = (time.perf_counter() - generate_started) * 1000

    trimmed = generated[0][input_tokens:]
    output_tokens = int(trimmed.shape[-1])
    text = processor.batch_decode(
        [trimmed], skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    answer, reasoning = split_reasoning(text)

    return InferenceResult(
        answer=answer,
        reasoning=reasoning,
        truncated=output_tokens >= request.max_new_tokens,
        input_tokens=input_tokens,
        text_tokens=input_tokens - visual_tokens,
        visual_tokens=visual_tokens,
        output_tokens=output_tokens,
        preprocess_ms=preprocess_ms,
        generate_ms=generate_ms,
    )
