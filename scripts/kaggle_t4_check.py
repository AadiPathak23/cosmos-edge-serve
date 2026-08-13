#!/usr/bin/env python3
"""Free T4 validation run: prove the fp16 path and measure the real decode rate.

Run this on a **Kaggle T4** before spending a paid minute on EC2. It answers the
two questions Phase 2 cannot be sized without:

1. Does the fp16 path (`COSMOS_QUANT=none`) actually load and infer on a Turing
   T4? Everything so far has only ever run NF4 on an RTX 3060.
2. How many tokens per second does a T4 really decode? The laptop managed
   0.6-2.9 tok/s under NF4, which says nothing about a T4 — 4-bit dequant
   overhead dominates on a 6 GB laptop GPU. Benchmark profile B's runtime, and
   therefore its dollar cost, is a guess until this number exists.

Kaggle has no Docker daemon, so this runs the app **natively from a clone**
rather than in the container. It deliberately calls the service's own
`load_model()` and `run_inference()` rather than reimplementing them, so the
numbers come from the same code path the benchmark will use — including the
`torch.cuda.synchronize()` after `generate()`, without which every timing would
measure kernel launch instead of execution.

What it does NOT cover: the FastAPI layer, the bounded queue, and the container
itself. Those are covered by the pytest suite and by the Docker smoke test
respectively, and the queue is exactly why EC2 is still needed for Phase 2.

    python scripts/kaggle_t4_check.py
    python scripts/kaggle_t4_check.py --trials 5 --skip-video

See docs/KAGGLE.md for the notebook cells that set this up.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
from pathlib import Path

# Kaggle's "GPU T4 x2" accelerator exposes two cards; g4dn.xlarge has one, and
# `device_map={"": 0}` must land on a single device for these numbers to transfer
# to EC2 at all. This has to happen before torch is imported anywhere, which is
# why it sits above the imports rather than in main().
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from app.config import Settings  # noqa: E402
from app.inference import InferenceRequest, run_inference  # noqa: E402
from app.logging_conf import configure_logging, print_banner  # noqa: E402
from app.model import ModelLoadError, load_model  # noqa: E402

ASSETS = ROOT / "assets"

# A live model reports this, NOT the model card's 2,438,696,960: Qwen3-VL-2B ties
# lm_head to embed_tokens and model.parameters() deduplicates shared tensors, so
# the difference is exactly one embedding table (151936 x 2048 = 311,164,928).
# Verified on the RTX 3060 under NF4. The tying is architectural, so fp16 on a T4
# must report the same figure — if it does not, something is genuinely wrong.
EXPECTED_TOTAL_PARAMS = 2_127_532_032

# Profile A is the headline benchmark (comparable p50/p95, ~15 min sweep);
# profile B shows what full chain-of-thought reasoning actually costs.
PROFILES = {"A": 256, "B": 1024}

PROMPTS = {
    "image": "Describe this scene. Is it safe for a vehicle to proceed straight ahead?",
    "video": "What happens in this video? Describe the motion.",
}


def check_gpu(allow_any: bool) -> dict[str, object]:
    """Refuse to report numbers from the wrong accelerator.

    Kaggle hands out P100s and (occasionally) L4s as well as T4s. A P100 is
    Pascal and a whole generation away from the g4dn.xlarge target; publishing
    its decode rate as "the T4 number" would quietly invalidate the entire
    Phase 2 budget, which is the same class of mistake as measuring kernel
    launch latency instead of execution.
    """
    if not torch.cuda.is_available():
        raise SystemExit(
            "No CUDA device. On Kaggle: Notebook settings -> Accelerator -> GPU T4 x2, "
            "then restart the session."
        )

    name = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)
    capability = f"sm_{major}{minor}"
    total_vram = torch.cuda.get_device_properties(0).total_memory

    info: dict[str, object] = {
        "gpu_name": name,
        "capability": capability,
        "total_vram_bytes": total_vram,
        "visible_devices": torch.cuda.device_count(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "python": platform.python_version(),
        # Informational only. torch >= 2.6 reports emulated bf16 support on some
        # pre-Ampere cards, so this flag is NOT a reliable Turing test — the
        # compute capability is. Turing has no bf16 hardware regardless of what
        # this says, which is why the service pins fp16.
        "bf16_supported_flag": bool(torch.cuda.is_bf16_supported()),
    }

    is_t4 = "T4" in name and capability == "sm_75"
    if not is_t4 and not allow_any:
        raise SystemExit(
            f"Expected a Tesla T4 (sm_75), got {name} ({capability}).\n"
            "These numbers would not transfer to the g4dn.xlarge target, so refusing to\n"
            "record them. On Kaggle, switch the accelerator to 'GPU T4 x2' and restart.\n"
            "Pass --allow-any-gpu if you want a datapoint from this card anyway."
        )
    info["is_target_gpu"] = is_t4
    return info


def build_settings(max_vision_tokens: int) -> Settings:
    """fp16 on a 16 GB T4 — the Phase 2 configuration, not the laptop's.

    Passed as explicit kwargs rather than environment variables so the run is
    reproducible from the script alone and cannot be silently altered by a
    stray .env in the clone.
    """
    return Settings(
        quant="none",
        device="cuda",
        dtype="float16",
        # NF4 used only 1.59 GB of 6 GB on the laptop, so a 16 GB card at fp16
        # has ample room to raise the vision budget from the laptop default.
        max_vision_tokens=max_vision_tokens,
    )


def measure(loaded, media: Path, kind: str, max_new_tokens: int, trials: int) -> dict:
    """Run one leg and return median statistics over `trials` timed runs.

    The first run is discarded. `load_model()` already warms up on a synthetic
    448x448 image, but that never touches the video decode path, so the first
    real video request still pays a one-time torchcodec initialisation cost that
    would skew a 3-sample median.
    """
    request = InferenceRequest(
        media_path=media,
        media_kind=kind,
        prompt=PROMPTS[kind],
        max_new_tokens=max_new_tokens,
        fps=loaded.settings.video_fps,
    )

    print("    warmup...", end="", flush=True)
    first = run_inference(loaded, request)
    print(f" {first.output_tokens} tok in {first.generate_ms / 1000:.1f}s")

    samples: list = []
    for i in range(trials):
        result = run_inference(loaded, request)
        samples.append(result)
        print(
            f"    trial {i + 1}/{trials}: {result.output_tokens} tok, "
            f"{result.generate_ms / 1000:.1f}s, {result.tokens_per_second:.1f} tok/s"
        )

    last = samples[-1]
    return {
        "media_kind": kind,
        "max_new_tokens": max_new_tokens,
        "trials": trials,
        "tokens_per_second_median": round(
            statistics.median(s.tokens_per_second for s in samples), 2
        ),
        "tokens_per_second_min": round(min(s.tokens_per_second for s in samples), 2),
        "tokens_per_second_max": round(max(s.tokens_per_second for s in samples), 2),
        "generate_seconds_median": round(
            statistics.median(s.generate_ms for s in samples) / 1000, 2
        ),
        "preprocess_ms_median": round(statistics.median(s.preprocess_ms for s in samples), 1),
        "output_tokens_median": int(statistics.median(s.output_tokens for s in samples)),
        "input_tokens": last.input_tokens,
        "text_tokens": last.text_tokens,
        "visual_tokens": last.visual_tokens,
        # `truncated` at 256 tokens is expected and not a failure: the model emits
        # a long <think> block and profile A caps it deliberately.
        "truncated": last.truncated,
        "sample_answer": last.answer[:500],
        "sample_reasoning": last.reasoning[:500],
    }


def project_budget(results: list[dict]) -> list[tuple[str, str]]:
    """Turn measured per-request latency into the paid wall-clock it implies.

    Serialised, because the service runs ONE worker behind a bounded queue by
    design — concurrency changes the queue wait, not the total compute. So a
    sweep's wall clock is (requests x per-request compute) regardless of how many
    virtual users drive it, which is what makes this projection honest.
    """
    # 1 + 4 + 8 virtual users x 10 iterations each = 130 requests per sweep.
    requests_per_sweep = 130

    rows: list[tuple[str, str]] = []
    total_minutes = 0.0
    for profile, cap in PROFILES.items():
        legs = [r for r in results if r["max_new_tokens"] == cap]
        if not legs:
            continue
        per_request = statistics.mean(r["generate_seconds_median"] for r in legs)
        sweep_minutes = per_request * requests_per_sweep / 60
        total_minutes += sweep_minutes
        rows.append(
            (
                f"profile {profile} ({cap} tok)",
                f"{per_request:.1f} s/request  ->  ~{sweep_minutes:.0f} min for a "
                f"1/4/8 sweep of 10 iterations each ({requests_per_sweep} requests)",
            )
        )
    if rows:
        rows.append(("both sweeps", f"~{total_minutes:.0f} min of GPU time"))
        rows.append(
            ("at $0.32/hr spot", f"~${total_minutes / 60 * 0.32:.2f} for the benchmark itself")
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=3, help="timed runs per leg (default 3)")
    parser.add_argument("--max-vision-tokens", type=int, default=4096)
    parser.add_argument("--skip-video", action="store_true")
    parser.add_argument(
        "--allow-any-gpu",
        action="store_true",
        help="record numbers even if this is not a T4 (they will be flagged as off-target)",
    )
    parser.add_argument("--out", default="kaggle_t4_results.json")
    args = parser.parse_args()

    configure_logging("INFO")

    print("==> Environment")
    gpu = check_gpu(args.allow_any_gpu)
    print_banner("GPU", [(k, str(v)) for k, v in gpu.items()])

    image, video = ASSETS / "sample.jpg", ASSETS / "sample.mp4"
    if not image.exists():
        raise SystemExit(f"{image} missing. Run: python scripts/make_assets.py")

    print("\n==> Loading fp16 (this downloads ~5 GB the first time)")
    try:
        loaded = load_model(build_settings(args.max_vision_tokens))
    except ModelLoadError as exc:
        print(f"\nMODEL LOAD FAILED:\n{exc}", file=sys.stderr)
        return 1

    report = loaded.report.as_dict()
    problems: list[str] = []

    # The three things this run exists to confirm about the fp16 path.
    if report["dtype"] != "float16":
        problems.append(f"dtype is {report['dtype']}, expected float16")
    if report["quantization"] != "none":
        problems.append(f"quantization is {report['quantization']}, expected none")
    if report["params"]["total"] != EXPECTED_TOTAL_PARAMS:
        problems.append(
            f"total params {report['params']['total']:,} != the {EXPECTED_TOTAL_PARAMS:,} "
            "measured under NF4 on the 3060 — the tied-embedding dedup is architectural, "
            "so this should be identical across precisions. Worth investigating."
        )

    results: list[dict] = []
    legs = [(image, "image")] + ([] if args.skip_video else [(video, "video")])
    for media, kind in legs:
        if not media.exists():
            print(f"\n==> Skipping {kind}: {media} missing (run scripts/make_assets.py)")
            continue
        for profile, cap in PROFILES.items():
            print(f"\n==> {kind}, profile {profile} (max_new_tokens={cap})")
            try:
                results.append(measure(loaded, media, kind, cap, args.trials))
            except Exception as exc:  # noqa: BLE001 - report and continue to the next leg
                print(f"    FAILED: {type(exc).__name__}: {exc}")
                problems.append(f"{kind}/profile {profile} failed: {type(exc).__name__}: {exc}")

    print()
    print_banner(
        "DECODE RATE — Tesla T4, fp16",
        [
            (
                f"{r['media_kind']} @ {r['max_new_tokens']} tok",
                f"{r['tokens_per_second_median']:.1f} tok/s median "
                f"({r['tokens_per_second_min']:.1f}-{r['tokens_per_second_max']:.1f}), "
                f"{r['generate_seconds_median']:.1f}s, visual={r['visual_tokens']}",
            )
            for r in results
        ],
    )

    budget = project_budget(results)
    if budget:
        print_banner("PROJECTED PHASE 2 BUDGET (serialised — one worker by design)", budget)

    payload = {
        "gpu": gpu,
        "load_report": report,
        "measurements": results,
        "expected_total_params": EXPECTED_TOTAL_PARAMS,
        "problems": problems,
    }
    out = Path(args.out)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {out.resolve()}")

    print("\n" + "=" * 70)
    if problems:
        print(f"CHECK FAILED — {len(problems)} problem(s):")
        for p in problems:
            print(f"  - {p}")
        return 1
    if not results:
        print("CHECK FAILED — no measurements were taken.")
        return 1
    print("CHECK PASSED — fp16 path works on a T4 and the decode rate is recorded.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
