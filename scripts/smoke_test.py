#!/usr/bin/env python3
"""End-to-end smoke test against a running service.

Standard library only — no requests, no httpx. That way it runs inside the
runtime container (which has no dev dependencies) as well as on your laptop.

    python scripts/smoke_test.py
    python scripts/smoke_test.py --url http://localhost:8000 --max-new-tokens 128

Unlike `pytest`, this needs the real model and real weights. It is the only thing
in the repo that verifies the model actually produces sensible output.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"


def _encode_multipart(fields: dict[str, str], file_path: Path) -> tuple[bytes, str]:
    boundary = f"----cosmos{uuid.uuid4().hex}"
    content_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
    parts: list[bytes] = []

    for name, value in fields.items():
        parts += [
            f"--{boundary}\r\n".encode(),
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
            f"{value}\r\n".encode(),
        ]

    parts += [
        f"--{boundary}\r\n".encode(),
        (
            f'Content-Disposition: form-data; name="file"; filename="{file_path.name}"\r\n'
            f"Content-Type: {content_type}\r\n\r\n"
        ).encode(),
        file_path.read_bytes(),
        b"\r\n",
        f"--{boundary}--\r\n".encode(),
    ]
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def wait_for_health(url: str, timeout_s: float) -> dict:
    """Poll /health until the model reports ready.

    A cold start downloads ~5 GB and then runs a warmup pass, so this legitimately
    takes many minutes the first time.
    """
    deadline = time.monotonic() + timeout_s
    last = ""
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{url}/health", timeout=10) as response:
                if response.status == 200:
                    return json.load(response)
        except urllib.error.HTTPError as exc:
            last = f"HTTP {exc.code}"
        except OSError as exc:
            last = str(exc)
        print(f"  waiting for service... ({last or 'not ready'})")
        time.sleep(5)
    raise SystemExit(f"Service not healthy after {timeout_s:.0f}s. Last: {last}")


def infer(url: str, file_path: Path, prompt: str, max_new_tokens: int | None) -> dict:
    fields = {"prompt": prompt}
    if max_new_tokens:
        fields["max_new_tokens"] = str(max_new_tokens)

    body, content_type = _encode_multipart(fields, file_path)
    request = urllib.request.Request(
        f"{url}/infer", data=body, headers={"Content-Type": content_type}, method="POST"
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.load(response)


def _report(label: str, result: dict) -> None:
    tokens, timing = result["tokens"], result["timing_ms"]
    print(f"\n--- {label} " + "-" * (60 - len(label)))
    print(f"  answer     : {result['answer'][:300] or '(empty — output was all reasoning)'}")
    if result["reasoning"]:
        print(f"  reasoning  : {result['reasoning'][:200]}...")
    print(f"  truncated  : {result['truncated']}")
    print(
        f"  tokens     : in={tokens['input']} (text={tokens['text']}, "
        f"visual={tokens['visual']})  out={tokens['output']}"
    )
    print(
        f"  timing     : queue={timing['queue_wait']:.0f}ms  pre={timing['preprocess']:.0f}ms  "
        f"gen={timing['generate']:.0f}ms  total={timing['total']:.0f}ms"
    )
    print(f"  throughput : {result['tokens_per_second']:.1f} tok/s")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--health-timeout", type=float, default=1800)
    parser.add_argument("--skip-video", action="store_true")
    args = parser.parse_args()

    url = args.url.rstrip("/")

    print(f"==> Waiting for {url}/health")
    health = wait_for_health(url, args.health_timeout)
    model = health.get("model", {})
    print(
        f"    ready: {model.get('model_class')} on {model.get('device')} "
        f"({model.get('gpu_name') or 'CPU'}), {model.get('quantization')}, "
        f"{model.get('params', {}).get('total', 0):,} params"
    )

    image = ASSETS / "sample.jpg"
    video = ASSETS / "sample.mp4"
    if not image.exists():
        raise SystemExit(f"{image} missing. Run: python scripts/make_assets.py")

    failures = 0

    print("\n==> Image request")
    result = infer(
        url,
        image,
        "Describe this scene. Is it safe for a vehicle to proceed straight ahead?",
        args.max_new_tokens,
    )
    _report("image", result)
    if result["tokens"]["visual"] <= 0:
        print("  FAIL: visual token count is 0 — the image was not tokenised as vision input.")
        failures += 1
    if not (result["answer"] or result["reasoning"]):
        print("  FAIL: model returned nothing at all.")
        failures += 1

    if not args.skip_video:
        if not video.exists():
            print(f"\n==> Skipping video: {video} missing. Run scripts/make_assets.py")
        else:
            print("\n==> Video request")
            result = infer(
                url, video, "What happens in this video? Describe the motion.", args.max_new_tokens
            )
            _report("video", result)
            if result["tokens"]["visual"] <= 0:
                print("  FAIL: visual token count is 0 for video.")
                failures += 1

    print("\n" + "=" * 70)
    if failures:
        print(f"SMOKE TEST FAILED — {failures} problem(s) above.")
        return 1
    print("SMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
