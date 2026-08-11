#!/usr/bin/env python3
"""Generate sample media for smoke tests.

Real test media is not committed — it would bloat the repo and the licensing of
any real driving footage is a question nobody needs. These synthetic clips are
enough to prove the plumbing works end to end: the image path, the video decode
path, and the visual token accounting.

Run inside the container (deps are already there):
    docker compose exec cosmos python scripts/make_assets.py
Or in a local venv with requirements.txt installed:
    python scripts/make_assets.py
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

ASSETS = Path(__file__).resolve().parents[1] / "assets"
WIDTH, HEIGHT = 640, 480
FPS = 4
SECONDS = 3


def _frame(index: int, total: int) -> Image.Image:
    """A crude road scene with a box that moves left to right across frames.

    Motion matters: a static clip would not exercise whether temporal information
    survives preprocessing, and the model would have nothing to reason about.
    """
    image = Image.new("RGB", (WIDTH, HEIGHT), (135, 178, 214))
    draw = ImageDraw.Draw(image)

    draw.rectangle([0, HEIGHT // 2, WIDTH, HEIGHT], fill=(78, 78, 82))
    for x in range(0, WIDTH, 80):
        draw.rectangle([x + 20, HEIGHT - 60, x + 60, HEIGHT - 50], fill=(230, 230, 230))

    progress = index / max(total - 1, 1)
    box_x = int(progress * (WIDTH - 120)) + 10
    draw.rectangle([box_x, HEIGHT // 2 - 40, box_x + 100, HEIGHT // 2 + 30], fill=(200, 60, 55))
    draw.rectangle([box_x + 15, HEIGHT // 2 - 30, box_x + 85, HEIGHT // 2 - 5], fill=(40, 44, 60))

    # The model card notes it recognises timestamps burned into the frame bottom.
    draw.text((10, HEIGHT - 20), f"t={index / FPS:.2f}s  frame {index}/{total}", fill=(255, 255, 0))
    return image


def make_image(path: Path) -> None:
    _frame(6, SECONDS * FPS).save(path, quality=92)
    print(f"wrote {path}  ({path.stat().st_size / 1024:.0f} KB)")


def make_video(path: Path) -> None:
    import av

    total = SECONDS * FPS
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=FPS)
        stream.width, stream.height = WIDTH, HEIGHT
        stream.pix_fmt = "yuv420p"
        for index in range(total):
            frame = av.VideoFrame.from_image(_frame(index, total))
            container.mux(stream.encode(frame))
        container.mux(stream.encode())
    print(f"wrote {path}  ({path.stat().st_size / 1024:.0f} KB, {total} frames @ {FPS} fps)")


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    make_image(ASSETS / "sample.jpg")
    make_video(ASSETS / "sample.mp4")


if __name__ == "__main__":
    main()
