"""Structured JSON logging.

One JSON object per line, so container logs pipe straight into anything without
a custom parser. The startup banner is the deliberate exception — it goes to
stdout as plain aligned text because a human reads it.
"""

from __future__ import annotations

import logging
import sys

import structlog


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper(), logging.INFO),
    )

    # transformers is extremely chatty at INFO and drowns out our own load report.
    logging.getLogger("transformers").setLevel(logging.WARNING)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        logger_factory=structlog.PrintLoggerFactory(sys.stdout),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str = "cosmos") -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)


def print_banner(title: str, rows: list[tuple[str, str]], width: int = 80) -> None:
    """Print an aligned key/value block to stdout.

    Used for the model load report. This is intentionally NOT structured logging:
    the whole point is that a human scanning container startup output cannot miss
    what actually got loaded.
    """
    label_width = max((len(k) for k, _ in rows), default=0)
    lines = [
        "=" * width,
        title,
        "-" * width,
        *(f"  {k.ljust(label_width)}   {v}" for k, v in rows),
        "=" * width,
    ]
    print("\n".join(lines), flush=True)
