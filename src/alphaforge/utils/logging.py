"""Structured logging for every stage of the research pipeline.

The platform never uses bare ``print`` for operational output. All modules
obtain a logger through :func:`get_logger` and emit timestamped, levelled
records both to stderr and (optionally) to a rotating file sink.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from loguru import logger

_CONFIGURED = False
_DEFAULT_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)


def configure_logging(
    level: str = "INFO",
    log_file: str | None = None,
    json_format: bool = False,
) -> None:
    """Configure the global loguru sink. Idempotent."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    logger.remove()
    logger.add(
        sys.stderr,
        level=level.upper(),
        format=_DEFAULT_FORMAT,
        colorize=True,
        backtrace=False,
        diagnose=False,
    )
    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            str(path),
            level=level.upper(),
            serialize=json_format,
            rotation="20 MB",
            retention="14 days",
            compression="gz",
            enqueue=True,
        )
    _CONFIGURED = True


def get_logger(name: str, **context: Any):
    """Return a logger bound to a component name."""
    if not _CONFIGURED:
        configure_logging()
    bound = logger.bind(component=name)
    if context:
        bound = bound.bind(**context)
    return bound


class Timer:
    """Small context manager that logs the wall time of a pipeline stage."""

    def __init__(self, stage: str, log=None) -> None:
        self.stage = stage
        self.log = log or get_logger("timer")
        self.elapsed: float = 0.0

    def __enter__(self) -> Timer:
        import time

        self._t0 = time.perf_counter()
        self.log.info(f"[stage] {self.stage} - start")
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        import time

        self.elapsed = time.perf_counter() - self._t0
        if exc_type is None:
            self.log.info(f"[stage] {self.stage} - done in {self.elapsed:.2f}s")
        else:
            self.log.error(f"[stage] {self.stage} - failed after {self.elapsed:.2f}s: {exc}")
        return False


__all__ = ["configure_logging", "get_logger", "Timer", "logger"]
