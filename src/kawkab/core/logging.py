"""Logging configuration using Loguru."""

from __future__ import annotations

import sys

from loguru import logger

from kawkab.core.paths import get_paths


def setup_logging(debug: bool = False) -> None:
    """Configure application logging.

    Args:
        debug: Enable debug-level logging
    """
    paths = get_paths()
    log_file = paths.logs / "kawkab.log"

    logger.remove()

    logger.add(
        sys.stderr,
        level="DEBUG" if debug else "INFO",
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
        colorize=True,
    )

    logger.add(
        log_file,
        level="DEBUG",
        format=("{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"),
        rotation="10 MB",
        retention="1 week",
        compression="zip",
    )


def get_logger(name: str | None = None):
    """Get a logger instance.

    Args:
        name: Optional logger name (for context)

    Returns:
        Configured logger
    """
    if name:
        return logger.bind(name=name)
    return logger
