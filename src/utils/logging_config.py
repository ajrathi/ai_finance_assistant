"""Structured logging setup for Finnie."""
from __future__ import annotations

import logging
import sys
from typing import Optional


def setup_logging(level: str = "INFO", name: Optional[str] = None) -> logging.Logger:
    """Configure root (or named) logger with a clean, structured format."""
    log_level = getattr(logging, level.upper(), logging.INFO)
    fmt = "%(asctime)s [%(levelname)s] %(name)s — %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(fmt, datefmt))

    root = logging.getLogger(name)
    if not root.handlers:
        root.addHandler(handler)
    root.setLevel(log_level)
    # Suppress noisy third-party loggers
    for noisy in ("httpx", "httpcore", "urllib3", "faiss"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    return root


def get_logger(name: str) -> logging.Logger:
    """Return a named logger. Call setup_logging() once at app startup."""
    return logging.getLogger(name)
