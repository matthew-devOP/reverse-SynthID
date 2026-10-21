"""
Centralized logging configuration for reverse-synthid.

Usage in any module:
    from src.utils.log import get_logger
    log = get_logger(__name__)
    log.info("Processing %d images", n)
"""

import logging
import sys

_CONFIGURED = False


def _configure_root():
    global _CONFIGURED
    if _CONFIGURED:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            "[%(levelname).1s %(name)s] %(message)s",
        )
    )
    root = logging.getLogger("src")
    root.setLevel(logging.INFO)
    root.addHandler(handler)
    root.propagate = False
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a logger for *name*, ensuring the root handler is attached."""
    _configure_root()
    return logging.getLogger(name)
