"""Utility helpers for consistent logging configuration."""

import logging
import sys
from typing import Final

_DEFAULT_FORMAT: Final[str] = "[%(asctime)s] %(levelname)s %(name)s: %(message)s"


def get_logger(name: str) -> logging.Logger:
    """Return a logger configured with a standard stream handler."""

    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(_DEFAULT_FORMAT)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
