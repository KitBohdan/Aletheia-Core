"""Utility helpers for consistent logging configuration."""

from __future__ import annotations

import json
import logging
import os
import sys
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Final, Iterator

_RESERVED_RECORD_FIELDS: Final = {
    "args",
    "asctime",
    "created",
    "exc_info",
    "exc_text",
    "filename",
    "funcName",
    "levelname",
    "levelno",
    "lineno",
    "module",
    "msecs",
    "message",
    "msg",
    "name",
    "pathname",
    "process",
    "processName",
    "relativeCreated",
    "stack_info",
    "thread",
    "threadName",
}

_CORRELATION_ID: ContextVar[str | None] = ContextVar("correlation_id", default=None)

_DEFAULT_LOG_PATH = Path(os.getenv("VCT_LOG_FILE", "/tmp/vct/app.log"))
_DEFAULT_MAX_BYTES = int(os.getenv("VCT_LOG_MAX_BYTES", "1048576"))
_DEFAULT_BACKUP_COUNT = int(os.getenv("VCT_LOG_BACKUP_COUNT", "5"))


def set_correlation_id(correlation_id: str | None) -> None:
    """Explicitly set the correlation identifier for subsequent log records."""

    _CORRELATION_ID.set(correlation_id)


def get_correlation_id() -> str | None:
    """Return the correlation identifier currently bound to the context."""

    return _CORRELATION_ID.get()


@contextmanager
def correlation_context(correlation_id: str) -> Iterator[None]:
    """Context manager that temporarily sets the correlation identifier."""

    token = _CORRELATION_ID.set(correlation_id)
    try:
        yield
    finally:
        _CORRELATION_ID.reset(token)


class CorrelationIdFilter(logging.Filter):
    """Inject the active correlation identifier into each log record."""

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: D401 - documented by class
        record.correlation_id = get_correlation_id() or "n/a"
        return True


class StructuredJsonFormatter(logging.Formatter):
    """Format log records as structured JSON strings."""

    @staticmethod
    def _json_safe(value: Any) -> Any:
        """Return a JSON-serializable representation of ``value``."""

        if value is None or isinstance(value, (str, int, float, bool)):
            return value

        if isinstance(value, datetime):
            # ``datetime`` is commonly attached to extras and should emit ISO strings.
            return value.astimezone(timezone.utc).isoformat()

        if isinstance(value, Path):
            return str(value)

        if isinstance(value, Mapping):
            return {str(key): StructuredJsonFormatter._json_safe(item) for key, item in value.items()}

        if isinstance(value, set):
            return [StructuredJsonFormatter._json_safe(item) for item in value]

        if isinstance(value, (bytes, bytearray)):
            return value.decode(errors="replace")

        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return [StructuredJsonFormatter._json_safe(item) for item in value]

        return str(value)

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "correlation_id": getattr(record, "correlation_id", "n/a"),
        }

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        extra = {
            key: value
            for key, value in record.__dict__.items()
            if key not in _RESERVED_RECORD_FIELDS and not key.startswith("_")
        }
        if extra:
            payload["extra"] = {
                key: StructuredJsonFormatter._json_safe(value)
                for key, value in extra.items()
            }

        return json.dumps(payload, ensure_ascii=False, default=str)


def _rotating_file_handler() -> RotatingFileHandler:
    _DEFAULT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        filename=_DEFAULT_LOG_PATH,
        maxBytes=_DEFAULT_MAX_BYTES,
        backupCount=_DEFAULT_BACKUP_COUNT,
    )
    handler.setFormatter(StructuredJsonFormatter())
    return handler


def _stream_handler() -> logging.StreamHandler:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(StructuredJsonFormatter())
    return handler


def _ensure_filter(logger: logging.Logger) -> None:
    if not any(isinstance(flt, CorrelationIdFilter) for flt in logger.filters):
        logger.addFilter(CorrelationIdFilter())


def get_logger(name: str) -> logging.Logger:
    """Return a logger configured for structured output and rotation."""

    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.addHandler(_stream_handler())
        logger.addHandler(_rotating_file_handler())
        logger.setLevel(logging.INFO)
    _ensure_filter(logger)
    return logger
