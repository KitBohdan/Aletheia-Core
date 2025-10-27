import json
import logging
from logging.handlers import RotatingFileHandler

import pytest

from vct.utils.logging import (
    CorrelationIdFilter,
    StructuredJsonFormatter,
    correlation_context,
    get_logger,
    get_correlation_id,
    set_correlation_id,
)


def test_get_logger_configures_handler_and_level() -> None:
    logger_name = "vct.test.logger"
    logger = logging.getLogger(logger_name)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    logger.handlers.clear()

    configured = get_logger(logger_name)
    assert configured.handlers, "Expected handler to be attached"
    stream_handler = next(
        (handler for handler in configured.handlers if isinstance(handler, logging.StreamHandler)),
        None,
    )
    assert stream_handler is not None
    assert isinstance(stream_handler.formatter, StructuredJsonFormatter)
    file_handler = next(
        (handler for handler in configured.handlers if isinstance(handler, RotatingFileHandler)),
        None,
    )
    assert file_handler is not None
    assert isinstance(file_handler.formatter, StructuredJsonFormatter)
    assert configured.level == logging.INFO


def test_get_logger_does_not_duplicate_handlers() -> None:
    logger_name = "vct.test.singleton"
    logger = logging.getLogger(logger_name)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    logger.handlers.clear()

    first = get_logger(logger_name)
    second = get_logger(logger_name)
    assert first is second
    assert {id(handler) for handler in first.handlers} == {id(handler) for handler in second.handlers}


def test_correlation_context_injects_identifier(caplog: pytest.LogCaptureFixture) -> None:
    logger = get_logger("vct.test.correlation")

    with caplog.at_level(logging.INFO):
        with correlation_context("corr-123"):
            logger.info("hello world")

    assert caplog.records
    record = caplog.records[0]
    assert getattr(record, "correlation_id", "") == "corr-123"


def test_structured_output_is_valid_json(caplog: pytest.LogCaptureFixture) -> None:
    logger = get_logger("vct.test.json")
    for handler in logger.handlers:
        handler.addFilter(CorrelationIdFilter())

    with caplog.at_level(logging.INFO):
        set_correlation_id("cid-456")
        logger.info("payload", extra={"user": "alice"})

    assert caplog.records
    record = caplog.records[0]
    formatter = StructuredJsonFormatter()
    rendered = formatter.format(record)
    parsed = json.loads(rendered)
    assert parsed["message"] == "payload"
    assert parsed["correlation_id"] == "cid-456"
    assert parsed["extra"]["user"] == "alice"


def test_get_correlation_id_defaults_to_none() -> None:
    set_correlation_id(None)
    assert get_correlation_id() is None
