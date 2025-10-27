import logging

from vct.utils.logging import _DEFAULT_FORMAT, get_logger


def test_get_logger_configures_handler_and_level() -> None:
    logger_name = "vct.test.logger"
    logger = logging.getLogger(logger_name)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    logger.handlers.clear()

    configured = get_logger(logger_name)
    assert configured.handlers, "Expected handler to be attached"
    handler = configured.handlers[0]
    assert isinstance(handler, logging.StreamHandler)
    assert handler.formatter is not None
    assert handler.formatter._fmt == _DEFAULT_FORMAT
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
    assert len(second.handlers) == 1
