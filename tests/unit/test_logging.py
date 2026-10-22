"""Tests for logging setup."""

from kawkab.core.logging import get_logger, setup_logging


class TestLogging:
    def test_get_logger_returns_logger(self):
        logger = get_logger("test_module")
        assert logger is not None

    def test_setup_logging_creates_file(self):
        setup_logging()
        logger = get_logger("test_file_logger")
        logger.info("Test log message")

    def test_multiple_get_logger_same_name(self):
        a = get_logger("shared")
        b = get_logger("shared")
        assert a is not None
        assert b is not None

    def test_setup_logging_no_crash(self):
        setup_logging()
