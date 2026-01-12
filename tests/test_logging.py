"""Tests for sda.logging module."""

from __future__ import annotations

import logging

from sda.logging import get_logger, quick_setup, setup_logging


class TestSetupLogging:
    def test_default_setup(self):
        setup_logging()
        log = get_logger("test")
        assert log is not None

    def test_debug_level(self):
        setup_logging(level=logging.DEBUG)
        log = get_logger("test")
        assert log is not None

    def test_string_level(self):
        setup_logging(level="WARNING")
        log = get_logger("test")
        assert log is not None

    def test_json_logs(self):
        setup_logging(json_logs=True)
        log = get_logger("test")
        assert log is not None


class TestQuickSetup:
    def test_default(self):
        quick_setup()
        log = get_logger("test")
        assert log is not None

    def test_debug_mode(self):
        quick_setup(debug=True)
        log = get_logger("test")
        assert log is not None

    def test_json_mode(self):
        quick_setup(json=True)
        log = get_logger("test")
        assert log is not None


class TestGetLogger:
    def test_named_logger(self):
        log = get_logger("my_module")
        assert log is not None

    def test_anonymous_logger(self):
        log = get_logger()
        assert log is not None

    def test_logger_methods(self):
        setup_logging()
        log = get_logger("test")
        # Check that standard logging methods exist
        assert hasattr(log, "info")
        assert hasattr(log, "debug")
        assert hasattr(log, "warning")
        assert hasattr(log, "error")
