"""Tests for the configuration module."""

import os
from unittest.mock import patch

import pytest

from src.config import SUPPORTED_TONES, setup_logging, validate_api_key


def test_supported_tones_contains_expected_values() -> None:
    expected = ["formal", "casual", "urgent", "empathetic", "persuasive", "apologetic"]
    assert SUPPORTED_TONES == expected


def test_validate_api_key_raises_when_empty() -> None:
    with patch("src.config.OPENAI_API_KEY", ""):
        with pytest.raises(ValueError, match="OPENAI_API_KEY is not set"):
            validate_api_key()


def test_validate_api_key_passes_when_set() -> None:
    with patch("src.config.OPENAI_API_KEY", "sk-test-key"):
        validate_api_key()  # Should not raise


def test_setup_logging_returns_logger() -> None:
    logger = setup_logging()
    assert logger.name == "email_assistant"
    assert len(logger.handlers) >= 1
