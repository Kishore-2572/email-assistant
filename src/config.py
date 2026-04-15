"""Configuration module for the Email Generation Assistant."""

import logging
import os
import sys

from dotenv import load_dotenv

load_dotenv()

# --- API Configuration ---
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o")

# --- Application Constants ---
SUPPORTED_TONES: list[str] = [
    "formal",
    "casual",
    "urgent",
    "empathetic",
    "persuasive",
    "apologetic",
]

STRATEGIES: list[str] = ["advanced", "baseline"]

# --- Input Validation Limits ---
MAX_INTENT_LENGTH: int = 500
MAX_FACT_LENGTH: int = 500
MAX_FACTS_COUNT: int = 20
MIN_FACTS_COUNT: int = 1

# --- API Retry Configuration ---
MAX_RETRIES: int = 1
RETRY_BASE_DELAY: float = 2.0


def validate_api_key() -> None:
    """Validate that the OpenAI API key is configured."""
    if not OPENAI_API_KEY:
        raise ValueError(
            "OPENAI_API_KEY is not set. "
            "Please set it in your .env file or environment variables."
        )


def setup_logging() -> logging.Logger:
    """Configure structured logging for the application.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger("email_assistant")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


logger = setup_logging()
