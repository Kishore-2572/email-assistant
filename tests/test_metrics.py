"""Tests for evaluation metrics."""

import pytest

from src.evaluation.metrics import _calculate_automated_conciseness


# --- Automated conciseness score tests ---


def test_conciseness_optimal_ratio() -> None:
    """Ratio 0.8-1.2 should score 10.0."""
    ref = "word " * 100  # 100 words
    gen = "word " * 100  # ratio = 1.0
    assert _calculate_automated_conciseness(gen, ref) == 10.0


def test_conciseness_slightly_shorter() -> None:
    """Ratio 0.8 should still score 10.0."""
    ref = "word " * 100
    gen = "word " * 80  # ratio = 0.8
    assert _calculate_automated_conciseness(gen, ref) == 10.0


def test_conciseness_slightly_longer() -> None:
    """Ratio 1.2 should still score 10.0."""
    ref = "word " * 100
    gen = "word " * 120  # ratio = 1.2
    assert _calculate_automated_conciseness(gen, ref) == 10.0


def test_conciseness_very_short() -> None:
    """Very short text should score low."""
    ref = "word " * 100
    gen = "word " * 10  # ratio = 0.1
    score = _calculate_automated_conciseness(gen, ref)
    assert score < 3.0


def test_conciseness_very_long() -> None:
    """Very long text should score low."""
    ref = "word " * 100
    gen = "word " * 250  # ratio = 2.5
    score = _calculate_automated_conciseness(gen, ref)
    assert score == 0.0


def test_conciseness_empty_reference() -> None:
    """Empty reference should return 5.0 as default."""
    assert _calculate_automated_conciseness("some text", "") == 5.0


def test_conciseness_moderately_long() -> None:
    """Ratio 1.8 should score between 0 and 10."""
    ref = "word " * 100
    gen = "word " * 180  # ratio = 1.8
    score = _calculate_automated_conciseness(gen, ref)
    assert 0 < score < 10


def test_conciseness_score_never_negative() -> None:
    """Score should never be negative."""
    ref = "word " * 10
    gen = "word " * 500  # ratio = 50
    score = _calculate_automated_conciseness(gen, ref)
    assert score >= 0.0


def test_conciseness_score_capped_at_10() -> None:
    """Score should never exceed 10."""
    ref = "word " * 100
    gen = "word " * 100
    score = _calculate_automated_conciseness(gen, ref)
    assert score <= 10.0
