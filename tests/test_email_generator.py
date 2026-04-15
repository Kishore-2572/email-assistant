"""Tests for email generation core logic."""

import pytest

from src.email_generator import (
    EmailInput,
    parse_email_response,
    sanitize_input,
    validate_input,
)


# --- sanitize_input tests ---


def test_sanitize_strips_whitespace() -> None:
    assert sanitize_input("  hello world  ") == "hello world"


def test_sanitize_escapes_html() -> None:
    result = sanitize_input("<script>alert('xss')</script>")
    assert "<script>" not in result
    assert "&lt;script&gt;" in result


def test_sanitize_escapes_ampersand() -> None:
    assert "&amp;" in sanitize_input("A & B")


# --- validate_input tests ---


def test_validate_empty_intent() -> None:
    inp = EmailInput(intent="", key_facts=["fact"], tone="formal")
    errors = validate_input(inp)
    assert any("Intent" in e for e in errors)


def test_validate_whitespace_intent() -> None:
    inp = EmailInput(intent="   ", key_facts=["fact"], tone="formal")
    errors = validate_input(inp)
    assert any("Intent" in e for e in errors)


def test_validate_long_intent() -> None:
    inp = EmailInput(intent="x" * 501, key_facts=["fact"], tone="formal")
    errors = validate_input(inp)
    assert any("maximum length" in e for e in errors)


def test_validate_no_facts() -> None:
    inp = EmailInput(intent="test", key_facts=[], tone="formal")
    errors = validate_input(inp)
    assert any("fact" in e.lower() for e in errors)


def test_validate_empty_string_facts_filtered() -> None:
    inp = EmailInput(intent="test", key_facts=["", "  ", "real fact"], tone="formal")
    errors = validate_input(inp)
    assert len(errors) == 0  # "real fact" is valid after filtering


def test_validate_too_many_facts() -> None:
    inp = EmailInput(intent="test", key_facts=["f"] * 21, tone="formal")
    errors = validate_input(inp)
    assert any("Maximum" in e for e in errors)


def test_validate_invalid_tone() -> None:
    inp = EmailInput(intent="test", key_facts=["fact"], tone="angry")
    errors = validate_input(inp)
    assert any("Unsupported tone" in e for e in errors)


def test_validate_valid_input() -> None:
    inp = EmailInput(intent="Follow up", key_facts=["Met Sarah"], tone="formal")
    errors = validate_input(inp)
    assert len(errors) == 0


def test_validate_tone_case_insensitive() -> None:
    inp = EmailInput(intent="test", key_facts=["fact"], tone="FORMAL")
    errors = validate_input(inp)
    assert len(errors) == 0


# --- parse_email_response tests ---


def test_parse_with_subject_prefix() -> None:
    response = "Subject: Test Subject\n\nDear Sir,\n\nThis is the body."
    subject, body = parse_email_response(response)
    assert subject == "Test Subject"
    assert "Dear Sir" in body


def test_parse_subject_case_insensitive() -> None:
    response = "subject: My Subject\n\nBody text here."
    subject, body = parse_email_response(response)
    assert subject == "My Subject"


def test_parse_fallback_no_subject_prefix() -> None:
    response = "This is a complete email without subject prefix. It has body too."
    subject, body = parse_email_response(response)
    assert subject == "This is a complete email without subject prefix"
    assert body == response


def test_parse_empty_body() -> None:
    response = "Subject: Just a subject"
    subject, body = parse_email_response(response)
    assert subject == "Just a subject"
    assert body == ""
