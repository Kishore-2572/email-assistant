"""Tests for prompt template construction."""

from src.prompt_templates import (
    STRATEGY_MAP,
    build_advanced_prompt,
    build_baseline_prompt,
)


def test_advanced_prompt_has_system_and_user_messages() -> None:
    messages = build_advanced_prompt(
        intent="Test intent",
        key_facts=["Fact 1", "Fact 2"],
        tone="formal",
    )
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"


def test_advanced_prompt_contains_role_playing() -> None:
    messages = build_advanced_prompt(
        intent="Test intent",
        key_facts=["Fact 1"],
        tone="formal",
    )
    system_msg = messages[0]["content"]
    assert "senior executive communications specialist" in system_msg


def test_advanced_prompt_contains_few_shot_examples() -> None:
    messages = build_advanced_prompt(
        intent="Test intent",
        key_facts=["Fact 1"],
        tone="formal",
    )
    system_msg = messages[0]["content"]
    assert "Example 1:" in system_msg
    assert "Example 2:" in system_msg


def test_advanced_prompt_includes_all_facts() -> None:
    facts = ["First fact", "Second fact", "Third fact"]
    messages = build_advanced_prompt(
        intent="Test intent",
        key_facts=facts,
        tone="casual",
    )
    user_msg = messages[1]["content"]
    for fact in facts:
        assert fact in user_msg


def test_advanced_prompt_includes_tone() -> None:
    messages = build_advanced_prompt(
        intent="Test intent",
        key_facts=["Fact 1"],
        tone="urgent",
    )
    user_msg = messages[1]["content"]
    assert "urgent" in user_msg.lower()


def test_baseline_prompt_has_system_and_user_messages() -> None:
    messages = build_baseline_prompt(
        intent="Test intent",
        key_facts=["Fact 1"],
        tone="formal",
    )
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"


def test_baseline_prompt_is_simple() -> None:
    messages = build_baseline_prompt(
        intent="Test intent",
        key_facts=["Fact 1"],
        tone="formal",
    )
    system_msg = messages[0]["content"]
    # Baseline should NOT have role-playing or few-shot examples
    assert "senior executive communications specialist" not in system_msg
    assert "Example 1:" not in system_msg


def test_baseline_prompt_includes_facts() -> None:
    facts = ["Fact A", "Fact B"]
    messages = build_baseline_prompt(
        intent="Test intent",
        key_facts=facts,
        tone="formal",
    )
    user_msg = messages[1]["content"]
    for fact in facts:
        assert fact in user_msg


def test_strategy_map_contains_both_strategies() -> None:
    assert "advanced" in STRATEGY_MAP
    assert "baseline" in STRATEGY_MAP


def test_strategy_map_returns_callable() -> None:
    for strategy, func in STRATEGY_MAP.items():
        result = func("intent", ["fact"], "formal")
        assert isinstance(result, list)
        assert len(result) == 2
