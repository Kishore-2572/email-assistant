"""Core email generation logic using OpenAI API."""

import html
import time
from dataclasses import dataclass

from openai import APIConnectionError, APITimeoutError, OpenAI, RateLimitError

from src.config import (
    MAX_FACT_LENGTH,
    MAX_FACTS_COUNT,
    MAX_INTENT_LENGTH,
    MAX_RETRIES,
    MIN_FACTS_COUNT,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    RETRY_BASE_DELAY,
    SUPPORTED_TONES,
    logger,
    validate_api_key,
)
from src.prompt_templates import STRATEGY_MAP


@dataclass
class EmailInput:
    """Input data for email generation."""

    intent: str
    key_facts: list[str]
    tone: str


@dataclass
class GeneratedEmail:
    """Output from the email generation model."""

    subject: str
    body: str
    strategy: str
    model: str
    prompt_tokens: int
    completion_tokens: int


def sanitize_input(text: str) -> str:
    """Sanitize user input by escaping HTML and trimming whitespace.

    Args:
        text: Raw user input string.

    Returns:
        Sanitized string.
    """
    return html.escape(text.strip())


def validate_input(email_input: EmailInput) -> list[str]:
    """Validate email generation input fields.

    Args:
        email_input: The input to validate.

    Returns:
        List of validation error messages. Empty if valid.
    """
    errors: list[str] = []

    if not email_input.intent or not email_input.intent.strip():
        errors.append("Intent is required and cannot be empty.")
    elif len(email_input.intent.strip()) > MAX_INTENT_LENGTH:
        errors.append(
            f"Intent exceeds maximum length of {MAX_INTENT_LENGTH} characters."
        )

    facts = [f for f in email_input.key_facts if f and f.strip()]
    if len(facts) < MIN_FACTS_COUNT:
        errors.append(f"At least {MIN_FACTS_COUNT} key fact is required.")
    elif len(facts) > MAX_FACTS_COUNT:
        errors.append(
            f"Maximum of {MAX_FACTS_COUNT} key facts allowed."
        )
    for i, fact in enumerate(facts, 1):
        if len(fact.strip()) > MAX_FACT_LENGTH:
            errors.append(
                f"Key fact #{i} exceeds maximum length of {MAX_FACT_LENGTH} characters."
            )

    if email_input.tone.lower() not in SUPPORTED_TONES:
        errors.append(
            f"Unsupported tone '{email_input.tone}'. "
            f"Supported tones: {', '.join(SUPPORTED_TONES)}"
        )

    return errors


def parse_email_response(response_text: str) -> tuple[str, str]:
    """Parse the LLM response to extract subject and body.

    Args:
        response_text: Raw text response from the LLM.

    Returns:
        Tuple of (subject, body).
    """
    text = response_text.strip()

    if text.lower().startswith("subject:"):
        parts = text.split("\n", 1)
        subject = parts[0].replace("Subject:", "").replace("subject:", "").strip()
        body = parts[1].strip() if len(parts) > 1 else ""
        return subject, body

    # Fallback: use first sentence as subject
    sentences = text.split(".")
    if sentences:
        subject = sentences[0].strip()
        body = text
        return subject, body

    return "Email", text


def generate_email(email_input: EmailInput, strategy: str = "advanced") -> GeneratedEmail:
    """Generate an email using the specified prompting strategy.

    Args:
        email_input: The input parameters for email generation.
        strategy: Prompting strategy to use ("advanced" or "baseline").

    Returns:
        GeneratedEmail with subject, body, and metadata.

    Raises:
        ValueError: If input validation fails or strategy is unknown.
        RuntimeError: If API call fails after retries.
    """
    validate_api_key()

    if strategy not in STRATEGY_MAP:
        raise ValueError(
            f"Unknown strategy '{strategy}'. Valid strategies: {list(STRATEGY_MAP.keys())}"
        )

    errors = validate_input(email_input)
    if errors:
        raise ValueError(f"Input validation failed: {'; '.join(errors)}")

    # Sanitize inputs
    intent = sanitize_input(email_input.intent)
    key_facts = [sanitize_input(f) for f in email_input.key_facts if f and f.strip()]
    tone = email_input.tone.lower().strip()

    # Build prompt
    messages = STRATEGY_MAP[strategy](intent, key_facts, tone)

    # Call OpenAI API with retry
    client = OpenAI(api_key=OPENAI_API_KEY)
    last_error: Exception | None = None

    for attempt in range(MAX_RETRIES + 1):
        try:
            logger.info(
                "Generating email | strategy=%s | model=%s | attempt=%d",
                strategy,
                OPENAI_MODEL,
                attempt + 1,
            )
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages,
                temperature=0.7,
                max_tokens=1024,
            )

            response_text = response.choices[0].message.content or ""
            subject, body = parse_email_response(response_text)

            usage = response.usage
            prompt_tokens = usage.prompt_tokens if usage else 0
            completion_tokens = usage.completion_tokens if usage else 0

            logger.info(
                "Email generated | strategy=%s | tokens=%d+%d",
                strategy,
                prompt_tokens,
                completion_tokens,
            )

            return GeneratedEmail(
                subject=subject,
                body=body,
                strategy=strategy,
                model=OPENAI_MODEL,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )

        except (RateLimitError, APITimeoutError, APIConnectionError) as e:
            last_error = e
            if attempt < MAX_RETRIES:
                delay = RETRY_BASE_DELAY * (2**attempt)
                logger.warning(
                    "API call failed (attempt %d), retrying in %.1fs | error=%s",
                    attempt + 1,
                    delay,
                    type(e).__name__,
                )
                time.sleep(delay)
            else:
                logger.error(
                    "API call failed after %d attempts | error=%s",
                    MAX_RETRIES + 1,
                    type(e).__name__,
                )

        except Exception as e:
            logger.error("Unexpected error during email generation | error=%s", str(e))
            raise RuntimeError(
                "An unexpected error occurred while generating the email. "
                "Please try again later."
            ) from e

    raise RuntimeError(
        "Unable to generate email due to a temporary service issue. "
        "Please try again in a few moments."
    ) from last_error
