"""Custom evaluation metrics for the Email Generation Assistant.

Implements three metrics:
1. Fact Recall — LLM-as-a-Judge for fact inclusion accuracy
2. Tone Accuracy — LLM-as-a-Judge for tone alignment
3. Conciseness & Clarity — Hybrid (30% automated + 70% LLM judge)
"""

import json
import time
from dataclasses import dataclass

from openai import APIConnectionError, APITimeoutError, OpenAI, RateLimitError

from src.config import (
    MAX_RETRIES,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    RETRY_BASE_DELAY,
    logger,
)


@dataclass
class MetricScore:
    """Score from a single metric evaluation."""

    metric_name: str
    score: float
    reasoning: str


@dataclass
class MetricDefinition:
    """Definition and logic for a custom metric."""

    name: str
    key: str
    description: str
    scoring_logic: str
    scale: str


METRIC_DEFINITIONS: list[MetricDefinition] = [
    MetricDefinition(
        name="Fact Recall",
        key="fact_recall",
        description=(
            "Measures whether all key facts from the input are present and "
            "accurately represented in the generated email. Evaluates fact "
            "presence, accuracy, and natural integration."
        ),
        scoring_logic=(
            "LLM-as-a-Judge evaluates each key fact for presence, accuracy, and "
            "natural integration. 10 = all facts present and naturally woven in. "
            "0 = few or no facts included."
        ),
        scale="0-10",
    ),
    MetricDefinition(
        name="Tone Accuracy",
        key="tone_accuracy",
        description=(
            "Measures how well the generated email matches the requested tone "
            "in vocabulary, sentence structure, and overall feel."
        ),
        scoring_logic=(
            "LLM-as-a-Judge evaluates vocabulary appropriateness, sentence "
            "structure, emotional register, and overall feel against the "
            "requested tone. 10 = perfect match. 0 = completely wrong tone."
        ),
        scale="0-10",
    ),
    MetricDefinition(
        name="Conciseness & Clarity",
        key="conciseness_clarity",
        description=(
            "Measures whether the email is appropriately concise while remaining "
            "clear, well-structured, and complete. Combines automated word-count "
            "analysis (30%) with LLM judgment (70%)."
        ),
        scoring_logic=(
            "Hybrid metric: 30% automated (word count ratio vs reference, optimal "
            "range 0.8-1.2) + 70% LLM-as-a-Judge (evaluates filler, redundancy, "
            "structure, readability). 10 = perfectly concise and clear. "
            "0 = extremely verbose or incomprehensible."
        ),
        scale="0-10",
    ),
]


# --- Judge Prompt Templates ---

FACT_RECALL_PROMPT = """You are an expert email quality evaluator. Evaluate the following generated email for FACT RECALL — how well it includes all the key facts provided.

Original Inputs:
- Intent: {intent}
- Key Facts:
{key_facts}
- Tone: {tone}

Reference Email (ideal output):
Subject: {ref_subject}

{ref_body}

Generated Email (to evaluate):
Subject: {gen_subject}

{gen_body}

Score the generated email from 0-10 for Fact Recall using this rubric:
- 10: All facts present, accurately stated, naturally woven into email
- 8-9: All facts present and accurate, minor integration awkwardness
- 6-7: Most facts present (80%+), minor inaccuracies or forced inclusion
- 4-5: Some facts missing (50-80% present), or noticeable inaccuracies
- 2-3: Many facts missing (<50% present) or significant distortions
- 0-1: Few or no facts included, or facts are grossly inaccurate

You MUST respond with ONLY a JSON object in this exact format, no other text:
{{"score": <number>, "reasoning": "<brief explanation>"}}"""

TONE_ACCURACY_PROMPT = """You are an expert email quality evaluator. Evaluate the following generated email for TONE ACCURACY — how well it matches the requested tone.

Original Inputs:
- Intent: {intent}
- Tone Requested: {tone}

Tone-Specific Indicators for "{tone}":
{tone_indicators}

Reference Email (ideal output):
Subject: {ref_subject}

{ref_body}

Generated Email (to evaluate):
Subject: {gen_subject}

{gen_body}

Score the generated email from 0-10 for Tone Accuracy using this rubric:
- 10: Perfect tone match — vocabulary, structure, and feel all align exactly
- 8-9: Strong tone match with very minor deviations
- 6-7: Generally correct tone but with noticeable inconsistencies in parts
- 4-5: Mixed tone — partly matches but significant sections feel off
- 2-3: Mostly wrong tone — occasional correct elements but overall mismatch
- 0-1: Completely wrong tone or tone-deaf email

You MUST respond with ONLY a JSON object in this exact format, no other text:
{{"score": <number>, "reasoning": "<brief explanation>"}}"""

CONCISENESS_JUDGE_PROMPT = """You are an expert email quality evaluator. Evaluate the following generated email for CONCISENESS AND CLARITY — whether it is appropriately concise while remaining clear and complete.

Original Inputs:
- Intent: {intent}
- Key Facts: {key_facts}

Reference Email (ideal output):
Subject: {ref_subject}

{ref_body}

Generated Email (to evaluate):
Subject: {gen_subject}

{gen_body}

Score the generated email from 0-10 for Conciseness & Clarity using this rubric:
- 10: Perfectly concise and crystal clear — every sentence serves a purpose
- 8-9: Very concise and clear with minimal unnecessary content
- 6-7: Generally clear but contains some filler or could be tighter
- 4-5: Noticeably verbose or somewhat unclear in places
- 2-3: Quite verbose with significant filler, or unclear structure
- 0-1: Extremely verbose, rambling, or incomprehensible

You MUST respond with ONLY a JSON object in this exact format, no other text:
{{"score": <number>, "reasoning": "<brief explanation>"}}"""


TONE_INDICATORS: dict[str, str] = {
    "formal": (
        "- Professional salutation (Dear Mr./Ms.)\n"
        "- Complete sentences, no contractions\n"
        "- Structured paragraphs\n"
        "- Polite closings (Best regards, Sincerely)"
    ),
    "casual": (
        "- Friendly greeting (Hey, Hi)\n"
        "- Contractions are fine (we're, it's)\n"
        "- Conversational, shorter sentences\n"
        "- Informal closings (Cheers, Thanks)"
    ),
    "urgent": (
        "- Action-oriented language\n"
        "- Time-sensitive phrasing (immediately, deadline)\n"
        "- Clear deadlines and expectations\n"
        "- Imperative mood (Please submit, Confirm receipt)"
    ),
    "empathetic": (
        "- Acknowledgment of feelings/situation\n"
        "- Supportive, compassionate language\n"
        "- Understanding tone (I know, I understand)\n"
        "- Warm closings (Warm regards, With appreciation)"
    ),
    "persuasive": (
        "- Compelling arguments and benefits\n"
        "- Data-driven points\n"
        "- Confident, assertive language\n"
        "- Clear call-to-action"
    ),
    "apologetic": (
        "- Sincere expression of regret\n"
        "- Acknowledgment of the issue and impact\n"
        "- Commitment to resolution/prevention\n"
        "- Humble, accountable tone"
    ),
}


def _call_judge(prompt: str) -> dict:
    """Call the LLM judge and parse the JSON response.

    Args:
        prompt: The complete judge prompt.

    Returns:
        Parsed JSON dict with 'score' and 'reasoning'.

    Raises:
        RuntimeError: If the judge call fails after retries.
    """
    client = OpenAI(api_key=OPENAI_API_KEY)
    last_error: Exception | None = None

    for attempt in range(MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=256,
            )

            text = response.choices[0].message.content or ""
            text = text.strip()

            # Handle markdown code blocks
            if text.startswith("```"):
                lines = text.split("\n")
                text = "\n".join(
                    line for line in lines
                    if not line.strip().startswith("```")
                )
                text = text.strip()

            result = json.loads(text)

            # Validate and clamp score
            score = float(result.get("score", 0))
            score = max(0.0, min(10.0, score))
            reasoning = str(result.get("reasoning", "No reasoning provided."))

            return {"score": score, "reasoning": reasoning}

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            last_error = e
            if attempt < MAX_RETRIES:
                logger.warning(
                    "Judge response parsing failed (attempt %d), retrying | error=%s",
                    attempt + 1,
                    str(e),
                )
                time.sleep(RETRY_BASE_DELAY)
            else:
                logger.error("Judge response parsing failed after retries | error=%s", str(e))

        except (RateLimitError, APITimeoutError, APIConnectionError) as e:
            last_error = e
            if attempt < MAX_RETRIES:
                delay = RETRY_BASE_DELAY * (2**attempt)
                logger.warning(
                    "Judge API call failed (attempt %d), retrying in %.1fs | error=%s",
                    attempt + 1,
                    delay,
                    type(e).__name__,
                )
                time.sleep(delay)
            else:
                logger.error("Judge API call failed after retries | error=%s", type(e).__name__)

        except Exception as e:
            logger.error("Unexpected judge error | error=%s", str(e))
            raise RuntimeError("Unexpected error during evaluation.") from e

    raise RuntimeError(
        f"Judge evaluation failed after {MAX_RETRIES + 1} attempts."
    ) from last_error


def evaluate_fact_recall(
    intent: str,
    key_facts: list[str],
    tone: str,
    gen_subject: str,
    gen_body: str,
    ref_subject: str,
    ref_body: str,
) -> MetricScore:
    """Evaluate fact recall using LLM-as-a-Judge.

    Args:
        intent: Original email intent.
        key_facts: List of key facts that should be included.
        tone: Requested tone.
        gen_subject: Generated email subject.
        gen_body: Generated email body.
        ref_subject: Reference email subject.
        ref_body: Reference email body.

    Returns:
        MetricScore with fact recall evaluation.
    """
    facts_formatted = "\n".join(f"  {i + 1}. {f}" for i, f in enumerate(key_facts))

    prompt = FACT_RECALL_PROMPT.format(
        intent=intent,
        key_facts=facts_formatted,
        tone=tone,
        ref_subject=ref_subject,
        ref_body=ref_body,
        gen_subject=gen_subject,
        gen_body=gen_body,
    )

    result = _call_judge(prompt)
    return MetricScore(
        metric_name="fact_recall",
        score=result["score"],
        reasoning=result["reasoning"],
    )


def evaluate_tone_accuracy(
    intent: str,
    tone: str,
    gen_subject: str,
    gen_body: str,
    ref_subject: str,
    ref_body: str,
) -> MetricScore:
    """Evaluate tone accuracy using LLM-as-a-Judge.

    Args:
        intent: Original email intent.
        tone: Requested tone.
        gen_subject: Generated email subject.
        gen_body: Generated email body.
        ref_subject: Reference email subject.
        ref_body: Reference email body.

    Returns:
        MetricScore with tone accuracy evaluation.
    """
    indicators = TONE_INDICATORS.get(tone.lower(), "No specific indicators available.")

    prompt = TONE_ACCURACY_PROMPT.format(
        intent=intent,
        tone=tone,
        tone_indicators=indicators,
        ref_subject=ref_subject,
        ref_body=ref_body,
        gen_subject=gen_subject,
        gen_body=gen_body,
    )

    result = _call_judge(prompt)
    return MetricScore(
        metric_name="tone_accuracy",
        score=result["score"],
        reasoning=result["reasoning"],
    )


def _calculate_automated_conciseness(gen_text: str, ref_text: str) -> float:
    """Calculate automated conciseness score based on word count ratio.

    Optimal ratio is 0.8-1.2 (full score). Degrades linearly outside that range.

    Args:
        gen_text: Generated email full text.
        ref_text: Reference email full text.

    Returns:
        Score from 0.0 to 10.0.
    """
    gen_words = len(gen_text.split())
    ref_words = len(ref_text.split())

    if ref_words == 0:
        return 5.0

    ratio = gen_words / ref_words

    if 0.8 <= ratio <= 1.2:
        return 10.0
    elif ratio < 0.8:
        # Too short: linearly degrade from 10 at 0.8 to 0 at 0.0
        return max(0.0, (ratio / 0.8) * 10.0)
    else:
        # Too long: linearly degrade from 10 at 1.2 to 0 at 2.5
        return max(0.0, ((2.5 - ratio) / (2.5 - 1.2)) * 10.0)


def evaluate_conciseness_clarity(
    intent: str,
    key_facts: list[str],
    gen_subject: str,
    gen_body: str,
    ref_subject: str,
    ref_body: str,
) -> MetricScore:
    """Evaluate conciseness & clarity using hybrid approach.

    30% automated word-count analysis + 70% LLM-as-a-Judge.

    Args:
        intent: Original email intent.
        key_facts: List of key facts.
        gen_subject: Generated email subject.
        gen_body: Generated email body.
        ref_subject: Reference email subject.
        ref_body: Reference email body.

    Returns:
        MetricScore with conciseness and clarity evaluation.
    """
    gen_full = f"{gen_subject}\n\n{gen_body}"
    ref_full = f"{ref_subject}\n\n{ref_body}"

    # Automated component (30%)
    automated_score = _calculate_automated_conciseness(gen_full, ref_full)

    # LLM judge component (70%)
    facts_formatted = "\n".join(f"  {i + 1}. {f}" for i, f in enumerate(key_facts))

    prompt = CONCISENESS_JUDGE_PROMPT.format(
        intent=intent,
        key_facts=facts_formatted,
        ref_subject=ref_subject,
        ref_body=ref_body,
        gen_subject=gen_subject,
        gen_body=gen_body,
    )

    judge_result = _call_judge(prompt)
    judge_score = judge_result["score"]

    # Weighted combination
    final_score = round(0.3 * automated_score + 0.7 * judge_score, 2)

    reasoning = (
        f"Automated (word ratio): {automated_score:.1f}/10, "
        f"LLM Judge: {judge_score:.1f}/10. "
        f"{judge_result['reasoning']}"
    )

    return MetricScore(
        metric_name="conciseness_clarity",
        score=final_score,
        reasoning=reasoning,
    )
