"""Prompt engineering strategies for email generation.

Implements two prompting strategies:
- Advanced (Few-Shot + Role-Playing): Uses expert role assignment and 2 diverse
  few-shot examples to maximize output quality and reliability.
- Baseline (Zero-Shot): Minimal instruction with no examples for comparison.
"""

from typing import Callable

# --- Few-Shot Examples for Advanced Strategy ---

FEW_SHOT_EXAMPLE_1 = """
Example 1:
Intent: Follow up after a quarterly review meeting with a partner firm
Key Facts:
1. Met with GlobalTech's Director of Operations, James Park
2. Reviewed Q3 performance metrics showing 18% revenue growth
3. Identified three optimization opportunities in supply chain
4. Follow-up workshop scheduled for November 10

Tone: formal

Subject: Follow-Up: Q3 Quarterly Review with GlobalTech

Dear Mr. Park,

Thank you for meeting with us to review the Q3 performance metrics. It was encouraging to see the 18% revenue growth we have achieved together this quarter.

As discussed, we identified three key optimization opportunities within the supply chain that we believe can further strengthen our results heading into Q4. I will have our team prepare a detailed analysis of each opportunity for your review.

I look forward to our follow-up workshop on November 10, where we can dive deeper into these initiatives and align on next steps.

Please do not hesitate to reach out if you have any questions before then.

Best regards
""".strip()

FEW_SHOT_EXAMPLE_2 = """
Example 2:
Intent: Share a project update with the engineering team
Key Facts:
1. Successfully completed the API refactoring sprint
2. Response times improved by 35% across all endpoints
3. Zero downtime during the migration
4. Team retrospective happening Thursday at 3 PM

Tone: casual

Subject: API Refactoring Done — Great Work Team!

Hey everyone,

Quick update — we wrapped up the API refactoring sprint and the results are fantastic!

Response times are down by 35% across all endpoints, which is even better than we were targeting. And the best part? We pulled off the entire migration with zero downtime. That's a huge win for the team.

Let's chat about what went well (and what we can do even better) at our retro on Thursday at 3 PM.

Seriously great work, everyone. This was a big one and you all delivered.

Cheers
""".strip()

# --- System Messages ---

ADVANCED_SYSTEM_MESSAGE = (
    "You are a senior executive communications specialist with 15 years of "
    "experience crafting professional emails across industries. You excel at:\n"
    "- Adapting tone precisely to match the requested style\n"
    "- Weaving factual details naturally into the email body\n"
    "- Writing concise yet complete emails\n"
    "- Creating appropriate subject lines that capture the email's purpose\n\n"
    "You MUST:\n"
    "1. Include ALL key facts provided — do not omit any\n"
    "2. Match the requested tone exactly\n"
    "3. Produce both a Subject line and Body\n"
    "4. Format your response as:\n"
    "   Subject: <subject line>\n\n"
    "   <email body>\n\n"
    "Here are two examples of high-quality emails:\n\n"
    f"{FEW_SHOT_EXAMPLE_1}\n\n"
    f"{FEW_SHOT_EXAMPLE_2}"
)

BASELINE_SYSTEM_MESSAGE = (
    "Write a professional email based on the user's request. "
    "Include a subject line and email body."
)


def build_advanced_prompt(
    intent: str, key_facts: list[str], tone: str
) -> list[dict[str, str]]:
    """Build the advanced few-shot + role-playing prompt.

    Args:
        intent: Core purpose of the email.
        key_facts: List of facts to include in the email.
        tone: Desired tone for the email.

    Returns:
        List of message dicts for the OpenAI chat API.
    """
    numbered_facts = "\n".join(
        f"{i + 1}. {fact}" for i, fact in enumerate(key_facts)
    )

    user_message = (
        f"Write an email with the following details:\n\n"
        f"Intent: {intent}\n\n"
        f"Key Facts:\n{numbered_facts}\n\n"
        f"Tone: {tone}\n\n"
        f"Format your response as:\n"
        f"Subject: <subject line>\n\n"
        f"<email body>"
    )

    return [
        {"role": "system", "content": ADVANCED_SYSTEM_MESSAGE},
        {"role": "user", "content": user_message},
    ]


def build_baseline_prompt(
    intent: str, key_facts: list[str], tone: str
) -> list[dict[str, str]]:
    """Build the baseline zero-shot prompt.

    Args:
        intent: Core purpose of the email.
        key_facts: List of facts to include in the email.
        tone: Desired tone for the email.

    Returns:
        List of message dicts for the OpenAI chat API.
    """
    facts_text = "\n".join(f"- {fact}" for fact in key_facts)

    user_message = (
        f"Generate an email with a subject line and body.\n\n"
        f"Purpose: {intent}\n"
        f"Include these facts:\n{facts_text}\n"
        f"Tone: {tone}"
    )

    return [
        {"role": "system", "content": BASELINE_SYSTEM_MESSAGE},
        {"role": "user", "content": user_message},
    ]


STRATEGY_MAP: dict[str, Callable[..., list[dict[str, str]]]] = {
    "advanced": build_advanced_prompt,
    "baseline": build_baseline_prompt,
}
