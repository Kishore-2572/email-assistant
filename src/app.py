"""Streamlit web UI for the Email Generation Assistant."""

import streamlit as st

import sys
import os

# Ensure the project root is on the Python path so that both
# `streamlit run src/app.py` and `python -m src.app` work.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import SUPPORTED_TONES, logger
from src.email_generator import EmailInput, GeneratedEmail, generate_email

st.set_page_config(
    page_title="Email Generation Assistant",
    page_icon="✉️",
    layout="centered",
)

# --- Sidebar ---
with st.sidebar:
    st.header("Configuration")

    strategy = st.radio(
        "Prompting Strategy",
        options=["Advanced (Few-Shot)", "Baseline (Zero-Shot)"],
        index=0,
        help=(
            "**Advanced**: Uses role-playing and 2 few-shot examples for "
            "higher quality output.\n\n"
            "**Baseline**: Simple zero-shot prompt with minimal instructions."
        ),
    )
    strategy_key = "advanced" if "Advanced" in strategy else "baseline"

    tone = st.selectbox(
        "Tone",
        options=[t.title() for t in SUPPORTED_TONES],
        index=0,
    )

    st.divider()
    st.caption("Built with OpenAI GPT-4o")

# --- Main Area ---
st.title("✉️ Email Generation Assistant")
st.caption("Generate professional emails using AI with advanced prompt engineering")

intent = st.text_input(
    "Intent",
    placeholder="e.g., Follow up after Monday's client meeting",
    max_chars=500,
)

key_facts_text = st.text_area(
    "Key Facts (one per line)",
    placeholder="e.g.,\nMet with Sarah Chen, VP of Engineering\nDiscussed cloud migration timeline\nNext meeting on December 5",
    height=150,
)

# Parse key facts
key_facts = [line.strip() for line in key_facts_text.split("\n") if line.strip()]
if key_facts_text.strip():
    st.caption(f"Facts detected: {len(key_facts)}")

# --- Generate Button ---
generate_clicked = st.button("Generate Email", type="primary", use_container_width=True)

if generate_clicked:
    # Validate inputs
    if not intent.strip():
        st.warning("Please enter an intent for the email.")
    elif not key_facts:
        st.warning("Please enter at least one key fact.")
    else:
        email_input = EmailInput(
            intent=intent.strip(),
            key_facts=key_facts,
            tone=tone.lower(),
        )

        with st.spinner("Generating email..."):
            try:
                logger.info(
                    "UI generation request | strategy=%s | tone=%s | facts=%d",
                    strategy_key,
                    tone.lower(),
                    len(key_facts),
                )
                result: GeneratedEmail = generate_email(email_input, strategy_key)
                st.session_state["generated_email"] = result
            except ValueError as e:
                st.error(f"Input error: {e}")
                logger.warning("UI validation error | error=%s", str(e))
            except RuntimeError as e:
                st.error(str(e))
                logger.error("UI generation error | error=%s", str(e))

# --- Output Area ---
if "generated_email" in st.session_state and st.session_state["generated_email"]:
    result = st.session_state["generated_email"]

    st.divider()
    st.subheader(f"📧 {result.subject}")
    st.markdown(result.body.replace("\n", "  \n"))

    with st.expander("Details"):
        st.markdown(f"**Strategy**: {result.strategy.title()}")
        st.markdown(f"**Model**: {result.model}")
        st.markdown(
            f"**Tokens**: {result.prompt_tokens} prompt + "
            f"{result.completion_tokens} completion = "
            f"{result.prompt_tokens + result.completion_tokens} total"
        )
