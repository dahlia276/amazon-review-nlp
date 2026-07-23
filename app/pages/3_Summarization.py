import os
import sys

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
import json
from pathlib import Path

from src.config import IS_LOCAL

st.set_page_config(page_title="Review Summarization", page_icon="📝")

st.title("📝 Review Summarization")

if IS_LOCAL:
    st.write(
        """
Compare summaries generated using a traditional transformer model (BART)
and OpenAI's structured output API.
"""
    )
else:
    st.write(
        """
Explore structured summaries generated using OpenAI's structured output API.
"""
    )

SUMMARY_DIR = Path("outputs/summaries")

with open(
    SUMMARY_DIR / "openai_summaries.json",
    "r",
    encoding="utf-8",
) as f:
    openai_summaries = json.load(f)

bart_summaries = None

if IS_LOCAL:
    with open(
        SUMMARY_DIR / "bart_summaries.json",
        "r",
        encoding="utf-8",
    ) as f:
        bart_summaries = json.load(f)

cluster = st.selectbox(
    "Select a cluster",
    sorted(openai_summaries.keys(), key=int),
)

if IS_LOCAL:
    col1, col2 = st.columns(2)

    with col1:
        with st.container(border=True):
            st.subheader("🤖 BART")

            st.caption(
                "Traditional transformer-based abstractive summarization."
            )

            st.markdown("#### Summary")

            st.write(bart_summaries[cluster])

    with col2:
        with st.container(border=True):
            summary = openai_summaries[cluster]

            st.subheader("✨ OpenAI")

            st.caption(
                "Structured LLM analysis with extracted insights."
            )

            st.markdown("#### Themes")

            for theme in summary["themes"]:
                st.markdown(f"- {theme}")

            st.markdown("#### Positives")

            for item in summary["positives"]:
                st.markdown(f"- {item}")

            st.markdown("#### Negatives")

            for item in summary["negatives"]:
                st.markdown(f"- {item}")

            st.markdown("#### Overall Sentiment")

            st.success(summary["overall_sentiment"])

            st.markdown("#### Summary")

            st.write(summary["summary"])

else:
    summary = openai_summaries[cluster]

    with st.container(border=True):
        st.subheader("✨ OpenAI")

        st.caption(
            "Structured LLM analysis with extracted insights."
        )

        st.markdown("#### Themes")

        for theme in summary["themes"]:
            st.markdown(f"- {theme}")

        st.markdown("#### Positives")

        for item in summary["positives"]:
            st.markdown(f"- {item}")

        st.markdown("#### Negatives")

        for item in summary["negatives"]:
            st.markdown(f"- {item}")

        st.markdown("#### Overall Sentiment")

        st.success(summary["overall_sentiment"])

        st.markdown("#### Summary")

        st.write(summary["summary"])
