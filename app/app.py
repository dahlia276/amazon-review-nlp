import streamlit as st

st.set_page_config(
    page_title="Amazon Review NLP",
    page_icon="🛍️",
    layout="wide",
)

st.title("🛍️ Amazon Review NLP Dashboard")

st.markdown(
    """
Welcome!

This project analyzes Amazon customer reviews using several NLP techniques.

### Features

- 😊 Sentiment Analysis
    - TF-IDF + Logistic Regression
    - RoBERTa Transformer

- 🔍 Semantic Review Clustering
    - Sentence Transformers
    - K-Means
    - PCA Visualization

- 📝 Review Summarization
    - BART (baseline)
    - OpenAI GPT (enhanced)

Use the navigation menu on the left to explore each feature.
"""
)