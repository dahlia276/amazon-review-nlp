import streamlit as st

st.set_page_config(
    page_title="Amazon Review NLP",
    page_icon="⭐",
    layout="wide"
)

st.title("Amazon Review Analyzer")

st.write(
    "Upload Amazon reviews to perform sentiment analysis, "
    "semantic clustering, and AI-powered summarization."
)