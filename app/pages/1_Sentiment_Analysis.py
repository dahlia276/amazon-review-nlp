import os
import sys

# Add the project root to Python's path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st

from src.classifier import predict_logistic
from src.transformer_classifier import predict_roberta

st.title("😊 Sentiment Analysis")

st.write(
    "Compare a traditional machine learning classifier with a transformer model."
)

review = st.text_area(
    "Enter an Amazon review:",
    height=200,
)

if st.button("Analyze Sentiment"):

    if not review.strip():
        st.warning("Please enter a review.")
        st.stop()

    with st.spinner("Analyzing review..."):
        logistic_prediction = predict_logistic(review)
        roberta_prediction = predict_roberta(review)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("TF-IDF + Logistic Regression")
        st.success(logistic_prediction.capitalize())

    with col2:
        st.subheader("RoBERTa Transformer")
        st.success(roberta_prediction.capitalize())