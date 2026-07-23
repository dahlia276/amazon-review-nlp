import os
import sys

# Add the project root to Python's path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
from src.classifier import predict_logistic
from src.config import IS_LOCAL
from src.transformer_classifier import predict_roberta

st.title("😊 Sentiment Analysis")

if IS_LOCAL:
    st.write(
        "Compare a traditional machine learning classifier with a transformer model."
    )
else:
    st.write(
        "Predict the sentiment of an Amazon review using the deployed classifier."
    )

    st.info(
        "The deployed version uses the TF-IDF + Logistic Regression model for "
        "live predictions. The RoBERTa transformer is available in the local "
        "version of this project because it exceeds the memory limits of the "
        "free hosting tier."
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

        if IS_LOCAL:
            roberta_prediction = predict_roberta(review)

    if IS_LOCAL:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("TF-IDF + Logistic Regression")
            st.success(logistic_prediction.capitalize())

        with col2:
            st.subheader("RoBERTa Transformer")
            st.success(roberta_prediction.capitalize())

    else:
        st.subheader("TF-IDF + Logistic Regression")
        st.success(logistic_prediction.capitalize())