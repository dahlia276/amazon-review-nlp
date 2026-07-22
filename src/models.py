from functools import lru_cache

import joblib
from sentence_transformers import SentenceTransformer
from transformers import pipeline


@lru_cache
def get_embedding_model():
    return SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )


@lru_cache
def get_summarizer():
    return pipeline(
        "summarization",
        model="facebook/bart-large-cnn",
    )


@lru_cache
def get_roberta_classifier():
    return pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        truncation=True,
    )


@lru_cache
def get_logistic_model():
    model = joblib.load("outputs/classifier.joblib")
    vectorizer = joblib.load("outputs/vectorizer.joblib")
    return model, vectorizer