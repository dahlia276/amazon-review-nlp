from functools import lru_cache

import joblib

from src.config import IS_LOCAL


@lru_cache
def get_embedding_model():
    if not IS_LOCAL:
        raise RuntimeError(
            "SentenceTransformer is only available in local mode."
        )

    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )


@lru_cache
def get_summarizer():
    if not IS_LOCAL:
        raise RuntimeError(
            "BART summarizer is only available in local mode."
        )

    from transformers import pipeline

    return pipeline(
        "summarization",
        model="facebook/bart-large-cnn",
    )


@lru_cache
def get_roberta_classifier():
    from transformers import pipeline

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