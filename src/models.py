from functools import lru_cache
from sentence_transformers import SentenceTransformer
from transformers import pipeline


@lru_cache
def get_embedding_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


@lru_cache
def get_summarizer():
    return pipeline(
        "summarization",
        model="facebook/bart-large-cnn",
    )