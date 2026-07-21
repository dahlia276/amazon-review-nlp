from transformers import pipeline
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

MODEL_NAME = "facebook/bart-large-cnn"


def load_summarizer():
    """
    Load the pretrained BART summarizer.
    """
    return pipeline(
        "summarization",
        model=MODEL_NAME,
        device=-1,
    )


def summarize_text(
    summarizer,
    text,
):
    """
    Summarize a single block of text.
    """

    summary = summarizer(
        text,
        max_length=120,
        min_length=40,
        do_sample=False,
        truncation=True,
    )

    return summary[0]["summary_text"]


def summarize_reviews(
    summarizer,
    reviews,
    batch_size=5,
):
    """
    Summarize a collection of reviews using hierarchical summarization.
    """

    if not reviews:
        return "No reviews available."

    batch_summaries = []

    for i in range(0, len(reviews), batch_size):

        batch = reviews[i : i + batch_size]

        text = "\n".join(batch)

        # Keep each batch comfortably below BART's limit
        text = text[:2500]

        batch_summary = summarize_text(
            summarizer,
            text,
        )

        batch_summaries.append(batch_summary)

    combined = "\n".join(batch_summaries)

    if len(batch_summaries) == 1:
        return combined

    combined = combined[:3000]

    final_summary = summarize_text(
        summarizer,
        combined,
    )

    return final_summary