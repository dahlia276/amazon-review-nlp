from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel
import os

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)


class ReviewSummary(BaseModel):
    themes: list[str]
    positives: list[str]
    negatives: list[str]
    overall_sentiment: str
    summary: str


def summarize_reviews_openai(reviews):
    """
    Summarize up to 20 Amazon reviews using OpenAI.

    Args:
        reviews: List of review texts.

    Returns:
        ReviewSummary containing themes, positives,
        negatives, overall sentiment, and a concise summary.
    """

    reviews = reviews[:20]

    prompt = "\n\n".join(reviews)

    response = client.responses.parse(
        model="gpt-4.1-mini",
        input=f"""
    Analyze these Amazon product reviews.

    Extract:

    - Main product themes
    - Common positive feedback
    - Common negative feedback
    - Overall sentiment
    - A concise summary (maximum 150 words)

    Reviews:

    {prompt}
    """,
        text_format=ReviewSummary,
    )

    return response.output_parsed