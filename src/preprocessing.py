import pandas as pd
import re

SENTIMENT_MAP = {
    1: "negative",
    2: "negative",
    3: "neutral",
    4: "positive",
    5: "positive",
}

REQUIRED_COLUMNS = [
    "title",
    "text",
    "rating",
    "parent_asin",
]


def load_data(filepath: str) -> pd.DataFrame:
    """Load the raw dataset."""
    return pd.read_json(filepath, lines=True)


def validate_columns(df: pd.DataFrame) -> None:
    """Ensure the dataset has the expected columns."""
    missing = set(REQUIRED_COLUMNS) - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def clean_text(text: str) -> str:
    """Remove HTML tags and normalize whitespace."""
    if pd.isna(text):
        return ""

    text = str(text)
    text = re.sub(r"<[^>]+>", " ", text)
    return " ".join(text.split())


def preprocess_data(filepath: str) -> pd.DataFrame:
    """Complete preprocessing pipeline."""

    df = load_data(filepath)

    validate_columns(df)

    df = df[REQUIRED_COLUMNS].copy()

    df.dropna(subset=["text", "rating"], inplace=True)

    df.drop_duplicates(subset=["text"], inplace=True)

    df["text"] = df["text"].apply(clean_text)

    df["sentiment"] = df["rating"].map(SENTIMENT_MAP)

    return df.reset_index(drop=True)