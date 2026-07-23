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
    """Normalize review text while retaining sentiment-bearing negation."""
    if pd.isna(text):
        return ""

    text = str(text)
    text = text.replace("’", "'")
    text = re.sub(r"<[^>]+>", " ", text)
    # Expand negated contractions before vectorization so phrases such as
    # "don't like" become the meaningful bigram "not like".
    text = re.sub(r"\bcan't\b", "can not", text, flags=re.IGNORECASE)
    text = re.sub(r"\bwon't\b", "will not", text, flags=re.IGNORECASE)
    text = re.sub(r"n't\b", " not", text, flags=re.IGNORECASE)
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
