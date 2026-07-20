from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.config import DATA_PATH
from src.preprocessing import preprocess_data


OUTPUT_DIR = Path("outputs/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    df = preprocess_data(DATA_PATH)

    print(f"Dataset shape: {df.shape}\n")

    print("Sentiment distribution:")
    print(df["sentiment"].value_counts())
    print()

    print("Rating distribution:")
    print(df["rating"].value_counts().sort_index())

    # Rating distribution
    plt.figure(figsize=(6, 4))
    df["rating"].value_counts().sort_index().plot(kind="bar")
    plt.title("Rating Distribution")
    plt.xlabel("Rating")
    plt.ylabel("Number of Reviews")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "rating_distribution.png")
    plt.close()

    # Sentiment distribution
    plt.figure(figsize=(6, 4))
    df["sentiment"].value_counts().plot(kind="bar")
    plt.title("Sentiment Distribution")
    plt.xlabel("Sentiment")
    plt.ylabel("Number of Reviews")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "sentiment_distribution.png")
    plt.close()

    # Review length
    df["review_length"] = df["text"].str.split().str.len()

    plt.figure(figsize=(6, 4))
    plt.hist(df["review_length"], bins=40)
    plt.title("Review Length Distribution")
    plt.xlabel("Words")
    plt.ylabel("Number of Reviews")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "review_length_distribution.png")
    plt.close()

    print("\nFigures saved to outputs/figures/")


if __name__ == "__main__":
    main()