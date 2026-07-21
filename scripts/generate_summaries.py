import json
import os
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.openai_summarizer import summarize_reviews_openai
from src.summarizer import (
    load_summarizer,
    summarize_reviews,
)

OUTPUT_DIR = Path("outputs")
SUMMARY_DIR = OUTPUT_DIR / "summaries"
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("Loading clustered reviews...")

    df = pd.read_csv(OUTPUT_DIR / "clustered_reviews.csv")

    summarizer = load_summarizer()

    bart_summaries = {}
    openai_summaries = {}

    for cluster in sorted(df["cluster"].unique()):

        print(f"Summarizing cluster {cluster}...")

        reviews = (
            df[df["cluster"] == cluster]["text"]
            .sample(
                min(20, len(df[df["cluster"] == cluster])),
                random_state=42,
            )
            .tolist()
        )

        bart_summary = summarize_reviews(
            summarizer,
            reviews,
        )

        openai_summary = summarize_reviews_openai(
            reviews,
        )

        bart_summaries[str(cluster)] = bart_summary

        openai_summaries[str(cluster)] = (
            openai_summary.model_dump()
        )

    with open(
        SUMMARY_DIR / "bart_summaries.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            bart_summaries,
            f,
            indent=4,
            ensure_ascii=False,
        )

    with open(
        SUMMARY_DIR / "openai_summaries.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            openai_summaries,
            f,
            indent=4,
            ensure_ascii=False,
        )

    print("\nDone!")
    print("Generated:")
    print("- outputs/summaries/bart_summaries.json")
    print("- outputs/summaries/openai_summaries.json")


if __name__ == "__main__":
    main()