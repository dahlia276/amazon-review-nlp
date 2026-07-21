import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from src.clustering import (
    cluster_reviews,
    create_embeddings,
    load_embedding_model,
)
from src.config import DATA_PATH
from src.preprocessing import preprocess_data
from src.summarizer import (
    load_summarizer,
    summarize_reviews,
)

def main():
    df = preprocess_data(DATA_PATH)

    df = df.sample(
        1000,
        random_state=42,
    ).reset_index(drop=True)

    embedding_model = load_embedding_model()

    embeddings = create_embeddings(
        embedding_model,
        df["text"].tolist(),
    )

    labels, _ = cluster_reviews(
        embeddings,
        n_clusters=5,
    )

    df["cluster"] = labels

    summarizer = load_summarizer()

    for cluster in sorted(df["cluster"].unique()):

        reviews = (
            df[df["cluster"] == cluster]["text"]
            .sample(
                min(25, len(df[df["cluster"] == cluster])),
                random_state=42,
            )
            .tolist()
        )

        print("\n" + "=" * 80)
        print(f"CLUSTER {cluster}")
        print("=" * 80)

        print(f"Summarizing {len(reviews)} reviews...")

        summary = summarize_reviews(
            summarizer,
            reviews,
        )
        print()
        print(summary)
        print(summary)
    
if __name__ == "__main__":
    main()