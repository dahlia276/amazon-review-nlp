from pathlib import Path

import matplotlib.pyplot as plt

from src.clustering import (
    cluster_reviews,
    generate_embeddings,
    reduce_dimensions,
)
from src.config import DATA_PATH
from src.preprocessing import preprocess_data

OUTPUT_DIR = Path("outputs/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    df = preprocess_data(DATA_PATH)

    # Limit for speed while developing
    df = df.sample(3000, random_state=42)

    embeddings = generate_embeddings(df)

    labels = cluster_reviews(embeddings)

    reduced = reduce_dimensions(embeddings)

    plt.figure(figsize=(8, 6))
    plt.scatter(
        reduced[:, 0],
        reduced[:, 1],
        c=labels,
        s=8,
    )

    plt.title("Review Clusters")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "review_clusters.png")

    print("Cluster visualization saved.")


if __name__ == "__main__":
    main()