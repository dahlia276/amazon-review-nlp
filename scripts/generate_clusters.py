import json
from pathlib import Path
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
    
from src.clustering import (
    cluster_reviews,
    create_embeddings,
    find_best_k,
    get_cluster_keywords,
    load_embedding_model,
    plot_clusters,
    plot_silhouette_scores,
)
from src.config import DATA_PATH
from src.preprocessing import preprocess_data


OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

N_CLUSTERS = 5
SAMPLE_SIZE = 2000


def main():
    print("Loading and preprocessing data...")

    df = preprocess_data(DATA_PATH)

    df = (
        df.sample(
            SAMPLE_SIZE,
            random_state=42,
        )
        .reset_index(drop=True)
    )

    print("Generating embeddings...")

    model = load_embedding_model()

    embeddings = create_embeddings(
        model,
        df["text"].tolist(),
    )

    print("Running silhouette analysis...")

    scores = find_best_k(
        embeddings,
        min_k=2,
        max_k=8,
    )

    plot_silhouette_scores(scores)

    print("\nSilhouette Scores")
    print("=" * 40)

    for k, score in scores.items():
        print(f"k={k}: {score:.3f}")

    print(f"\nClustering reviews (k={N_CLUSTERS})...")

    labels, _ = cluster_reviews(
        embeddings,
        n_clusters=N_CLUSTERS,
    )

    df["cluster"] = labels

    keywords = get_cluster_keywords(df)

    print("Saving clustered reviews...")

    df.to_csv(
        OUTPUT_DIR / "clustered_reviews.csv",
        index=False,
    )

    with open(
        OUTPUT_DIR / "cluster_keywords.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            keywords,
            f,
            indent=4,
        )

    plot_clusters(
        embeddings,
        labels,
    )

    print("\nDone!")
    print("Generated:")
    print("- outputs/clustered_reviews.csv")
    print("- outputs/cluster_keywords.json")
    print("- outputs/figures/clusters.png")
    print("- outputs/figures/silhouette_scores.png")


if __name__ == "__main__":
    main()