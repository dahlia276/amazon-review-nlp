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

def main():
    df = preprocess_data(DATA_PATH)

    # Smaller sample for speed
    df = df.sample(
        2000,
        random_state=42,
    ).reset_index(drop=True)

    model = load_embedding_model()

    embeddings = create_embeddings(
        model,
        df["text"].tolist(),
    )
    
    scores = find_best_k(
    embeddings,
    min_k=2,
    max_k=8,
    )
    plot_silhouette_scores(scores)
    print("\nSilhouette Scores")
    print("=" * 40)

    for k, score in scores.items():
        print(f"k = {k}: {score:.3f}")

    # Silhouette analysis suggests larger k values only provide
    # marginal improvements. Chose k=5 because it produces
    # more interpretable clusters.
    N_CLUSTERS = 5
    labels, _ = cluster_reviews(
        embeddings,
        n_clusters=N_CLUSTERS,
    )

    df["cluster"] = labels
    
    keywords = get_cluster_keywords(df)

    print("\n" + "=" * 80)
    print("TOP KEYWORDS")
    print("=" * 80)

    for cluster, words in keywords.items():
        print(f"\nCluster {cluster}")
        print(", ".join(words))

    print("\nCluster counts:\n")
    print(df["cluster"].value_counts().sort_index())

    print("\n" + "=" * 80)

    for cluster in sorted(df["cluster"].unique()):
        print(f"\nCLUSTER {cluster}")
        print("-" * 80)

        examples = (
            df[df["cluster"] == cluster]["text"]
            .head(2)
            .tolist()
        )

        for i, review in enumerate(examples, start=1):
            print(f"\nReview {i}:")
            print(review[:300])
            print()

    plot_clusters(
        embeddings,
        labels,
    )

    print("\nCluster visualization saved to outputs/figures/clusters.png")

if __name__ == "__main__":
    main()