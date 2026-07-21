import json
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Review Clustering", page_icon="🧩")

st.title("🧩 Review Clustering")

st.write(
    """
This page explores clusters of semantically similar Amazon reviews created
using SentenceTransformer embeddings and K-Means clustering.
"""
)

OUTPUT_DIR = Path("outputs")

clustered_reviews = pd.read_csv(
    OUTPUT_DIR / "clustered_reviews.csv"
)

with open(
    OUTPUT_DIR / "cluster_keywords.json",
    "r",
    encoding="utf-8",
) as f:
    cluster_keywords = json.load(f)

st.subheader("Cluster Visualisation")

st.write(
    "The reviews are projected into two dimensions using PCA. Reviews that are "
    "closer together are more semantically similar."
)

st.image(
    "outputs/figures/clusters.png",
    width="stretch",
)

st.subheader("Silhouette Analysis")

st.write(
    "The silhouette score helps evaluate cluster quality. Higher values indicate "
    "better separation between clusters."
)

st.image(
    "outputs/figures/silhouette_scores.png",
    width="stretch",
)

st.subheader("Explore Clusters")

cluster_options = sorted(clustered_reviews["cluster"].unique())

selected_cluster = st.selectbox(
    "Select a cluster",
    cluster_options,
)

cluster_df = clustered_reviews[
    clustered_reviews["cluster"] == selected_cluster
]

cluster_size = len(cluster_df)

sentiment_distribution = cluster_df["sentiment"].value_counts(normalize=True)

positive_pct = sentiment_distribution.get("positive", 0) * 100
neutral_pct = sentiment_distribution.get("neutral", 0) * 100
negative_pct = sentiment_distribution.get("negative", 0) * 100

col1, col2, col3, col4 = st.columns(4)

col1.metric("Reviews", cluster_size)
col2.metric("Positive", f"{positive_pct:.1f}%")
col3.metric("Neutral", f"{neutral_pct:.1f}%")
col4.metric("Negative", f"{negative_pct:.1f}%")

st.info(
    f"Cluster {selected_cluster} contains **{cluster_size} reviews**. "
    "Use the keywords and example reviews below to understand the main theme."
)

st.markdown("### Top Keywords")

st.write(", ".join(cluster_keywords[str(selected_cluster)]))

st.markdown("### Example Reviews")

sample_reviews = cluster_df.sample(
    min(5, len(cluster_df)),
    random_state=42,
)

for i, row in enumerate(sample_reviews.itertuples(), start=1):
    with st.expander(f"Review {i} — ⭐ {row.rating} ({row.sentiment.title()})"):
        if pd.notna(row.title):
            st.markdown(f"**{row.title}**")

    st.write(row.text)