from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_embedding_model():
    return SentenceTransformer(MODEL_NAME)


def create_embeddings(model, texts):
    return model.encode(
        texts,
        show_progress_bar=True,
    )


def cluster_reviews(embeddings, n_clusters=5):
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init="auto",
    )

    labels = kmeans.fit_predict(embeddings)
    return labels, kmeans


def get_cluster_keywords(df, n_keywords=10):
    keywords = {}

    for cluster in sorted(df["cluster"].unique()):

        texts = df[df["cluster"] == cluster]["text"]

        vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=1000,
        )

        X = vectorizer.fit_transform(texts)

        scores = X.mean(axis=0).A1

        words = vectorizer.get_feature_names_out()

        top_indices = scores.argsort()[-n_keywords:][::-1]

        keywords[cluster] = [
            words[i]
            for i in top_indices
        ]

    return keywords

OUTPUT_DIR = Path("outputs/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def plot_clusters(embeddings, labels):
    pca = PCA(
        n_components=2,
        random_state=42,
    )
    reduced = pca.fit_transform(embeddings)

    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(
        reduced[:, 0],
        reduced[:, 1],
        c=labels,
        cmap="tab10",
        alpha=0.7,
    )

    plt.legend(
        *scatter.legend_elements(),
        title="Clusters",
    )
    plt.title("Review Clusters (PCA)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "clusters.png")
    plt.close()
    
def find_best_k(embeddings, min_k=2, max_k=8):
    scores = {}

    for k in range(min_k, max_k + 1):

        kmeans = KMeans(
            n_clusters=k,
            random_state=42,
            n_init="auto",
        )

        labels = kmeans.fit_predict(embeddings)

        score = silhouette_score(
            embeddings,
            labels,
        )

        scores[k] = score

    return scores

def plot_silhouette_scores(scores):
    plt.figure(figsize=(8, 5))

    plt.plot(
        list(scores.keys()),
        list(scores.values()),
        marker="o",
    )

    plt.title("Silhouette Score by Number of Clusters")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "silhouette_scores.png")
    plt.close()