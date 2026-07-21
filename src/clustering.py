from src.models import get_embedding_model


def generate_embeddings(df):
    model = get_embedding_model()

    embeddings = model.encode(
        df["text"].tolist(),
        show_progress_bar=True,
    )

    return embeddings