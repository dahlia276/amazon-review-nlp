from src.config import DATA_PATH
from src.evaluation import evaluate_classifier
from src.preprocessing import preprocess_data
from src.transformer_classifier import (
    load_classifier,
    predict,
)


def main():
    df = preprocess_data(DATA_PATH)

    # Use a subset for faster inference
    df = df.sample(
        2000,
        random_state=42,
    )

    classifier = load_classifier()

    predictions = predict(
        classifier,
        df["text"].tolist(),
    )

    evaluate_classifier(
        df["sentiment"],
        predictions,
    )

    print("\nConfusion matrix saved to outputs/figures/")


if __name__ == "__main__":
    main()