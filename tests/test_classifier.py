from pprint import pprint

from src.classifier import save_model, train_classifier
from src.config import DATA_PATH
from src.evaluation import evaluate_classifier
from src.preprocessing import preprocess_data


def main():
    df = preprocess_data(DATA_PATH)

    results = train_classifier(df)

    report = evaluate_classifier(
        results["y_test"],
        results["predictions"],
    )

    pprint(report)

    save_model(
        results["model"],
        results["vectorizer"],
    )

    print("\nModel saved successfully.")
    print("Confusion matrix saved to outputs/figures/")


if __name__ == "__main__":
    main()