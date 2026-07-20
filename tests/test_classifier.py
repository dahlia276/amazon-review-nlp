from src.classifier import save_model, train_classifier
from src.config import DATA_PATH
from src.preprocessing import preprocess_data


def main():
    df = preprocess_data(DATA_PATH)

    model, vectorizer = train_classifier(df)

    save_model(model, vectorizer)

    print("\nModel saved successfully.")


if __name__ == "__main__":
    main()