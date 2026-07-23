from src.models import get_roberta_classifier

def predict(classifier, reviews):
    predictions = classifier(
        reviews,
        batch_size=16,
        truncation=True,
        max_length=512,
    )

    mapping = {
        "negative": "negative",
        "neutral": "neutral",
        "positive": "positive",
    }

    return [
        mapping[p["label"].lower()]
        for p in predictions
    ]


def predict_roberta(review: str) -> str:
    """
    Predict sentiment for a single review using RoBERTa.
    """

    classifier = get_roberta_classifier()

    prediction = classifier(
        review,
        truncation=True,
        max_length=512,
    )[0]

    return prediction["label"].lower()