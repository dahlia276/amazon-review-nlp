from transformers import pipeline


def load_classifier():
    return pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        truncation=True,
    )


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