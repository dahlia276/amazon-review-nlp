import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from src.models import get_logistic_model

def train_classifier(df):
    """
    Train a TF-IDF + Logistic Regression classifier.
    """

    X = df["text"]
    y = df["sentiment"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    vectorizer = TfidfVectorizer(
        max_features=10000,
        stop_words="english",
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
    )

    model.fit(X_train_vec, y_train)

    predictions = model.predict(X_test_vec)

    return {
        "model": model,
        "vectorizer": vectorizer,
        "predictions": predictions,
        "y_test": y_test,
    }


def save_model(model, vectorizer):
    joblib.dump(model, "outputs/classifier.joblib")
    joblib.dump(vectorizer, "outputs/vectorizer.joblib")


def load_model():
    """
    Load the trained Logistic Regression model and TF-IDF vectorizer.
    """

    model = joblib.load("outputs/classifier.joblib")
    vectorizer = joblib.load("outputs/vectorizer.joblib")

    return model, vectorizer


def predict_logistic(review: str) -> str:
    model, vectorizer = get_logistic_model()

    review_vector = vectorizer.transform([review])

    prediction = model.predict(review_vector)[0]

    return prediction