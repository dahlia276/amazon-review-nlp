import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


def train_classifier(df):
    """
    Train a TF-IDF + Logistic Regression sentiment classifier.
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

    print(classification_report(y_test, predictions))

    return model, vectorizer


def save_model(model, vectorizer):
    joblib.dump(model, "outputs/classifier.joblib")
    joblib.dump(vectorizer, "outputs/vectorizer.joblib")