from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)

OUTPUT_DIR = Path("outputs/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_classifier(y_true, y_pred):
    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred))

    # Dictionary version for later use
    report = classification_report(
        y_true,
        y_pred,
        output_dict=True,
    )

    cm = confusion_matrix(y_true, y_pred)

    display = ConfusionMatrixDisplay(confusion_matrix=cm)

    fig, ax = plt.subplots(figsize=(6, 6))

    display.plot(ax=ax)

    plt.title("Confusion Matrix")

    plt.tight_layout()

    plt.savefig(OUTPUT_DIR / "confusion_matrix.png")

    plt.close(fig)

    return report