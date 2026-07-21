import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from src.config import DATA_PATH
from src.preprocessing import preprocess_data


def main():
    df = preprocess_data(DATA_PATH)

    print(df.head())
    print()
    print(df.info())
    print()
    print(df["sentiment"].value_counts())


if __name__ == "__main__":
    main()