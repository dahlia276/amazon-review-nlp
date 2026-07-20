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