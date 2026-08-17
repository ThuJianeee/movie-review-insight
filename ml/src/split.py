"""Single source of truth for the train/test split, so all models see identical data."""

from __future__ import annotations

from sklearn.model_selection import train_test_split

from data_loader import load_dataset_by_name

RANDOM_STATE = 42
TEST_SIZE = 0.2


def get_splits(dataset: str = "cornell", sample: int | None = None):
    df = load_dataset_by_name(dataset)
    if sample:
        df = df.head(sample)
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"].tolist(),
        df["label"].tolist(),
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["label"].tolist(),
    )
    print(f"Dataset={dataset}  train={len(X_train)}  test={len(X_test)}")
    return X_train, X_test, y_train, y_test