"""Model 1 (Baseline): Multinomial Naive Bayes + Bag-of-Words."""

from __future__ import annotations

import argparse
import os
import time

import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from evaluation import evaluate_model
from preprocess import clean_text
from split import get_splits

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")


def main(dataset: str, sample: int | None) -> None:
    X_train, X_test, y_train, y_test = get_splits(dataset, sample)

    print("Preprocessing ...")
    X_train = [clean_text(t) for t in X_train]
    X_test = [clean_text(t) for t in X_test]

    pipeline = Pipeline([
        ("bow", CountVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000)),
        ("clf", MultinomialNB(alpha=1.0)),
    ])

    start = time.time()
    pipeline.fit(X_train, y_train)
    elapsed = time.time() - start

    evaluate_model("Naive Bayes (BoW)", y_test, pipeline.predict(X_test), elapsed)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(pipeline, os.path.join(MODEL_DIR, "naive_bayes.joblib"))
    print(f"Saved model to {MODEL_DIR}/naive_bayes.joblib")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cornell", choices=["cornell", "imdb"])
    parser.add_argument("--sample", type=int, default=None, help="Use only N reviews (quick test)")
    args = parser.parse_args()
    main(args.dataset, args.sample)