"""Model 2 (Linear ML): Logistic Regression and Linear SVM on TF-IDF features."""

from __future__ import annotations

import argparse
import os
import time

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from evaluation import evaluate_model
from preprocess import clean_text
from split import get_splits

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")


def build(algorithm: str) -> Pipeline:
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000, sublinear_tf=True)
    if algorithm == "svm":
        clf = LinearSVC(C=1.0)
    else:
        clf = LogisticRegression(max_iter=1000, C=5.0)
    return Pipeline([("tfidf", vectorizer), ("clf", clf)])


def main(dataset: str, sample: int | None, algorithm: str) -> None:
    X_train, X_test, y_train, y_test = get_splits(dataset, sample)

    print("Preprocessing ...")
    X_train = [clean_text(t) for t in X_train]
    X_test = [clean_text(t) for t in X_test]

    pipeline = build(algorithm)
    start = time.time()
    pipeline.fit(X_train, y_train)
    elapsed = time.time() - start

    label = "Logistic Regression (TF-IDF)" if algorithm == "logreg" else "Linear SVM (TF-IDF)"
    evaluate_model(label, y_test, pipeline.predict(X_test), elapsed)

    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, f"tfidf_{algorithm}.joblib")
    joblib.dump(pipeline, path)
    print(f"Saved model to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cornell", choices=["cornell", "imdb"])
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--algorithm", default="logreg", choices=["logreg", "svm"])
    args = parser.parse_args()
    main(args.dataset, args.sample, args.algorithm)