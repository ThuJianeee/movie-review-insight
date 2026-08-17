"""Predict sentiment for your own sentences using a saved model."""

from __future__ import annotations

import argparse
import os

import joblib

from preprocess import clean_text

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
LABELS = {0: "NEGATIVE", 1: "POSITIVE"}


def load(model: str):
    path = os.path.join(MODEL_DIR, f"{model}.joblib")
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found - train the model first.")
    return joblib.load(path)


def predict(pipeline, text: str) -> str:
    return LABELS[int(pipeline.predict([clean_text(text)])[0])]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="tfidf_logreg",
                        choices=["naive_bayes", "tfidf_logreg", "tfidf_svm"])
    parser.add_argument("--text", default=None)
    args = parser.parse_args()

    pipe = load(args.model)
    if args.text:
        print(predict(pipe, args.text))
    else:
        print("Type a review (blank line to quit).")
        while True:
            line = input("> ").strip()
            if not line:
                break
            print(predict(pipe, line))