"""Dataset loading for the sentiment analysis project.

Two sources are supported:
  1. "cornell"  -> Cornell polarity dataset v2.0 (2000 movie reviews, 1000 pos / 1000 neg)
                   http://www.cs.cornell.edu/people/pabo/movie-review-data/
  2. "imdb"     -> Large IMDb dataset (50,000 reviews) via HuggingFace `datasets`

Both return a pandas DataFrame with columns: text, label (0 = negative, 1 = positive)
"""

from __future__ import annotations

import io
import os
import tarfile
import urllib.request

import pandas as pd

CORNELL_URL = (
    "http://www.cs.cornell.edu/people/pabo/movie-review-data/review_polarity.tar.gz"
)
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def _download_cornell() -> str:
    """Download + extract the Cornell polarity dataset. Returns the extracted folder."""
    os.makedirs(DATA_DIR, exist_ok=True)
    target = os.path.join(DATA_DIR, "txt_sentoken")
    if os.path.isdir(target):
        return target

    print("Downloading Cornell polarity dataset v2.0 ...")
    with urllib.request.urlopen(CORNELL_URL) as response:
        raw = response.read()
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tar:
        tar.extractall(DATA_DIR)
    print(f"Extracted to {target}")
    return target


def load_cornell() -> pd.DataFrame:
    root = _download_cornell()
    rows = []
    for label_name, label in (("neg", 0), ("pos", 1)):
        folder = os.path.join(root, label_name)
        for filename in sorted(os.listdir(folder)):
            with open(os.path.join(folder, filename), encoding="utf-8", errors="ignore") as fh:
                rows.append({"text": fh.read(), "label": label})
    return pd.DataFrame(rows).sample(frac=1.0, random_state=42).reset_index(drop=True)


def load_imdb() -> pd.DataFrame:
    from datasets import load_dataset  # imported lazily: only needed for this source

    ds = load_dataset("imdb")
    frames = [ds["train"].to_pandas(), ds["test"].to_pandas()]
    df = pd.concat(frames, ignore_index=True)[["text", "label"]]
    return df.sample(frac=1.0, random_state=42).reset_index(drop=True)


def load_crawled(filename: str = "crawled_reviews.csv") -> pd.DataFrame:
    """Reviews collected by src/crawler.py (assignment requirement 'c')."""
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run `python crawler.py` first to crawl reviews."
        )
    df = pd.read_csv(path)[["text", "label"]].dropna()
    df["label"] = df["label"].astype(int)
    return df.sample(frac=1.0, random_state=42).reset_index(drop=True)


def load_dataset_by_name(name: str = "cornell") -> pd.DataFrame:
    name = name.lower()
    if name == "cornell":
        return load_cornell()
    if name == "imdb":
        return load_imdb()
    if name == "crawled":
        return load_crawled()
    raise ValueError(f"Unknown dataset '{name}'. Use 'cornell', 'imdb' or 'crawled'.")


if __name__ == "__main__":
    frame = load_dataset_by_name("cornell")
    print(frame.shape)
    print(frame.label.value_counts())
    print(frame.head())