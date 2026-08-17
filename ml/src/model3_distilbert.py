"""Model 3 (Deep Learning): fine-tuned DistilBERT."""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from evaluation import evaluate_model
from preprocess import clean_for_transformer
from split import get_splits

MODEL_NAME = "distilbert-base-uncased"
MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "distilbert"
)


class ReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len: int):
        self.encodings = tokenizer(
            texts, truncation=True, padding="max_length", max_length=max_len
        )
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


def main(dataset: str, sample: int | None, epochs: int, batch_size: int, max_len: int, lr: float):
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")
    if device == "cpu":
        print("WARNING: CPU training is slow. Try --sample 2000 --epochs 1 first, "
              "or run this script on Google Colab with a free GPU.")

    X_train, X_test, y_train, y_test = get_splits(dataset, sample)
    X_train = [clean_for_transformer(t) for t in X_train]
    X_test = [clean_for_transformer(t) for t in X_test]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(device)

    train_loader = DataLoader(
        ReviewDataset(X_train, y_train, tokenizer, max_len), batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        ReviewDataset(X_test, y_test, tokenizer, max_len), batch_size=batch_size
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, 0, total_steps)

    start = time.time()
    model.train()
    for epoch in range(epochs):
        running = 0.0
        for step, batch in enumerate(train_loader, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            outputs.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            running += outputs.loss.item()
            if step % 20 == 0:
                print(f"epoch {epoch + 1} step {step}/{len(train_loader)} loss {running / step:.4f}")
        print(f"Epoch {epoch + 1} average loss: {running / len(train_loader):.4f}")
    elapsed = time.time() - start

    model.eval()
    preds = []
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            preds.extend(np.argmax(logits.detach().cpu().numpy(), axis=1).tolist())

    evaluate_model("DistilBERT (fine-tuned)", y_test, preds, elapsed)

    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    print(f"Saved model to {MODEL_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cornell", choices=["cornell", "imdb"])
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()
    main(args.dataset, args.sample, args.epochs, args.batch_size, args.max_len, args.lr)