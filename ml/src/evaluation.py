"""Shared evaluation helpers so all three models are scored identically."""

from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"
)
os.makedirs(RESULTS_DIR, exist_ok=True)


def evaluate_model(name: str, y_true, y_pred, train_seconds: float = 0.0) -> dict:
    metrics = {
        "model": name,
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, average="macro")), 4),
        "recall": round(float(recall_score(y_true, y_pred, average="macro")), 4),
        "f1": round(float(f1_score(y_true, y_pred, average="macro")), 4),
        "train_seconds": round(float(train_seconds), 2),
    }

    print(f"\n=== {name} ===")
    print(classification_report(y_true, y_pred, target_names=["negative", "positive"], digits=4))
    print(f"Training time: {metrics['train_seconds']}s")

    _plot_confusion(name, y_true, y_pred)
    _append_metrics(metrics)
    return metrics


def _plot_confusion(name: str, y_true, y_pred) -> None:
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4, 3.4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["negative", "positive"], yticklabels=["negative", "positive"],
    )
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, f"confusion_{name.lower().replace(' ', '_')}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def _append_metrics(metrics: dict) -> None:
    csv_path = os.path.join(RESULTS_DIR, "metrics.csv")
    df = pd.read_csv(csv_path) if os.path.exists(csv_path) else pd.DataFrame()
    df = df[df.get("model", pd.Series(dtype=str)) != metrics["model"]] if len(df) else df
    df = pd.concat([df, pd.DataFrame([metrics])], ignore_index=True)
    df.to_csv(csv_path, index=False)
    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as fh:
        json.dump(df.to_dict(orient="records"), fh, indent=2)
    print(f"Saved {csv_path}")


def plot_comparison() -> None:
    """Bar chart comparing every model recorded so far (for section 4.1)."""
    csv_path = os.path.join(RESULTS_DIR, "metrics.csv")
    if not os.path.exists(csv_path):
        print("No metrics yet - train a model first.")
        return
    df = pd.read_csv(csv_path)
    melted = df.melt(
        id_vars="model", value_vars=["accuracy", "precision", "recall", "f1"],
        var_name="metric", value_name="score",
    )
    plt.figure(figsize=(8, 4.5))
    sns.barplot(data=melted, x="metric", y="score", hue="model")
    plt.ylim(0, 1)
    plt.title("Model comparison")
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "model_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(df.to_string(index=False))
    print(f"Saved {path}")


if __name__ == "__main__":
    plot_comparison()