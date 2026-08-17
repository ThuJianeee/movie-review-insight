"""Run the whole benchmark end-to-end and produce the comparison chart."""

from __future__ import annotations

import argparse

import evaluation
import model1_naive_bayes
import model2_tfidf_linear


def main(dataset: str, sample: int | None, with_bert: bool) -> None:
    model1_naive_bayes.main(dataset, sample)
    model2_tfidf_linear.main(dataset, sample, "logreg")
    model2_tfidf_linear.main(dataset, sample, "svm")
    if with_bert:
        import model3_distilbert

        model3_distilbert.main(dataset, sample, epochs=2, batch_size=16, max_len=256, lr=2e-5)
    evaluation.plot_comparison()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cornell", choices=["cornell", "imdb"])
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--with_bert", action="store_true", help="Also fine-tune DistilBERT")
    args = parser.parse_args()
    main(args.dataset, args.sample, args.with_bert)