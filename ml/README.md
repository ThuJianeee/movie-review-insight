# Sentiment Analysis of Text Reviews using Deep Learning

Code for the BMCS2003 / BMCS2203 / BMCS2074 Artificial Intelligence assignment.
Three models are benchmarked on the **same** train/test split:

| # | Model | Features | Script |
|---|-------|----------|--------|
| 1 | Multinomial Naive Bayes | Bag-of-Words | `src/model1_naive_bayes.py` |
| 2 | Logistic Regression / Linear SVM | TF-IDF | `src/model2_tfidf_linear.py` |
| 3 | Fine-tuned DistilBERT | Contextual embeddings | `src/model3_distilbert.py` |

Datasets: Cornell **polarity dataset v2.0** (2,000 reviews, downloads automatically)
or the large **IMDb** dataset (50,000 reviews, via HuggingFace).

---

## 1. Set up the Anaconda environment

Open **Anaconda Prompt** in this `ml` folder and run:

```bash
conda env create -f environment.yml
conda activate sentiment-ai
```

(If `torch` is slow to install and you only want models 1 and 2, you can delete the
`torch`/`transformers`/`datasets` lines from `environment.yml` first.)

## 2. Get the data

Nothing to download by hand — the first run fetches the Cornell dataset into `data/`.
To check it works:

```bash
cd src
python data_loader.py
```

### Optional: crawl your own data (assignment requirement c)

```bash
cd src
python crawler.py --titles tt0111161 tt0068646 --pages 3
```

This crawls IMDb user reviews (polite 1.5 s delay), maps star ratings to sentiment
labels (>=7 positive, <=4 negative, 5–6 discarded) and writes
`data/crawled_reviews.csv`. Every training script then accepts `--dataset crawled`.

## 3. Train and evaluate

```bash
cd src
python model1_naive_bayes.py                       # baseline
python model2_tfidf_linear.py --algorithm logreg   # linear ML
python model2_tfidf_linear.py --algorithm svm
python model3_distilbert.py --epochs 2             # deep learning (slow on CPU)
```

Or run everything and produce the comparison chart in one go:

```bash
python run_all.py                 # models 1 & 2
python run_all.py --with_bert     # all three
```

Useful flags: `--dataset imdb`, `--sample 2000` (quick test on fewer reviews).

**No GPU?** DistilBERT on CPU takes a long time. Either use
`python model3_distilbert.py --sample 2000 --epochs 1 --max_len 128`,
or upload `src/` to Google Colab and switch the runtime to GPU.

## 4. Outputs for your report

Everything lands in `results/`:

- `metrics.csv` / `metrics.json` — accuracy, precision, recall, F1, training time (section 3.4 / 4.1)
- `confusion_*.png` — one confusion matrix per model
- `model_comparison.png` — grouped bar chart of all models

Trained models are saved in `models/`.

## 5. Interactive demo

```bash
streamlit run app.py
```

Paste any review and the selected model predicts POSITIVE / NEGATIVE.
Screenshot this for the report. Or use the terminal version:

```bash
cd src
python predict.py --model tfidf_logreg --text "The plot was predictable and dull."
```

---

## Project structure

```
ml/
├── environment.yml            conda environment
├── app.py                     Streamlit demo UI
├── src/
│   ├── data_loader.py         downloads/loads Cornell or IMDb data
│   ├── preprocess.py          HTML strip, lowercase, stopwords, lemmatisation
│   ├── split.py               one shared 80/20 stratified split (seed 42)
│   ├── evaluation.py          metrics, confusion matrices, comparison chart
│   ├── model1_naive_bayes.py
│   ├── model2_tfidf_linear.py
│   ├── model3_distilbert.py
│   ├── run_all.py
│   └── predict.py
├── data/                      downloaded datasets (auto-created)
├── models/                    saved trained models (auto-created)
└── results/                   metrics + figures (auto-created)
```

## Data / tool sources (for the References section)

- Pang, B., & Lee, L. (2004). *A sentimental education: Sentiment analysis using subjectivity summarization based on minimum cuts.* Proceedings of the ACL. Dataset: http://www.cs.cornell.edu/people/pabo/movie-review-data/
- Maas, A. L., et al. (2011). *Learning word vectors for sentiment analysis.* Proceedings of the ACL. (IMDb dataset)
- Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). *DistilBERT, a distilled version of BERT.* arXiv:1910.01108
- Vaswani, A., et al. (2017). *Attention is all you need.* NeurIPS.
- Devlin, J., et al. (2019). *BERT: Pre-training of deep bidirectional transformers for language understanding.* NAACL.
- Pedregosa, F., et al. (2011). *Scikit-learn: Machine learning in Python.* JMLR, 12, 2825-2830.