# Background Study (report sections 1–2)

## 1. The chosen NLP problem
**Sentiment analysis (binary polarity classification) of movie reviews.**
Given a free-text review, the system predicts whether the writer's opinion is
POSITIVE or NEGATIVE. It is a supervised text-classification problem: input is a
variable-length document, output is one of two classes.

Data sources used in this project:
- Cornell polarity dataset v2.0 — 2,000 labelled movie reviews (Pang & Lee, 2004)
- Large IMDb dataset — 50,000 labelled reviews (Maas et al., 2011)
- Our own crawled IMDb user reviews (`src/crawler.py`), labelled from star ratings
  (>= 7 positive, <= 4 negative, 5–6 neutral discarded)

## 2. Significance and real-world applications
- **Brand and product monitoring** — companies track thousands of reviews/tweets per
  day; manual reading does not scale.
- **Customer support triage** — angry messages can be routed to human agents first.
- **Finance** — news and social sentiment feeds trading and risk models.
- **Public health / policy** — measuring public reaction to campaigns or events.
- **Recommendation and search ranking** — surfacing well-received items.
Automating this turns unstructured opinion text into a measurable signal, at a cost
and speed impossible for human annotators.

## 3. Common methods and techniques
| Family | Representation | Typical models | Notes |
|---|---|---|---|
| Lexicon-based | Sentiment dictionaries (VADER, SentiWordNet) | rule scoring | No training data needed; weak on sarcasm/negation |
| Bag-of-Words | Raw/count n-gram vectors | Multinomial Naïve Bayes | Fast, strong baseline, ignores word order |
| TF-IDF | Term weighting by rarity | Logistic Regression, Linear SVM | Usually the best classical option for long reviews |
| Static embeddings | Word2Vec, GloVe, FastText | averaged vectors + classifier, CNN/LSTM | Captures semantic similarity, one vector per word type |
| Contextual embeddings | BERT, DistilBERT, RoBERTa | fine-tuned transformer | State of the art; handles negation, context, sarcasm better; needs GPU |

**Preprocessing** normally shared by all of the above: HTML stripping, lowercasing,
punctuation/number removal, tokenisation, stop-word removal (keeping negation words
such as *not*, *never*), and stemming or lemmatisation. Transformers are fed
near-raw text because their sub-word tokenizer expects natural casing and morphology.

**Evaluation** for binary classification: Accuracy, Precision, Recall, F1 (macro
averaged here since classes are balanced), plus a confusion matrix. Training time is
also recorded to show the accuracy/cost trade-off between classical and deep models.

## 4. How this project maps to the assignment
| Requirement | Where it is implemented |
|---|---|
| (a) NLP problem | Sentiment analysis — this document, §1 |
| (b) Background study | This document, §1–3 |
| (c) Web crawler / reliable dataset | `src/crawler.py` (IMDb crawl) + `src/data_loader.py` (Cornell, IMDb) |
| (d) Preprocessing | `src/preprocess.py` — cleaning, tokenisation, stop-words, lemmatisation, optional Porter stemming; feature extraction in each model (BoW, TF-IDF, DistilBERT embeddings) |
| (e) Three different solutions | `model1_naive_bayes.py`, `model2_tfidf_linear.py` (LogReg + Linear SVM), `model3_distilbert.py` |
| (f) Comparison and metrics | `src/evaluation.py` → `results/metrics.csv`, `confusion_*.png`, `model_comparison.png` |

## References
- Pang, B., & Lee, L. (2004). *A sentimental education.* Proceedings of the ACL.
- Maas, A. L., et al. (2011). *Learning word vectors for sentiment analysis.* ACL.
- Sanh, V., et al. (2019). *DistilBERT, a distilled version of BERT.* arXiv:1910.01108.
- Devlin, J., et al. (2019). *BERT.* NAACL.
- Mikolov, T., et al. (2013). *Efficient estimation of word representations in vector space.* arXiv:1301.3781.
- Pedregosa, F., et al. (2011). *Scikit-learn.* JMLR, 12, 2825–2830.
