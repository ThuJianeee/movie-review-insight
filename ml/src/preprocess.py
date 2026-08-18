"""Shared text preprocessing pipeline (Objective 1 in the documentation).

Steps: HTML stripping -> lowercasing -> punctuation/number removal ->
       tokenisation -> stop-word removal -> lemmatisation.
"""

from __future__ import annotations

import re

import nltk
from bs4 import BeautifulSoup
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize


def ensure_nltk() -> None:
    """Download the NLTK resources we rely on (safe to call repeatedly)."""
    for pkg in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4",
                "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng"]:
        try:
            nltk.download(pkg, quiet=True)
        except Exception:  # offline: rely on whatever is already cached
            pass


ensure_nltk()

_LEMMATIZER = WordNetLemmatizer()
_STEMMER = PorterStemmer()
_STOPWORDS = set(stopwords.words("english"))
# Negation words carry sentiment ("not good"), so we keep them.
_KEEP = {"not", "no", "nor", "never", "n't", "against", "very", "too", "but"}
_STOPWORDS -= _KEEP

_HTML_RE = re.compile(r"<[^>]+>")
_NON_ALPHA_RE = re.compile(r"[^a-z\s']")
_SPACE_RE = re.compile(r"\s+")


def _wordnet_pos(tag: str):
    if tag.startswith("J"):
        return wordnet.ADJ
    if tag.startswith("V"):
        return wordnet.VERB
    if tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN


def clean_text(
    text: str,
    lemmatize: bool = True,
    remove_stopwords: bool = True,
    stem: bool = False,
) -> str:
    """Full cleaning pipeline used by the classical ML models."""
    if not isinstance(text, str):
        return ""

    # 1. Strip HTML markup (IMDb reviews contain <br /> tags)
    if "<" in text:
        text = BeautifulSoup(text, "html.parser").get_text(" ")
    text = _HTML_RE.sub(" ", text)

    # 2. Lowercase, 3. drop digits/punctuation
    text = text.lower()
    text = _NON_ALPHA_RE.sub(" ", text)
    text = _SPACE_RE.sub(" ", text).strip()

    # 4. Tokenise
    tokens = word_tokenize(text)

    # 5. Stop-word removal (negations preserved) + short-token removal
    if remove_stopwords:
        tokens = [t for t in tokens if t not in _STOPWORDS and len(t) > 1]

    # 6. POS-aware lemmatisation
    if lemmatize and tokens:
        try:
            tagged = nltk.pos_tag(tokens)
            tokens = [_LEMMATIZER.lemmatize(tok, _wordnet_pos(tag)) for tok, tag in tagged]
        except Exception:
            tokens = [_LEMMATIZER.lemmatize(tok) for tok in tokens]

    # 7. Optional Porter stemming (comparison experiment: stemming vs lemmatisation)
    if stem and tokens:
        tokens = [_STEMMER.stem(tok) for tok in tokens]

    return " ".join(tokens)


def clean_for_transformer(text: str) -> str:
    """DistilBERT needs raw-ish text: only HTML and whitespace are normalised."""
    if not isinstance(text, str):
        return ""
    if "<" in text:
        text = BeautifulSoup(text, "html.parser").get_text(" ")
    return _SPACE_RE.sub(" ", text).strip()


def clean_series(series, lemmatize: bool = True):
    return series.astype(str).map(lambda t: clean_text(t, lemmatize=lemmatize))


if __name__ == "__main__":
    sample = "<br />This movie was NOT good at all... 10/10 acting though!"
    print("RAW  :", sample)
    print("CLEAN:", clean_text(sample))