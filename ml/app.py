"""Simple Streamlit demo UI: paste a review, get a sentiment prediction.

Run with:  streamlit run app.py
"""

import os
import sys

import streamlit as st

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import joblib  # noqa: E402

from preprocess import clean_text  # noqa: E402

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
OPTIONS = {
    "Naive Bayes (BoW)": "naive_bayes.joblib",
    "Logistic Regression (TF-IDF)": "tfidf_logreg.joblib",
    "Linear SVM (TF-IDF)": "tfidf_svm.joblib",
}

st.set_page_config(page_title="Movie Review Sentiment Analysis", page_icon="🎬")
st.title("🎬 Sentiment Analysis of Movie Reviews")
st.caption("BMCS2003 Artificial Intelligence assignment demo")

choice = st.selectbox("Model", list(OPTIONS))
text = st.text_area("Paste a movie review", height=180)

if st.button("Analyse", type="primary"):
    path = os.path.join(MODEL_DIR, OPTIONS[choice])
    if not os.path.exists(path):
        st.error(f"Model not trained yet. Run the training script first ({path} missing).")
    elif not text.strip():
        st.warning("Please enter some text.")
    else:
        pipeline = joblib.load(path)
        cleaned = clean_text(text)
        label = int(pipeline.predict([cleaned])[0])
        if hasattr(pipeline, "predict_proba"):
            conf = float(pipeline.predict_proba([cleaned])[0][label])
            st.metric("Confidence", f"{conf:.1%}")
        st.success("POSITIVE 😀") if label == 1 else st.error("NEGATIVE 😞")
        with st.expander("Preprocessed text"):
            st.write(cleaned)