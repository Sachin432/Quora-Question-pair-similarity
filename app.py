"""
app.py
Streamlit inference app using Word2Vec
"""

import streamlit as st
from gensim.models import Word2Vec
import xgboost as xgb
from features import build_features


# ---------------------------------------------------
# Load trained artifacts
# ---------------------------------------------------
model = xgb.XGBClassifier()
model.load_model("model.json")          # XGBoost model

w2v_model = Word2Vec.load("w2v.model")  # Word2Vec embeddings


# ---------------------------------------------------
# Streamlit UI
# ---------------------------------------------------
st.set_page_config(page_title="Quora Similarity Detector")

st.title("Quora Question Pair Similarity")

q1 = st.text_area("Question 1")
q2 = st.text_area("Question 2")


# ---------------------------------------------------
# Inference
# ---------------------------------------------------
if st.button("Check Similarity"):
    if not q1.strip() or not q2.strip():
        st.warning("Both questions are required")
    else:
        features = build_features(q1, q2, w2v_model)
        prob = model.predict_proba(features)[0][1]

        st.subheader("Result")
        st.write(f"Duplicate Probability: {prob:.2f}")

        if prob > 0.5:
            st.success("Likely Duplicate Questions")
        else:
            st.error("Likely Not Duplicate")
