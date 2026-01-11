"""
train.py
Train ML model using Word2Vec embeddings and save artifacts
"""

import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
from gensim.models import Word2Vec

from utils import preprocess_text
from features import build_features


# ---------------------------------------------------
# 1. Load dataset (only 15k rows for faster training)
# ---------------------------------------------------
df = pd.read_csv("train.csv").dropna()
df = df.sample(n=15000, random_state=42)


# ---------------------------------------------------
# 2. Text preprocessing
# ---------------------------------------------------
df["q1"] = df["question1"].apply(preprocess_text)
df["q2"] = df["question2"].apply(preprocess_text)


# ---------------------------------------------------
# 3. Train Word2Vec on Quora questions
# ---------------------------------------------------
sentences = pd.concat([df["q1"], df["q2"]]) \
                .apply(lambda x: x.split()) \
                .tolist()

w2v_model = Word2Vec(
    sentences=sentences,
    vector_size=300,
    window=5,
    min_count=2,
    workers=4,
    sg=1              # Skip-gram for better semantic learning
)


# ---------------------------------------------------
# 4. Feature matrix creation
# ---------------------------------------------------
X = []
y = df["is_duplicate"].values

for q1, q2 in zip(df["question1"], df["question2"]):
    X.append(build_features(q1, q2, w2v_model)[0])

X = np.array(X)


# ---------------------------------------------------
# 5. Train-test split
# ---------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ---------------------------------------------------
# 6. Model training (XGBoost)
# ---------------------------------------------------
model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss"
)

model.fit(X_train, y_train)


# ---------------------------------------------------
# 7. Save trained artifacts
# ---------------------------------------------------
model.save_model("model.json")        # lightweight & faster than pickle
w2v_model.save("w2v.model")

print("Training completed successfully.")
print("Saved files: model.json, w2v.model")
