"""
features.py
Creates numerical features from question pairs using Word2Vec
"""

import numpy as np
from fuzzywuzzy import fuzz
from sklearn.metrics.pairwise import cosine_similarity
from utils import preprocess_text


# ---------------------------------------------------
# Basic statistical features
# ---------------------------------------------------
def basic_features(q1, q2):
    """
    Length, overlap, and ratio-based features
    """
    q1_words = q1.split()
    q2_words = q2.split()

    common = len(set(q1_words) & set(q2_words))

    return [
        len(q1_words),
        len(q2_words),
        len(q1),
        len(q2),
        common,
        common / (min(len(q1_words), len(q2_words)) + 1)
    ]


# ---------------------------------------------------
# Fuzzy string similarity features
# ---------------------------------------------------
def fuzzy_features(q1, q2):
    """
    String similarity features
    """
    return [
        fuzz.ratio(q1, q2),
        fuzz.partial_ratio(q1, q2),
        fuzz.token_sort_ratio(q1, q2),
        fuzz.token_set_ratio(q1, q2)
    ]


# ---------------------------------------------------
# Word2Vec sentence embedding
# ---------------------------------------------------
def sentence_vector(sentence, w2v_model):
    """
    Average Word2Vec embedding for a sentence
    """
    vectors = []

    for word in sentence.split():
        if word in w2v_model.wv:
            vectors.append(w2v_model.wv[word])

    if len(vectors) == 0:
        return np.zeros(w2v_model.vector_size)

    return np.mean(vectors, axis=0)


# ---------------------------------------------------
# Final feature builder (USED IN TRAIN + INFERENCE)
# ---------------------------------------------------
def build_features(q1_raw, q2_raw, w2v_model):
    """
    Builds final feature vector for a question pair
    """

    # Preprocess text
    q1 = preprocess_text(q1_raw)
    q2 = preprocess_text(q2_raw)

    features = []

    # Basic + fuzzy features
    features.extend(basic_features(q1, q2))
    features.extend(fuzzy_features(q1, q2))

    # Word2Vec embeddings
    q1_vec = sentence_vector(q1, w2v_model)
    q2_vec = sentence_vector(q2, w2v_model)

    # Distance-based features
    features.append(cosine_similarity([q1_vec], [q2_vec])[0][0])
    features.append(np.linalg.norm(q1_vec - q2_vec))

    # Append raw embeddings
    features.extend(q1_vec)
    features.extend(q2_vec)

    return np.array(features).reshape(1, -1)
