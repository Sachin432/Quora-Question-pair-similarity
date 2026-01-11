# Quora Question Pair Similarity Detection

🔗 **Live Web App Link (Streamlit App):**

[https://quora-question-pair-similarity-2tcezp2awbpxzjm9ui2oen.streamlit.app/](https://quora-question-pair-similarity-2tcezp2awbpxzjm9ui2oen.streamlit.app/)

---

## Overview

Online platforms like Quora receive thousands of questions daily, many of which are semantically identical but phrased differently. This project addresses the problem of **detecting duplicate question pairs** using **Natural Language Processing (NLP)** and **Machine Learning (ML)**.

The system predicts whether two questions have the **same underlying meaning**, even when their wording, structure, or vocabulary differs. This helps reduce redundancy, improve content quality, and enhance user experience.

---

## Objective

The primary objective is to build an **end-to-end machine learning system** that:

* Accurately classifies whether two questions are duplicates
* Uses **robust feature engineering** rather than relying on embeddings alone
* Is **lightweight, interpretable, and deployable**
* Supports **real-time inference via a Streamlit web app**

---

## Dataset

* **Source:** Kaggle – Quora Question Pairs
  [https://www.kaggle.com/c/quora-question-pairs](https://www.kaggle.com/c/quora-question-pairs)
* **Size:** ~404,000 question pairs
* **Columns:**

  * `qid1`, `qid2`: Question IDs
  * `question1`, `question2`: Question text
  * `is_duplicate`: Target label

    * `1` → Duplicate
    * `0` → Not Duplicate

During development, a **random sample of 15,000 rows** was used for faster experimentation.

---

## Project Workflow (End-to-End)

1. Data loading and sampling
2. Text preprocessing
3. Feature engineering
4. Model training and evaluation
5. Model serialization
6. Real-time inference using Streamlit

---

## Exploratory Data Analysis (EDA)

* Distribution of duplicate vs non-duplicate pairs
* Word count analysis for both questions
* Comparative length statistics
* Visualization of frequent and repetitive question patterns

EDA helped identify **data imbalance** and guided feature design.

---

## Text Preprocessing

Each question undergoes the same preprocessing during **training and inference**:

* Lowercasing
* HTML and punctuation removal
* Stopword removal
* Stemming

Example:

**Before**

```
How can I learn machine learning?
```

**After**

```
learn machin learn
```

This normalization improves feature consistency and reduces noise.

---

## Feature Engineering

The model uses a **hybrid feature approach**, combining lexical, syntactic, and semantic signals.

### 1. Basic Statistical Features

* Question length (words & characters)
* Common word count
* Word overlap ratio

### 2. Fuzzy Matching Features

Character-level similarity using:

* `fuzz.ratio`
* `partial_ratio`
* `token_sort_ratio`
* `token_set_ratio`

These features capture paraphrasing and reordering.

### 3. Semantic Features (Word2Vec)

* A **domain-specific Word2Vec model** trained on Quora questions
* Each question converted into a **300-dimensional vector**
* Sentence embedding created by averaging word vectors

### 4. Distance-Based Features

* Cosine similarity between question embeddings
* Euclidean distance between embeddings

---

### Final Feature Vector

```
Basic features          → 6
Fuzzy features          → 4
Distance features       → 2
Word2Vec (Q1 + Q2)      → 600
--------------------------------
Total features          → 612
```

---

## Modeling

* **Algorithm:** XGBoost Classifier

* **Why XGBoost:**

  * Handles high-dimensional tabular data well
  * Captures non-linear relationships
  * Robust to noisy and correlated features
  * Strong performance with engineered features

* **Evaluation Metrics:**

  * Log Loss (primary)
  * Precision
  * Recall
  * F1-Score

The model outputs a **probability score**, which is converted to a final decision using a configurable **threshold** (default: 0.5).

---

## Visualization & Analysis

* Feature distribution analysis
* Duplicate vs non-duplicate separability
* t-SNE visualization of question embeddings (2D / 3D)
* Validation of semantic clustering

These visualizations confirm that duplicate pairs cluster closer in embedding space.

---

## Deployment

The trained model is deployed as a **Streamlit web application**.

### Live Application

🔗 [https://quora-question-pair-similarity-2tcezp2awbpxzjm9ui2oen.streamlit.app/](https://quora-question-pair-similarity-2tcezp2awbpxzjm9ui2oen.streamlit.app/)

### App Features

* Input two questions
* Real-time similarity prediction
* Probability score output
* Adjustable decision threshold
* Clear duplicate / non-duplicate classification

---

## Tech Stack

* **Language:** Python
* **NLP:** NLTK, Word2Vec (Gensim), FuzzyWuzzy
* **ML:** Scikit-learn, XGBoost
* **Visualization:** Matplotlib, Seaborn
* **Deployment:** Streamlit

---


## Key Learnings

* Hybrid feature engineering often outperforms embeddings-only approaches
* Threshold tuning is as important as model accuracy
* Lightweight NLP pipelines are easier to deploy and explain
* Reusing the same preprocessing and features for training and inference is critical
