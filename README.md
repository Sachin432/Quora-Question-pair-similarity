# Quora Question Pair Similarity Detection

This project explores the challenge of identifying whether two questions on Quora mean the same thing, even if they are worded differently. With thousands of similar queries being asked every day, this problem is important to improve the quality of content and user experience on platforms like Quora.

##  Objective

The goal is to build a machine learning model that can accurately classify whether a pair of questions are duplicates. This can help avoid redundancy and direct users to existing answers.

##  Dataset

- Source: [Kaggle - Quora Question Pairs](https://www.kaggle.com/c/quora-question-pairs)
- Format: CSV file with:
  - `qid1`, `qid2`: Unique IDs for the questions
  - `question1`, `question2`: The actual question text
  - `is_duplicate`: Label (1 = duplicate, 0 = not duplicate)
- Size: ~404,000 rows

##  What’s in This Project

### 1. Exploratory Data Analysis
- Visual breakdown of duplicate vs non-duplicate pairs
- Word count statistics and distributions
- Word clouds for both duplicate and non-duplicate groups

### 2. Feature Engineering
- Word overlap
- Question length comparison
- Word share ratio
- Frequency of question IDs (to detect popular/repetitive ones)

### 3. Text Preprocessing
- Lowercasing and removing punctuation
- Expanding contractions (e.g. "can't" → "cannot")
- Removing stopwords
- Stemming

### 4. Advanced Features
- Fuzzy string similarity metrics:
  - `fuzz_ratio`
  - `partial_ratio`
  - `token_sort_ratio`
  - `token_set_ratio`
- Custom logic like matching first/last words, token overlap
- Word embeddings using SpaCy with TF-IDF weighting

### 5. Modeling
- A Gradient Boosting Classifier trained on engineered features
- Evaluation using precision, recall, and log loss

### 6. Visualization
- t-SNE plots for 2D and 3D visualization of question embeddings
- Feature distribution analysis

