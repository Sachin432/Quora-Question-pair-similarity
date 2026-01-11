"""
utils.py
Text preprocessing utilities
Used in both training and inference
"""

import re
import nltk
from nltk.stem import PorterStemmer

# ---------------------------------------------------
# Safe stopwords loading (no crash if not downloaded)
# ---------------------------------------------------
try:
    from nltk.corpus import stopwords
    STOP_WORDS = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords")
    from nltk.corpus import stopwords
    STOP_WORDS = set(stopwords.words("english"))

STEMMER = PorterStemmer()


def preprocess_text(text: str) -> str:
    """
    Cleans raw text using standard NLP steps:
    - lowercasing
    - HTML removal
    - symbol removal
    - stopword removal
    - stemming
    """

    if text is None:
        return ""

    # Lowercase
    text = text.lower()

    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)

    # Remove non-alphanumeric characters
    text = re.sub(r"[^a-z0-9 ]", " ", text)

    tokens = []
    for word in text.split():
        if word not in STOP_WORDS:
            tokens.append(STEMMER.stem(word))

    return " ".join(tokens)
