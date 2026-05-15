"""
This Python file is example of how your `pred.py` script should
look. Your file should contain a function `predict_all` that takes
in the name of a CSV file, and returns a list of predictions.

Your `pred.py` script can use different methods to process the input
data, but the format of the input it takes and the output your script produces should be the same.

Here's an example of how your script may be used in our test file:

    from example_pred import predict_all
    predict_all("example_test_set.csv")
"""

import sys
import csv
import random
import numpy as np
import pandas as pd


# FILE NAMES

WORD_VOCAB_FILE = "word_vocab.csv"
WORD_IDF_FILE = "word_idf.npy"
CHAR_VOCAB_FILE = "char_vocab.csv"
CHAR_IDF_FILE = "char_idf.npy"
WEIGHTS_FILE = "weights.npy"
BIAS_FILE = "bias.npy"
LABEL_MAP_FILE = "label_map.csv"


# EXACT SURVEY COLUMNS

TEXT_COLUMNS = [
    "Describe how this painting makes you feel.",
    "If this painting was a food, what would be?",
    "Imagine a soundtrack for this painting. Describe that soundtrack without naming any objects in the painting."
]

NUMERIC_COLUMNS = [
    "On a scale of 1–10, how intense is the emotion conveyed by the artwork?",
    "How many prominent colours do you notice in this painting?",
    "How many objects caught your eye in the painting?",
    "How much (in Canadian dollars) would you be willing to pay for this painting?"
]

LIKERT_COLUMNS = [
    "This art piece makes me feel sombre.",
    "This art piece makes me feel content.",
    "This art piece makes me feel calm.",
    "This art piece makes me feel uneasy."
]

MULTI_COLUMNS = {
    "If you could purchase this painting, which room would you put that painting in?": [
        "bedroom", "bathroom", "office", "living room", "dining room"
    ],
    "If you could view this art in person, who would you want to view it with?": [
        "friends", "family members", "coworkers/classmates", "strangers", "by yourself"
    ],
    "What season does this art piece remind you of?": [
        "spring", "summer", "fall", "winter"
    ]
}

# GLOBAL MODEL CACHE
MODEL = None


# HELPERS

def safe_str(x):
    if pd.isna(x):
        return ""
    return str(x)


def clean_text(s):
    s = safe_str(s).lower()

    bad_chars = [
        ",", ".", "!", "?", ":", ";", "(", ")", "[", "]", "{", "}",
        '"', "'", "$", "%", "&", "/", "\\", "-", "_", "=", "+", "*",
        "#", "@", "^", "~", "`", "|", "<", ">"
    ]
    for ch in bad_chars:
        s = s.replace(ch, " ")

    s = " ".join(s.split())
    return s


def split_multi(x):
    s = safe_str(x)
    if s.strip() == "":
        return []
    return [p.strip().lower() for p in s.split(",") if p.strip() != ""]


def parse_numeric(x):
    s = safe_str(x).strip()
    if s == "":
        return 0.0
    try:
        return float(s)
    except:
        return 0.0


def parse_price(x):
    s = safe_str(x).lower().strip()

    if s == "":
        return 0.0

    # normalize separators / words
    s = s.replace(",", "")
    s = s.replace("$", "")
    s = s.replace("cad", "")
    s = s.replace("dollars", "")
    s = s.replace("dollar", "")

    # collapse spaces between digits: "5 000 000" -> "5000000"
    out = []
    prev_digit = False
    for ch in s:
        if ch.isdigit():
            out.append(ch)
            prev_digit = True
        elif ch == ".":
            out.append(ch)
            prev_digit = False
        elif ch == " " and prev_digit:
            # maybe separator between digits, skip for now
            continue
        else:
            out.append(" ")
            prev_digit = False

    s2 = "".join(out)
    tokens = s2.split()

    nums = []
    for tok in tokens:
        try:
            float(tok)
            nums.append(tok)
        except:
            pass

    if len(nums) == 0:
        return 0.0

    try:
        val = float(nums[0])
    except:
        val = 0.0

    return np.log1p(val)


def parse_likert(x):
    s = safe_str(x).strip()
    if len(s) == 0:
        return 3.0

    if s[0].isdigit():
        try:
            v = int(s[0])
            if 1 <= v <= 5:
                return float(v)
        except:
            pass

    s_low = s.lower()
    if "strongly disagree" in s_low:
        return 1.0
    if "disagree" in s_low and "strongly" not in s_low:
        return 2.0
    if "neutral" in s_low or "unsure" in s_low:
        return 3.0
    if "agree" in s_low and "strongly" not in s_low:
        return 4.0
    if "strongly agree" in s_low:
        return 5.0

    return 3.0


# TEXT BUILDING

def build_combined_text_from_row(row):
    pieces = []

    for col in TEXT_COLUMNS:
        if col in row:
            val = safe_str(row[col]).strip()
            if val != "":
                pieces.append(val)

    # also append multi-select values into text
    for col in MULTI_COLUMNS:
        if col in row:
            vals = split_multi(row[col])
            if len(vals) > 0:
                pieces.append(" ".join(vals))

    return clean_text(" ".join(pieces))


# VOCAB LOADING

def load_vocab_csv(filename):
    vocab = {}
    with open(filename, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            token = row["token"]
            idx = int(row["index"])
            vocab[token] = idx
    return vocab


def load_label_map(filename):
    labels = []
    with open(filename, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = sorted(reader, key=lambda r: int(r["index"]))
        for row in rows:
            labels.append(row["label"])
    return labels


# ============================================================
# TF-IDF BUILDERS
# ============================================================

def tokenize_words(text):
    text = clean_text(text)
    if text == "":
        return []
    return text.split()


def build_word_ngrams(tokens):
    feats = []
    for tok in tokens:
        feats.append(tok)
    for i in range(len(tokens) - 1):
        feats.append(tokens[i] + " " + tokens[i + 1])
    return feats


def tokenize_char_ngrams(text, min_n=3, max_n=5):
    text = clean_text(text)
    if text == "":
        return []

    words = text.split()
    ngrams = []

    for word in words:
        padded = " " + word + " "
        L = len(padded)
        for n in range(min_n, max_n + 1):
            for i in range(L - n + 1):
                ngrams.append(padded[i:i+n])

    return ngrams


def tfidf_from_vocab(texts, vocab, idf, mode="word"):
    n = len(texts)
    d = len(vocab)
    X = np.zeros((n, d), dtype=np.float32)

    for i, text in enumerate(texts):
        if mode == "word":
            feats = build_word_ngrams(tokenize_words(text))
        else:
            feats = tokenize_char_ngrams(text, 3, 5)

        counts = {}
        for feat in feats:
            if feat in vocab:
                idx = vocab[feat]
                counts[idx] = counts.get(idx, 0) + 1

        if len(feats) == 0:
            continue

        total = float(sum(counts.values()))
        if total <= 0:
            continue

        for idx, cnt in counts.items():
            tf = cnt / total
            X[i, idx] = tf * idf[idx]

        # L2 normalize row
        norm = np.sqrt(np.sum(X[i] * X[i]))
        if norm > 0:
            X[i] /= norm

    return X


# STRUCTURED FEATURES

def get_structured_feature_order():
    feature_names = []

    for col in NUMERIC_COLUMNS:
        feature_names.append("num__" + col)

    for col in LIKERT_COLUMNS:
        feature_names.append("likert__" + col)

    for col, options in MULTI_COLUMNS.items():
        for option in options:
            feature_names.append("multi__" + col + "__" + option)

    return feature_names


def build_structured_features(df):
    feature_order = get_structured_feature_order()
    feat_idx = {name: i for i, name in enumerate(feature_order)}

    n = len(df)
    d = len(feature_order)
    X = np.zeros((n, d), dtype=np.float32)

    for row_i, (_, row) in enumerate(df.iterrows()):

        # numeric
        for col in NUMERIC_COLUMNS:
            feat_name = "num__" + col

            if col == "How much (in Canadian dollars) would you be willing to pay for this painting?":
                val = parse_price(row[col]) if col in df.columns else 0.0
            else:
                val = parse_numeric(row[col]) if col in df.columns else 0.0

            X[row_i, feat_idx[feat_name]] = val

        # likert
        for col in LIKERT_COLUMNS:
            feat_name = "likert__" + col
            val = parse_likert(row[col]) if col in df.columns else 3.0
            X[row_i, feat_idx[feat_name]] = val

        # multi-hot
        for col, options in MULTI_COLUMNS.items():
            vals = set(split_multi(row[col])) if col in df.columns else set()
            for option in options:
                feat_name = "multi__" + col + "__" + option
                X[row_i, feat_idx[feat_name]] = 1.0 if option in vals else 0.0

    return X


# MODEL LOADING

def load_model():
    global MODEL
    if MODEL is not None:
        return MODEL

    word_vocab = load_vocab_csv(WORD_VOCAB_FILE)
    word_idf = np.load(WORD_IDF_FILE)

    char_vocab = load_vocab_csv(CHAR_VOCAB_FILE)
    char_idf = np.load(CHAR_IDF_FILE)

    W = np.load(WEIGHTS_FILE)
    b = np.load(BIAS_FILE)

    labels = load_label_map(LABEL_MAP_FILE)

    MODEL = {
        "word_vocab": word_vocab,
        "word_idf": word_idf,
        "char_vocab": char_vocab,
        "char_idf": char_idf,
        "W": W,
        "b": b,
        "labels": labels
    }
    return MODEL


# PREDICTION

def predict_all(filename):
    """
    Takes the name of a CSV file and returns a list of predictions.
    """
    model = load_model()

    df = pd.read_csv(filename)

    # Build text
    texts = []
    for _, row in df.iterrows():
        texts.append(build_combined_text_from_row(row))

    # Build features
    X_word = tfidf_from_vocab(
        texts,
        model["word_vocab"],
        model["word_idf"],
        mode="word"
    )

    X_char = tfidf_from_vocab(
        texts,
        model["char_vocab"],
        model["char_idf"],
        mode="char"
    )

    X_struct = build_structured_features(df)

    X = np.concatenate([X_word, X_char, X_struct], axis=1)

    # Linear scores
    scores = X @ model["W"] + model["b"]

    # Pick the highest score
    pred_idx = np.argmax(scores, axis=1)

    predictions = [model["labels"][i] for i in pred_idx]
    return predictions
