# automl/nlp_pipeline.py

import pandas as pd
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer

VECTORIZER_FOLDER = "vectorizers"


def detect_text_columns(df):

    text_cols = []

    for col in df.columns:

        if df[col].dtype == "object":

            avg_len = df[col].astype(str).str.len().mean()

            # ignore short strings (likely IDs)
            if avg_len > 30:
                text_cols.append(col)

    return text_cols


def apply_tfidf_training(df, text_cols):

    if not os.path.exists(VECTORIZER_FOLDER):
        os.makedirs(VECTORIZER_FOLDER)

    vectorizers = {}
    features = []

    for col in text_cols:

        tfidf = TfidfVectorizer(max_features=100)

        X = tfidf.fit_transform(df[col].astype(str)).toarray()

        X = pd.DataFrame(
            X,
            columns=[f"{col}_tfidf_{i}" for i in range(X.shape[1])]
        )

        # save vectorizer
        joblib.dump(tfidf, f"{VECTORIZER_FOLDER}/{col}_tfidf.pkl")

        vectorizers[col] = tfidf
        features.append(X)

    df = df.drop(columns=text_cols)

    if features:
        df = pd.concat([df.reset_index(drop=True)] + features, axis=1)

    return df


def apply_tfidf_prediction(df, text_cols):

    features = []

    for col in text_cols:

        path = f"{VECTORIZER_FOLDER}/{col}_tfidf.pkl"

        if os.path.exists(path):

            tfidf = joblib.load(path)

            X = tfidf.transform(df[col].astype(str)).toarray()

            X = pd.DataFrame(
                X,
                columns=[f"{col}_tfidf_{i}" for i in range(X.shape[1])]
            )

            features.append(X)

    df = df.drop(columns=text_cols)

    if features:
        df = pd.concat([df.reset_index(drop=True)] + features, axis=1)

    return df