"""
Cyberbullying Impact Analyzer - Model Training Script
-------------------------------------------------------
Dataset: "Cyberbullying Classification" (Kaggle)
https://www.kaggle.com/datasets/andrewmvd/cyberbullying-classification

Expected CSV columns:
    tweet_text          -> the raw text
    cyberbullying_type  -> one of:
        age, ethnicity, gender, religion,
        other_cyberbullying, not_cyberbullying

Usage:
    1. Download the CSV from Kaggle and place it at:
       data/cyberbullying_tweets.csv
    2. Run:  python train_model.py
    3. Outputs saved to model/ :
       - model.pkl        (trained classifier)
       - vectorizer.pkl   (TF-IDF vectorizer)
       - label_encoder.pkl
       - confusion_matrix.png
"""

import re
import string
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

DATA_PATH = "data/cyberbullying_tweets.csv"
MODEL_DIR = "model"


def clean_text(text: str) -> str:
    """Basic tweet cleaning: lowercase, remove URLs, mentions, hashtags, punctuation, numbers."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)          # URLs
    text = re.sub(r"@\w+", " ", text)                      # mentions
    text = re.sub(r"#", " ", text)                          # hashtag symbol (keep the word)
    text = re.sub(r"[^a-z\s]", " ", text)                   # numbers/punctuation
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main():
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    # Standardize expected column names if slightly different
    df = df.rename(columns={c: c.strip() for c in df.columns})
    text_col = "tweet_text" if "tweet_text" in df.columns else df.columns[0]
    label_col = "cyberbullying_type" if "cyberbullying_type" in df.columns else df.columns[1]

    df = df[[text_col, label_col]].dropna()
    df.columns = ["text", "label"]

    print(f"Rows loaded: {len(df)}")
    print(df["label"].value_counts())

    print("Cleaning text...")
    df["clean_text"] = df["text"].apply(clean_text)
    df = df[df["clean_text"].str.len() > 0]

    print("Encoding labels...")
    le = LabelEncoder()
    y = le.fit_transform(df["label"])

    print("Splitting train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"], y, test_size=0.2, random_state=42, stratify=y
    )

    print("Vectorizing (TF-IDF)...")
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), stop_words="english")
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print("Training Logistic Regression classifier...")
    clf = LogisticRegression(max_iter=1000, C=2.0, multi_class="ovr")
    clf.fit(X_train_vec, y_train)

    print("Evaluating...")
    y_pred = clf.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax, xticks_rotation=45, cmap="Blues", colorbar=False)
    plt.title(f"Confusion Matrix (Accuracy: {acc:.2%})")
    plt.tight_layout()
    plt.savefig(f"{MODEL_DIR}/confusion_matrix.png", dpi=150)
    print(f"Saved confusion matrix to {MODEL_DIR}/confusion_matrix.png")

    print("Saving model artifacts...")
    joblib.dump(clf, f"{MODEL_DIR}/model.pkl")
    joblib.dump(vectorizer, f"{MODEL_DIR}/vectorizer.pkl")
    joblib.dump(le, f"{MODEL_DIR}/label_encoder.pkl")
    print("Done. Model, vectorizer, and label encoder saved to model/")


if __name__ == "__main__":
    main()
