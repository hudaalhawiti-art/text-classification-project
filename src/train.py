"""
Train a text classification model using TF-IDF + Logistic Regression.
Usage: python src/train.py
"""


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "IMDB_Dataset.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model.joblib")
VECTORIZER_PATH = os.path.join(os.path.dirname(__file__), "..", "vectorizer.joblib")


def main():
    # 1. Load data
    df = pd.read_csv(DATA_PATH)
    X, y = df["review"], df["sentiment"]

    # 2. Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # 3. Feature extraction (TF-IDF)
    vectorizer = TfidfVectorizer(stop_words="english")
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # 4. Train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)

    # 5. Evaluate
    y_pred = model.predict(X_test_vec)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}\n")
    print(classification_report(y_test, y_pred))

    # 6. Save model + vectorizer for later use in predict.py
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print(f"\nModel saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
