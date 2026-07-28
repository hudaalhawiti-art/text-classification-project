"""
Predict the label of a new text sample using the trained model.
Usage: python src/predict.py "your text here"
"""

import sys
import os
import joblib

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model.joblib")
VECTORIZER_PATH = os.path.join(os.path.dirname(__file__), "..", "vectorizer.joblib")


def main():
    if len(sys.argv) < 2:
        print('Usage: python src/predict.py "your text here"')
        sys.exit(1)

    text = " ".join(sys.argv[1:])

    if not os.path.exists(MODEL_PATH):
        print("Model not found. Run 'python src/train.py' first.")
        sys.exit(1)

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)[0]
    proba = model.predict_proba(text_vec).max()

    print(f"Text: {text}")
    print(f"Predicted label: {prediction} (confidence: {proba:.2f})")


if __name__ == "__main__":
    main()
