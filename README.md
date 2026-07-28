# Text Classification with Machine Learning

A lightweight, well-structured NLP project that classifies text into categories using classic ML (TF-IDF + Logistic Regression). Built to demonstrate a clean, reproducible text classification workflow.

## 📁 Project Structure
```
text-classification-project/
├── data/
│   └── sample_data.csv      # Small example dataset (included)
├── src/
│   ├── train.py             # Trains and evaluates the model
│   └── predict.py           # Predicts the label of new text
├── requirements.txt
└── README.md
```

## ⚙️ How It Works
1. **Preprocessing:** Text is converted into numeric features using TF-IDF vectorization.
2. **Training:** A Logistic Regression classifier is trained on the labeled dataset.
3. **Evaluation:** Accuracy, precision, recall, and F1-score are reported on a held-out test split.
4. **Prediction:** The trained model classifies new, unseen text samples.

## 🚀 Getting Started

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the model
python src/train.py

# 3. Predict on new text
python src/predict.py "your text here"
```

## 📊 Results

> ⚠️ **Note:** `data/sample_data.csv` is a tiny demo file included so the pipeline runs out of the box. The results below were produced by swapping in the full IMDB dataset — see [Getting Started](#-getting-started) to do the same.

Trained and evaluated on the [IMDB 50K Movie Reviews](https://ai.stanford.edu/~amaas/data/sentiment/) dataset (binary sentiment: positive/negative), 70/30 train-test split:

```
Accuracy: 0.90

              precision    recall  f1-score   support

    negative       0.90      0.88      0.89      7500
    positive       0.89      0.91      0.90      7500

    accuracy                           0.90     15000
   macro avg       0.90      0.90      0.89     15000
weighted avg       0.90      0.90      0.89     15000
```

A smaller `data/sample_data.csv` is included in the repo so the pipeline runs out of the box; swap in a larger dataset (like IMDB above) for production-quality accuracy.

## 🛠️ Tools & Libraries
- Python
- scikit-learn
- pandas
- TF-IDF (feature extraction)

## 📌 Notes / Next Steps
- Swap `data/sample_data.csv` for a larger labeled dataset to scale up
- Try n-grams (`ngram_range=(1,2)`) or alternative models (Naive Bayes, Linear SVM) for comparison
- Could be extended with deep learning (e.g. fine-tuned BERT via Hugging Face `transformers`) for higher accuracy on harder datasets

## 👩‍💻 Author
**Huda Alhawiti** — Computer Engineer | AI & Machine Learning
[LinkedIn](https://linkedin.com/in/huda-h-alhawiti)
