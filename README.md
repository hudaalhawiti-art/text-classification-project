# Text Classification with Machine Learning

A simple, well-structured NLP project that classifies text into categories using classic ML models (TF-IDF + Logistic Regression / Naive Bayes). Built as a demonstration of the text classification work described in my CV.

## 📁 Project Structure
```
text-classification-project/
├── data/
│   └── sample_data.csv      # Example labeled dataset
├── src/
│   ├── train.py             # Trains and evaluates the model
│   └── predict.py           # Predicts the label of new text
├── requirements.txt
└── README.md
```

## ⚙️ How It Works
1. **Preprocessing:** Text is cleaned and converted into numeric features using TF-IDF vectorization.
2. **Training:** A Logistic Regression classifier is trained on the labeled dataset.
3. **Evaluation:** Accuracy, precision, recall, and F1-score are reported on a held-out test split.
4. **Prediction:** The trained model can classify new, unseen text samples.

## 🚀 Getting Started

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the model
python src/train.py

# 3. Predict on new text
python src/predict.py "your text here"
```

## 🛠️ Tools & Libraries
- Python
- scikit-learn
- pandas
- TF-IDF (feature extraction)

## 📈 Example Output
```
Accuracy: 0.91
              precision    recall  f1-score
negative          0.90      0.89      0.90
positive          0.92      0.93      0.92
```

## 📌 Notes
This is a lightweight demo version of larger-scale text classification work involving big datasets and deep learning models (Google Colab, TensorFlow). It's meant to showcase the workflow and code quality — swap `data/sample_data.csv` with a larger dataset to scale up.

## 👩‍💻 Author
**Huda Alhawiti** — Computer Engineer | AI & Machine Learning
[LinkedIn](https://linkedin.com/in/huda-h-alhawiti)
