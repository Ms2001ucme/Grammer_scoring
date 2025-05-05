
# 📚 Grammar Scoring using Transcribed Audio

This project predicts grammar quality scores (on a 0–5 scale) from transcribed audio recordings. It uses natural language processing (TF-IDF) and machine learning (SVM Regression) to build a scoring engine. The dataset includes transcripts and corresponding grammar labels.

---

## 📂 Dataset

- `transcripts.csv`: Training data containing columns:
  - `filename`: name of the audio file
  - `transcript`: transcribed text
  - `label`: ground truth grammar score
- `transcript_test.csv`: Test data containing:
  - `filename`: name of the test audio file
  - `transcript`: transcribed text (used for prediction)

---

## 🧠 Model Pipeline

1. **Text Preprocessing**  
   - Remove null entries.
   - Convert transcripts to string format.

2. **Feature Extraction**  
   - Apply TF-IDF vectorization (max 1000 features).

3. **Model Training**  
   - Train a **Support Vector Regressor (SVR)** with a linear kernel.
   - Evaluate with RMSE and R² Score.

4. **Inference**  
   - Apply the trained model on test set transcripts.
   - Save predictions as `submission.csv`.

---

## 📈 Results (Example)

| Metric       | Value     |
|--------------|-----------|
| RMSE         | ~1.02     |
| R² Score     | ~0.21     |

*Note: These scores may vary based on data quality and preprocessing.*

---

## 📝 Submission

The final output is saved as:
```
submission.csv
├── filename
└── label (Predicted Grammar Score)
```

---

## 🚀 How to Run

1. Upload dataset files to Kaggle or your local environment.
2. Run the notebook cells in order.
3. The submission file will be generated at the end.

---

## 🔧 Requirements

- Python 3.x
- pandas
- scikit-learn
- numpy
- (Optional for local dev) faster-whisper / nltk if working with audio transcription or grammar tools

---

## 📌 Notes

- Transcription was done using the `faster-whisper` library separately (not included in the notebook due to Kaggle restrictions).
- Grammar scores are learned only from transcribed text using regression techniques.
