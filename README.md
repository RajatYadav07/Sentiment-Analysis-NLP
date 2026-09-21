<div align="center">
  <h2>🎬 MovieReviewIQ — AI Movie Sentiment Intelligence</h2>
  <p>Production-grade NLP SaaS platform classifying movie review sentiment with real-time inference, feature attribution, and interactive model diagnostics.</p>
</div>

---

### 📖 Overview
**MovieReviewIQ** is a portfolio-grade, recruiter-ready Natural Language Processing (NLP) intelligence application. It performs binary and nuanced sentiment classification on film reviews using a mathematically grounded **TF-IDF + Logistic Regression** pipeline trained on 50,000 IMDb records.

The application features a cinematic **Amber/Gold + Deep Charcoal** SaaS interface built with Streamlit, complete with real-time inference, confidence breakdown, top contributing word attribution, comprehensive dataset analytics, and model performance benchmarking.

---

### 🧠 End-to-End NLP Architecture
The six-stage production pipeline operates as follows:

```
IMDb Dataset (50,000 Reviews)
       │
       ▼
Text Cleaning & Preprocessing (HTML strip, regex, stopword removal, WordNet lemmatization)
       │
       ▼
TF-IDF Vectorization (50,000 unigrams & bigrams, sublinear TF scaling)
       │
       ▼
Logistic Regression Classifier (C=5.0, saga solver, L2 penalty)
       │
       ▼
Real-Time Sentiment Prediction (Positive / Negative classification & confidence score)
       │
       ▼
Feature Attribution (Dynamic coefficient attribution identifying key driving words)
```

1. **Text Preprocessing**: Custom HTML/URL stripping, punctuation removal, and NLTK WordNet lemmatization with film-specific stopwords filtered.
2. **Feature Engineering**: Scikit-Learn `TfidfVectorizer` extracting 50,000 unigrams and bigrams with sublinear term-frequency scaling.
3. **Classification Engine**: Fine-tuned Logistic Regression (`C=5.0`, `saga` solver) benchmarked directly against Multinomial Naive Bayes.
4. **Interpretability & Attribution**: Real-time token-level feature attribution displaying the exact words and weights driving the model's decision.

---

### 📊 Verified Model Performance

Evaluated on a held-out, stratified test set of **10,000 reviews** (5,000 Positive / 5,000 Negative):

| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression (Primary)** | **90.4%** | **89.4%** | **91.7%** | **90.6%** |
| Multinomial Naive Bayes (Baseline) | 88.1% | 87.4% | 89.1% | 88.3% |

#### Detailed Confusion Matrix (Held-out Test Set)
- **True Positives (TP)**: 4,587
- **True Negatives (TN)**: 4,458
- **False Positives (FP)**: 542
- **False Negatives (FN)**: 413

<details>
<summary><b>View Classification Report (Logistic Regression)</b></summary>

```text
              precision    recall  f1-score   support

    Negative       0.91      0.89      0.90      5000
    Positive       0.89      0.92      0.91      5000

    accuracy                           0.90     10000
   macro avg       0.90      0.90      0.90     10000
weighted avg       0.90      0.90      0.90     10000
```
</details>

---

### 💻 Web Application Features

- **Sentiment Analyzer**:
  - Live review text area with real-time synchronized character counter (`0 / 5000`).
  - One-click sample selector and quick-test cards (Positive, Negative, Mixed).
  - Prominent Amber/Gold action button with responsive prediction engine.
  - Granular confidence percentage and positive/negative driver breakdown.
- **Model Performance Tab**:
  - Interactive KPI cards for Accuracy, Precision, Recall, and F1-Score.
  - Side-by-side comparison between Logistic Regression and Naive Bayes.
  - Confusion matrix heatmap breakdown.
- **Dataset Explorer Tab**:
  - Balanced 50/50 class distribution metrics (25,000 positive / 25,000 negative).
  - Visual 6-stage NLP pipeline architecture card.
- **SaaS Sidebar Navigation**:
  - Clean 260px fixed-width navigation rail with collapsible responsive support.
  - At-a-glance Model Info panel and test metrics summary.

---

### 🚀 Local Installation & Execution

1. **Clone the repository**
   ```bash
   git clone https://github.com/RajatYadav07/Sentiment-Analysis-NLP.git
   cd sentiment_analysis
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train Models (Optional if pre-trained models exist in `models/`)**
   ```bash
   python train.py
   ```

4. **Launch the Application**
   ```bash
   streamlit run app.py
   ```
   Open `http://localhost:8501` in your browser.

---

**Author**: Rajat Yadav  
**Tech Stack**: Python, Scikit-Learn, NLTK, Streamlit, Pandas, NumPy, Matplotlib.
