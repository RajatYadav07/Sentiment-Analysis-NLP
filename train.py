"""
Sentiment Analysis Pipeline - Train Script
==========================================
CV-Level Project: Amazon/IMDB Product Review Sentiment Analysis
Pipeline: Data Collection → Preprocessing → TF-IDF → Training → Evaluation → Visualization
"""

import os
import re
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# NLP
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# ML
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)

# Dataset
from datasets import load_dataset

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
# 0. Configuration
# ─────────────────────────────────────────────────────────────
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
PLOTS_DIR  = os.path.join(OUTPUT_DIR, "plots")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,  exist_ok=True)

# ─────────────────────────────────────────────────────────────
# 1. Download NLTK assets
# ─────────────────────────────────────────────────────────────
print("📥 Downloading NLTK assets ...")
for pkg in ['stopwords', 'wordnet', 'punkt', 'punkt_tab', 'omw-1.4']:
    nltk.download(pkg, quiet=True)

# ─────────────────────────────────────────────────────────────
# 2. Data Collection (CSV Loading)
# ─────────────────────────────────────────────────────────────
csv_path = os.path.join(OUTPUT_DIR, "IMDB Dataset.csv")
if not os.path.exists(csv_path):
    print("\n📂 Downloading IMDB dataset to CSV (First run only) ...")
    dataset = load_dataset("imdb")
    _df = pd.concat([dataset['train'].to_pandas(), dataset['test'].to_pandas()], ignore_index=True)
    _df.rename(columns={'label': 'sentiment'}, inplace=True)
    _df.to_csv(csv_path, index=False)

print("\n📂 Loading IMDB dataset from CSV ...")
df = pd.read_csv(csv_path)

# Handle missing values
initial_len = len(df)
df.dropna(inplace=True)
if len(df) < initial_len:
    print(f"   Removed {initial_len - len(df)} rows with missing values.")

print(f"   Dataset Size : {len(df):,} reviews")
print(f"   Class Dist   : Positive: {df['sentiment'].sum():,} | Negative: {(df['sentiment']==0).sum():,}")

# ─────────────────────────────────────────────────────────────
# 3. Preprocessing
# ─────────────────────────────────────────────────────────────
print("\n🔧 Preprocessing text ...")
lemmatizer  = WordNetLemmatizer()
stop_words  = set(stopwords.words('english'))
custom_stops = {'movie', 'film', 'one', 'character', 'time', 'story', 'scene', 'people', 'watch', 'make', 'even', 'really'}
stop_words.update(custom_stops)

HTML_TAG_RE = re.compile(r'<[^>]+>')
URL_RE      = re.compile(r'http\S+|www\S+')
PUNC_RE     = re.compile(r'[^a-z\s]')

def preprocess(text: str) -> str:
    text = text.lower()
    text = HTML_TAG_RE.sub(' ', text)     # remove HTML tags
    text = URL_RE.sub(' ', text)          # remove URLs
    text = PUNC_RE.sub(' ', text)         # remove punctuation / digits
    tokens = word_tokenize(text)
    tokens = [
        lemmatizer.lemmatize(t)
        for t in tokens
        if t not in stop_words and len(t) > 2
    ]
    return ' '.join(tokens)

df['clean_text'] = df['text'].apply(preprocess)
print("   Preprocessing complete ✔")

# ─────────────────────────────────────────────────────────────
# 4. Visualization – Sentiment Distribution
# ─────────────────────────────────────────────────────────────
print("\n🎨 Generating Sentiment Distribution plot ...")

# Sentiment Distribution
fig, ax = plt.subplots(figsize=(7, 4))
counts = df['sentiment'].value_counts().rename({0: 'Negative', 1: 'Positive'})
colors = ['#e74c3c', '#2ecc71']
bars = ax.bar(counts.index, counts.values, color=colors, edgecolor='white', linewidth=0.8, width=0.5)
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 300, f'{val:,}',
            ha='center', va='bottom', fontsize=12, fontweight='bold')
ax.set_title('Sentiment Distribution', fontsize=15, fontweight='bold')
ax.set_ylabel('Number of Reviews')
ax.set_xlabel('Sentiment')
ax.spines[['top', 'right']].set_visible(False)
fig.tight_layout()
dist_path = os.path.join(PLOTS_DIR, 'sentiment_distribution.png')
fig.savefig(dist_path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"   Saved: {dist_path}")

# ─────────────────────────────────────────────────────────────
# 5. Feature Engineering – TF-IDF
# ─────────────────────────────────────────────────────────────
print("\n⚙️  Vectorizing with TF-IDF ...")
X = df['clean_text']
y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

tfidf = TfidfVectorizer(max_features=50_000, ngram_range=(1, 2), sublinear_tf=True)
X_train_vec = tfidf.fit_transform(X_train)
X_test_vec  = tfidf.transform(X_test)
print(f"   Vocabulary size: {len(tfidf.vocabulary_):,}")

# Save TF-IDF vectorizer
tfidf_path = os.path.join(MODELS_DIR, 'tfidf_vectorizer.pkl')
with open(tfidf_path, 'wb') as f:
    pickle.dump(tfidf, f)
print(f"   Saved TF-IDF vectorizer → {tfidf_path}")

# ─────────────────────────────────────────────────────────────
# 6. Model Training & Evaluation
# ─────────────────────────────────────────────────────────────
print("\n🤖 Training models ...")

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, C=5.0, solver='saga', n_jobs=-1),
    'Naive Bayes':         MultinomialNB(alpha=0.1)
}

results = {}
best_model_name = None
best_f1 = -1

for name, model in models.items():
    print(f"\n   ▶ Training {name} ...")
    model.fit(X_train_vec, y_train)
    preds = model.predict(X_test_vec)

    acc  = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec  = recall_score(y_test, preds)
    f1   = f1_score(y_test, preds)
    cm   = confusion_matrix(y_test, preds)

    results[name] = {'model': model, 'preds': preds, 'cm': cm,
                     'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}

    print(f"   Accuracy : {acc:.4f}")
    print(f"   Precision: {prec:.4f}")
    print(f"   Recall   : {rec:.4f}")
    print(f"   F1-Score : {f1:.4f}")
    print(f"\n{classification_report(y_test, preds, target_names=['Negative','Positive'])}")

    if f1 > best_f1:
        best_f1 = f1
        best_model_name = name

# ─────────────────────────────────────────────────────────────
# 7. Confusion Matrix Heatmaps
# ─────────────────────────────────────────────────────────────
print("\n📊 Saving confusion matrix heatmaps ...")
for name, res in results.items():
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        res['cm'], annot=True, fmt='d', cmap='Blues',
        xticklabels=['Negative', 'Positive'],
        yticklabels=['Negative', 'Positive'],
        linewidths=0.5, linecolor='white', ax=ax
    )
    ax.set_title(f'Confusion Matrix – {name}', fontsize=13, fontweight='bold')
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    fig.tight_layout()
    safe_name = name.replace(' ', '_').lower()
    path = os.path.join(PLOTS_DIR, f'cm_{safe_name}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"   Saved: {path}")

# ─────────────────────────────────────────────────────────────
# 8. Summary Bar Chart – Model Comparison
# ─────────────────────────────────────────────────────────────
metrics = ['accuracy', 'precision', 'recall', 'f1']
x = np.arange(len(metrics))
width = 0.35
fig, ax = plt.subplots(figsize=(9, 5))
palette = ['#3498db', '#e67e22']
for i, (name, res) in enumerate(results.items()):
    vals = [res[m] for m in metrics]
    bars = ax.bar(x + i*width - width/2, vals, width, label=name, color=palette[i], edgecolor='white')
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f'{v:.3f}', ha='center', va='bottom', fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels([m.capitalize() for m in metrics])
ax.set_ylim(0.80, 1.02)
ax.set_ylabel('Score')
ax.set_title('Model Comparison – Evaluation Metrics', fontsize=14, fontweight='bold')
ax.legend()
ax.spines[['top', 'right']].set_visible(False)
fig.tight_layout()
cmp_path = os.path.join(PLOTS_DIR, 'model_comparison.png')
fig.savefig(cmp_path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"   Saved: {cmp_path}")

# ─────────────────────────────────────────────────────────────
# 9. Model-driven WordClouds (TF-IDF Coefficients)
# ─────────────────────────────────────────────────────────────
print("\n🎨 Generating feature-driven WordClouds from ML Model ...")

# Extract coefficients from the Logistic Regression model
feature_names = tfidf.get_feature_names_out()
lr_model = results['Logistic Regression']['model']
coefs = lr_model.coef_[0]

# Positive features (highest coefficients)
pos_indices = {feature_names[i]: coefs[i] for i in range(len(feature_names)) if coefs[i] > 0}
top_pos = dict(sorted(pos_indices.items(), key=lambda item: item[1], reverse=True)[:100])

# Negative features (lowest coefficients)
neg_indices = {feature_names[i]: -coefs[i] for i in range(len(feature_names)) if coefs[i] < 0}
top_neg = dict(sorted(neg_indices.items(), key=lambda item: item[1], reverse=True)[:100])

def generate_coef_wordcloud(freq_dict, bg_color, cmap, title, filename):
    wc = WordCloud(
        width=900, height=450, background_color=bg_color,
        colormap=cmap, max_words=100, collocations=False
    ).generate_from_frequencies(freq_dict)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    # Make title color match background contrast
    title_color = 'black' if bg_color == 'white' else 'white'
    ax.set_title(title, fontsize=16, fontweight='bold', pad=14, color=title_color)
    fig.patch.set_facecolor(bg_color)
    
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"   Saved: {path}")

generate_coef_wordcloud(top_pos, 'white', 'Greens', 'Top Positive Sentiment Drivers (LR Coefficients)', 'wordcloud_positive.png')
generate_coef_wordcloud(top_neg, '#0a0a0a', 'Reds', 'Top Negative Sentiment Drivers (LR Coefficients)', 'wordcloud_negative.png')

# ─────────────────────────────────────────────────────────────
# 10. Save Best Model
# ─────────────────────────────────────────────────────────────
best_model = results[best_model_name]['model']

# Save with canonical deliverable names (model.pkl / vectorizer.pkl)
for fname in ['model.pkl', 'best_model.pkl']:
    with open(os.path.join(MODELS_DIR, fname), 'wb') as f:
        pickle.dump(best_model, f)

import shutil
shutil.copy(tfidf_path, os.path.join(MODELS_DIR, 'vectorizer.pkl'))

print(f"\n🏆 Best Model : {best_model_name}  (F1 = {best_f1:.4f})")
print(f"   Saved → models/model.pkl  |  models/vectorizer.pkl")
print("\n✅ Training complete! Run `streamlit run app.py` to launch the app.")
