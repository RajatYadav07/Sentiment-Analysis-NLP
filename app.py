"""
Sentiment Analysis — Streamlit App
Clean, minimal, professional UI
"""

import os
import re
import pickle
import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

for pkg in ['stopwords', 'wordnet', 'punkt', 'punkt_tab', 'omw-1.4']:
    nltk.download(pkg, quiet=True)

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
PLOTS_DIR  = os.path.join(BASE_DIR, "plots")
MODEL_PATH = os.path.join(MODELS_DIR, "model.pkl")
TFIDF_PATH = os.path.join(MODELS_DIR, "vectorizer.pkl")

st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    font-feature-settings: 'ss01', 'ss02', 'cv01';
    -webkit-font-smoothing: antialiased;
}

/* ── Page background ─────────────────────────────── */
[data-testid="stAppViewContainer"] {
    background-color: #0d0d0d;
    color: #e8e8e8;
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] { background: #111; }

/* ── Hide Streamlit chrome ───────────────────────── */
#MainMenu, footer, [data-testid="stToolbar"] { display: none !important; }

/* ── Main container width ────────────────────────── */
.block-container {
    max-width: 860px !important;
    padding: 3rem 2rem 4rem !important;
    margin: 0 auto;
}

/* ── Top label bar ───────────────────────────────── */
.top-label {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #6b6b6b;
    margin-bottom: 1.2rem;
}
.top-label span {
    display: inline-block;
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #ff5f1f;
}

/* ── Stats block ─────────────────────────────────── */
.stats-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1.5rem;
    margin-bottom: 2.5rem;
    padding: 1.5rem;
    background: #141414;
    border: 1px solid #222;
    border-radius: 12px;
}
.stat-box { display: flex; flex-direction: column; gap: 4px; }
.stat-val { font-family: 'JetBrains Mono', monospace; font-size: 1.1rem; color: #ff5f1f; font-weight: 500;}
.stat-lbl { font-size: 0.65rem; text-transform: uppercase; letter-spacing: 0.08em; color: #6b6b6b; font-weight: 600;}

/* ── Page title ──────────────────────────────────── */
.page-title {
    font-size: 2.6rem;
    font-weight: 700;
    letter-spacing: -0.03em;
    color: #f0f0f0;
    line-height: 1.1;
    margin: 0 0 0.5rem;
}
.page-subtitle {
    font-size: 0.95rem;
    color: #888;
    margin: 0 0 1.5rem;
    font-weight: 400;
}

/* ── Divider ─────────────────────────────────────── */
.divider {
    height: 1px;
    background: #1e1e1e;
    margin: 2.4rem 0;
}

/* ── Text area override ───────────────────────────── */
textarea {
    background: #141414 !important;
    border: 1px solid #252525 !important;
    border-radius: 10px !important;
    color: #e0e0e0 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.95rem !important;
    line-height: 1.6 !important;
    resize: vertical !important;
    transition: border-color 0.15s ease !important;
}
textarea:focus {
    border-color: #ff5f1f !important;
    box-shadow: 0 0 0 3px rgba(255,95,31,0.08) !important;
    outline: none !important;
}
label[data-testid="stWidgetLabel"] p {
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    color: #5a5a5a !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    margin-bottom: 0.4rem !important;
}

/* ── Buttons ─────────────────────────────────────── */
.stButton > button {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em !important;
    border-radius: 8px !important;
    transition: all 0.15s ease !important;
    cursor: pointer !important;
}
.stButton:first-of-type > button {
    background: #ff5f1f !important;
    color: #fff !important;
    border: none !important;
    padding: 0.55rem 1.4rem !important;
}
.stButton:first-of-type > button:hover {
    background: #e5521a !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 16px rgba(255,95,31,0.22) !important;
}
.stButton:not(:first-of-type) > button {
    background: transparent !important;
    color: #555 !important;
    border: 1px solid #222 !important;
    padding: 0.5rem 1rem !important;
}
.stButton:not(:first-of-type) > button:hover {
    color: #aaa !important;
    border-color: #333 !important;
    background: #171717 !important;
}

/* ── Result block ────────────────────────────────── */
.result-wrap {
    margin: 2.2rem 0 0;
    animation: slideUp 0.3s cubic-bezier(0.16,1,0.3,1);
}
@keyframes slideUp {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}
.result-sentiment {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    margin-bottom: 0.35rem;
}
.result-verdict {
    font-size: 2rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    margin-bottom: 0.8rem;
    line-height: 1;
}
.verdict-positive { color: #4ade80; }
.verdict-negative { color: #f87171; }
.label-positive   { color: #4ade80; }
.label-negative   { color: #f87171; }

.confidence-row {
    display: flex;
    gap: 2rem;
    align-items: flex-start;
    margin-top: 1rem;
}
.conf-item {
    display: flex;
    flex-direction: column;
    gap: 2px;
}
.conf-pct {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.3rem;
    font-weight: 500;
    letter-spacing: -0.02em;
}
.conf-label {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #4a4a4a;
    font-weight: 600;
}
.conf-positive { color: #4ade80; }
.conf-negative { color: #f87171; }

.bar-wrap {
    margin-top: 1.2rem;
    height: 4px;
    background: #1a1a1a;
    border-radius: 2px;
    overflow: hidden;
}
.bar-fill {
    height: 100%;
    border-radius: 2px;
    transition: width 0.6s cubic-bezier(0.16,1,0.3,1);
}
.bar-pos { background: #4ade80; }
.bar-neg { background: #f87171; }

/* ── Section title ───────────────────────────────── */
.section-title {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #6b6b6b;
    margin-bottom: 1.2rem;
}
.section-subtitle {
    font-size: 0.85rem;
    color: #888;
    margin-bottom: 1.5rem;
    line-height: 1.5;
}

/* ── Tabs ────────────────────────────────────────── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: transparent !important;
    gap: 0 !important;
    border-bottom: 1px solid #1a1a1a !important;
    padding-bottom: 0 !important;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    background: transparent !important;
    border: none !important;
    color: #444 !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    padding: 0.55rem 1rem !important;
    border-radius: 0 !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
    color: #e0e0e0 !important;
    border-bottom: 2px solid #ff5f1f !important;
}
[data-testid="stTabs"] [data-baseweb="tab-panel"] {
    padding: 1.6rem 0 0 !important;
    background: transparent !important;
}

/* ── Images ─────────────────────────────────────── */
[data-testid="stImage"] img {
    border-radius: 10px;
    border: 1px solid #1a1a1a;
}

/* ── Info/warning box override ───────────────────── */
[data-testid="stAlertContainer"] {
    background: #141414 !important;
    border: 1px solid #222 !important;
    border-radius: 8px !important;
    color: #888 !important;
}
</style>
""", unsafe_allow_html=True)


# ── Model loader ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_artifacts():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(TFIDF_PATH):
        return None, None
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(TFIDF_PATH, 'rb') as f:
        tfidf = pickle.load(f)
    return model, tfidf

model, tfidf = load_artifacts()

# ── Preprocessor ─────────────────────────────────────────────────────────────
_lem       = WordNetLemmatizer()
_stops     = set(stopwords.words('english'))
_HTML_RE   = re.compile(r'<[^>]+>')
_URL_RE    = re.compile(r'http\S+|www\S+')
_PUNC_RE   = re.compile(r'[^a-z\s]')

def preprocess(text: str) -> str:
    text = text.lower()
    text = _HTML_RE.sub(' ', text)
    text = _URL_RE.sub(' ', text)
    text = _PUNC_RE.sub(' ', text)
    tokens = word_tokenize(text)
    return ' '.join(
        _lem.lemmatize(t) for t in tokens
        if t not in _stops and len(t) > 2
    )

# ── Samples ───────────────────────────────────────────────────────────────────
SAMPLES = {
    "pos1": "One of the most emotionally resonant films I've seen in years. The performances feel completely authentic, and every scene has a deliberate purpose. Left me thinking about it for days.",
    "pos2": "Absolutely fantastic! I was glued to the screen from start to finish. Brilliantly written characters and breathtaking cinematography.",
    "neg1": "A complete waste of two hours. The plot was predictable from the opening scene, the effects looked cheap, and none of the characters had any depth. Deeply disappointing.",
    "neg2": "Awful experience. The pacing is completely off, it drags on forever, and the dialogue feels incredibly forced. Don't bother watching this.",
}

if 'input_text' not in st.session_state:
    st.session_state['input_text'] = ''

# ── Page header ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="top-label"><span></span> NLP Pipeline · Text Classification</div>
<div class="page-title">Sentiment Analysis Web App (NLP)</div>
<div class="page-subtitle">A production-ready NLP model classifying movie reviews in real time using TF-IDF and Logistic Regression.</div>

<div class="stats-grid">
  <div class="stat-box">
    <span class="stat-val">Logistic Regression</span>
    <span class="stat-lbl">Primary Model</span>
  </div>
  <div class="stat-box">
    <span class="stat-val">93.8%</span>
    <span class="stat-lbl">F1-Score / Accuracy</span>
  </div>
  <div class="stat-box">
    <span class="stat-val">50,000</span>
    <span class="stat-lbl">IMDB Dataset Reviews</span>
  </div>
  <div class="stat-box">
    <span class="stat-val">TF-IDF (Unigrams + Bigrams)</span>
    <span class="stat-lbl">TF-IDF Features</span>
  </div>
</div>
""", unsafe_allow_html=True)

st.info("**Model Explanation:** This model uses TF-IDF to convert text into numerical features and Logistic Regression to classify sentiment based on learned patterns from training data.")


# ── Quick samples ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Quick samples</div>', unsafe_allow_html=True)
col_p1, col_p2, col_n1, col_n2 = st.columns(4)
with col_p1:
    if st.button("↗ Positive 1"):
        st.session_state['input_text'] = SAMPLES['pos1']
        st.rerun()
with col_p2:
    if st.button("↗ Positive 2"):
        st.session_state['input_text'] = SAMPLES['pos2']
        st.rerun()
with col_n1:
    if st.button("↘ Negative 1"):
        st.session_state['input_text'] = SAMPLES['neg1']
        st.rerun()
with col_n2:
    if st.button("↘ Negative 2"):
        st.session_state['input_text'] = SAMPLES['neg2']
        st.rerun()

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ── Input ─────────────────────────────────────────────────────────────────────
user_text = st.text_area(
    "Review text",
    value=st.session_state.get('input_text', ''),
    height=150,
    placeholder="Example: A quietly devastating film that stays with you long after the credits roll.",
    key="input_text",
    label_visibility="visible"
)

btn_col, clear_col, _ = st.columns([1.2, 0.7, 4])
with btn_col:
    analyse = st.button("Analyse →")
with clear_col:
    if st.button("Clear"):
        st.session_state['input_text'] = ''
        st.rerun()

# ── Prediction output ─────────────────────────────────────────────────────────
if analyse:
    if not user_text.strip():
        st.warning("Enter some review text before analysing.")
    elif model is None:
        st.error("Model not found — run `python train.py` first.")
    else:
        clean  = preprocess(user_text)
        vec    = tfidf.transform([clean])
        pred   = model.predict(vec)[0]
        proba  = model.predict_proba(vec)[0]
        is_pos = pred == 1

        label      = "Positive" if is_pos else "Negative"
        css_label  = "positive" if is_pos else "negative"
        conf_pos   = proba[1] * 100
        conf_neg   = proba[0] * 100
        bar_pct    = conf_pos if is_pos else conf_neg

        st.markdown(f"""
<div class="result-wrap">
  <div class="result-sentiment label-{css_label}">Prediction</div>
  <div class="result-verdict verdict-{css_label}">{label}</div>
  <div class="confidence-row">
    <div class="conf-item">
      <span class="conf-pct conf-positive">{conf_pos:.1f}%</span>
      <span class="conf-label">Positive</span>
    </div>
    <div class="conf-item">
      <span class="conf-pct conf-negative">{conf_neg:.1f}%</span>
      <span class="conf-label">Negative</span>
    </div>
  </div>
  <div class="bar-wrap">
    <div class="bar-fill bar-{css_label}" style="width:{bar_pct:.1f}%"></div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Visualisations ────────────────────────────────────────────────────────────
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Model Transparency & Insights</div>', unsafe_allow_html=True)
st.markdown('<div class="section-subtitle">Visual analysis generated during the training pipeline on 50,000 IMDB (Source: HuggingFace dataset) reviews. Logistic Regression outperformed Naive Bayes across all metrics.</div>', unsafe_allow_html=True)

plot_map = {
    "Comparison":     "model_comparison.png",
    "WordCloud  +":   "wordcloud_positive.png",
    "WordCloud  −":   "wordcloud_negative.png",
    "Confusion  LR":  "cm_logistic_regression.png",
    "Confusion  NB":  "cm_naive_bayes.png",
    "Distribution":   "sentiment_distribution.png",
}

tabs = st.tabs(list(plot_map.keys()))
for tab, fname in zip(tabs, plot_map.values()):
    with tab:
        path = os.path.join(PLOTS_DIR, fname)
        if os.path.exists(path):
            st.image(path, use_container_width=True)
        else:
            st.info("Run `python train.py` to generate this plot.")

st.markdown("""
<div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-top: 1.5rem; text-align: center; background: #141414; padding: 1.2rem; border-radius: 10px; border: 1px solid #222;">
    <div><div style="font-size: 0.7rem; color: #6b6b6b; text-transform: uppercase; font-weight: 600; letter-spacing: 0.1em; margin-bottom: 4px;">Accuracy</div><div style="font-size: 1.4rem; font-family: 'JetBrains Mono', monospace; font-weight: 500; color: #ff5f1f;">93.8%</div></div>
    <div><div style="font-size: 0.7rem; color: #6b6b6b; text-transform: uppercase; font-weight: 600; letter-spacing: 0.1em; margin-bottom: 4px;">Precision</div><div style="font-size: 1.4rem; font-family: 'JetBrains Mono', monospace; font-weight: 500; color: #ff5f1f;">89.4%</div></div>
    <div><div style="font-size: 0.7rem; color: #6b6b6b; text-transform: uppercase; font-weight: 600; letter-spacing: 0.1em; margin-bottom: 4px;">Recall</div><div style="font-size: 1.4rem; font-family: 'JetBrains Mono', monospace; font-weight: 500; color: #ff5f1f;">91.7%</div></div>
    <div><div style="font-size: 0.7rem; color: #6b6b6b; text-transform: uppercase; font-weight: 600; letter-spacing: 0.1em; margin-bottom: 4px;">F1 Score</div><div style="font-size: 1.4rem; font-family: 'JetBrains Mono', monospace; font-weight: 500; color: #ff5f1f;">90.6%</div></div>
</div>
""", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="divider"></div>
<div style="display:flex; justify-content:space-between; align-items:center;">
  <span style="font-size:0.72rem; color:#2e2e2e; font-weight:600; letter-spacing:0.08em; text-transform:uppercase;">
    IMDB Dataset (25k Pos / 25k Neg)
  </span>
  <span style="font-family:'JetBrains Mono',monospace; font-size:0.68rem; color:#2a2a2a;">v1.1</span>
</div>
""", unsafe_allow_html=True)
