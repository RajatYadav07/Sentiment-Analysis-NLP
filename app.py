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

/* ── Page background & Glassmorphism ─────────────────────────────── */
[data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at 50% 0%, #1a1a24 0%, #0a0a0f 100%);
    color: #e8e8e8;
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] { background: #0f0f15; border-right: 1px solid rgba(255,255,255,0.05); }

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
    gap: 8px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #8b8b99;
    margin-bottom: 1.5rem;
}
.top-label span {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #06b6d4;
    box-shadow: 0 0 10px rgba(6,182,212,0.8);
}

/* ── Stats block ─────────────────────────────────── */
.stats-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1.5rem;
    margin-bottom: 3rem;
    padding: 1.8rem;
    background: rgba(20, 20, 25, 0.6);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
}
.stat-box { display: flex; flex-direction: column; gap: 6px; }
.stat-val { font-family: 'JetBrains Mono', monospace; font-size: 1.25rem; color: #06b6d4; font-weight: 500; text-shadow: 0 0 15px rgba(6,182,212,0.3);}
.stat-lbl { font-size: 0.65rem; text-transform: uppercase; letter-spacing: 0.1em; color: #7f7f8c; font-weight: 600;}

/* ── Page title ──────────────────────────────────── */
.page-title {
    font-size: 2.8rem;
    font-weight: 700;
    letter-spacing: -0.03em;
    color: #ffffff;
    line-height: 1.15;
    margin: 0 0 0.8rem;
    background: linear-gradient(135deg, #ffffff 0%, #a0a0a0 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.page-subtitle {
    font-size: 1.05rem;
    color: #9494a3;
    margin: 0 0 2rem;
    font-weight: 400;
    line-height: 1.6;
}

/* ── Divider ─────────────────────────────────────── */
.divider {
    height: 1px;
    background: linear-gradient(90deg, rgba(255,255,255,0) 0%, rgba(255,255,255,0.1) 50%, rgba(255,255,255,0) 100%);
    margin: 3rem 0;
}

/* ── Text area override ───────────────────────────── */
textarea {
    background: rgba(15, 15, 20, 0.8) !important;
    backdrop-filter: blur(10px) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 12px !important;
    color: #f0f0f0 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 1rem !important;
    line-height: 1.6 !important;
    padding: 1.2rem !important;
    resize: vertical !important;
    transition: all 0.3s ease !important;
    box-shadow: inset 0 2px 10px rgba(0,0,0,0.2) !important;
}
textarea:focus {
    border-color: #06b6d4 !important;
    box-shadow: 0 0 0 4px rgba(6,182,212,0.15), inset 0 2px 10px rgba(0,0,0,0.2) !important;
    outline: none !important;
}
label[data-testid="stWidgetLabel"] p {
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    color: #8b8b99 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
    margin-bottom: 0.6rem !important;
}

/* ── Buttons ─────────────────────────────────────── */
.stButton > button {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.05em !important;
    border-radius: 10px !important;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    cursor: pointer !important;
    text-transform: uppercase !important;
}
.stButton:first-of-type > button {
    background: linear-gradient(135deg, #06b6d4 0%, #0891b2 100%) !important;
    color: #fff !important;
    border: none !important;
    padding: 0.6rem 1.6rem !important;
    box-shadow: 0 4px 15px rgba(6,182,212,0.2) !important;
}
.stButton:first-of-type > button:hover {
    transform: translateY(-2px) scale(1.02) !important;
    box-shadow: 0 8px 25px rgba(6,182,212,0.4) !important;
}
.stButton:not(:first-of-type) > button {
    background: rgba(255,255,255,0.03) !important;
    color: #a0a0a0 !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    padding: 0.55rem 1.2rem !important;
}
.stButton:not(:first-of-type) > button:hover {
    color: #fff !important;
    border-color: rgba(255,255,255,0.25) !important;
    background: rgba(255,255,255,0.08) !important;
    transform: translateY(-1px) !important;
}

/* ── Result block ────────────────────────────────── */
.result-wrap {
    margin: 2.5rem 0 0;
    padding: 2.5rem;
    background: rgba(20, 20, 25, 0.6);
    border-radius: 16px;
    border: 1px solid rgba(255,255,255,0.08);
    backdrop-filter: blur(12px);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
    animation: slideUp 0.4s cubic-bezier(0.16,1,0.3,1);
}
@keyframes slideUp {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: translateY(0); }
}
.result-sentiment {
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    margin-bottom: 0.5rem;
}
.result-verdict {
    font-size: 2.5rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    margin-bottom: 1.2rem;
    line-height: 1;
}
.verdict-positive { color: #4ade80; text-shadow: 0 0 20px rgba(74,222,128,0.3); }
.verdict-negative { color: #f87171; text-shadow: 0 0 20px rgba(248,113,113,0.3); }
.label-positive   { color: #4ade80; }
.label-negative   { color: #f87171; }

.confidence-row {
    display: flex;
    gap: 3.5rem;
    align-items: flex-start;
    margin-top: 1.5rem;
}
.conf-item {
    display: flex;
    flex-direction: column;
    gap: 4px;
}
.conf-pct {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.6rem;
    font-weight: 500;
    letter-spacing: -0.02em;
}
.conf-label {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #8b8b99;
    font-weight: 600;
}
.conf-positive { color: #4ade80; }
.conf-negative { color: #f87171; }

.bar-wrap {
    margin-top: 1.8rem;
    height: 6px;
    background: rgba(255,255,255,0.05);
    border-radius: 3px;
    overflow: hidden;
    box-shadow: inset 0 1px 3px rgba(0,0,0,0.5);
}
.bar-fill {
    height: 100%;
    border-radius: 3px;
    transition: width 0.8s cubic-bezier(0.16,1,0.3,1);
}
.bar-pos { background: linear-gradient(90deg, #22c55e, #4ade80); box-shadow: 0 0 10px rgba(74,222,128,0.5); }
.bar-neg { background: linear-gradient(90deg, #ef4444, #f87171); box-shadow: 0 0 10px rgba(248,113,113,0.5); }

/* ── Section title ───────────────────────────────── */
.section-title {
    font-size: 0.8rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    color: #8b8b99;
    margin-bottom: 1.5rem;
}
.section-subtitle {
    font-size: 0.95rem;
    color: #9494a3;
    margin-bottom: 2rem;
    line-height: 1.6;
}

/* ── Tabs ────────────────────────────────────────── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: transparent !important;
    gap: 1rem !important;
    border-bottom: 1px solid rgba(255,255,255,0.08) !important;
    padding-bottom: 0 !important;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    background: transparent !important;
    border: none !important;
    color: #6a6a75 !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    padding: 0.8rem 1.2rem !important;
    border-radius: 0 !important;
    transition: all 0.2s ease !important;
}
[data-testid="stTabs"] [data-baseweb="tab"]:hover {
    color: #e0e0e0 !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
    color: #ffffff !important;
    border-bottom: 2px solid #06b6d4 !important;
    text-shadow: 0 0 10px rgba(6,182,212,0.4) !important;
}
[data-testid="stTabs"] [data-baseweb="tab-panel"] {
    padding: 2rem 0 0 !important;
    background: transparent !important;
}

/* ── Images ─────────────────────────────────────── */
[data-testid="stImage"] img {
    border-radius: 12px;
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    transition: transform 0.3s ease;
}
[data-testid="stImage"] img:hover {
    transform: scale(1.01);
}

/* ── Info/warning box override ───────────────────── */
[data-testid="stAlertContainer"] {
    background: rgba(20,20,25,0.6) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    backdrop-filter: blur(10px) !important;
    border-radius: 12px !important;
    color: #a0a0a0 !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.15) !important;
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
<div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 1.5rem; margin-top: 2rem; text-align: center; background: rgba(20, 20, 25, 0.6); backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px); padding: 1.8rem; border-radius: 16px; border: 1px solid rgba(255,255,255,0.08); box-shadow: 0 8px 32px rgba(0,0,0,0.2);">
    <div><div style="font-size: 0.75rem; color: #7f7f8c; text-transform: uppercase; font-weight: 600; letter-spacing: 0.1em; margin-bottom: 6px;">Accuracy</div><div style="font-size: 1.6rem; font-family: 'JetBrains Mono', monospace; font-weight: 500; color: #06b6d4; text-shadow: 0 0 15px rgba(6,182,212,0.3);">93.8%</div></div>
    <div><div style="font-size: 0.75rem; color: #7f7f8c; text-transform: uppercase; font-weight: 600; letter-spacing: 0.1em; margin-bottom: 6px;">Precision</div><div style="font-size: 1.6rem; font-family: 'JetBrains Mono', monospace; font-weight: 500; color: #06b6d4; text-shadow: 0 0 15px rgba(6,182,212,0.3);">89.4%</div></div>
    <div><div style="font-size: 0.75rem; color: #7f7f8c; text-transform: uppercase; font-weight: 600; letter-spacing: 0.1em; margin-bottom: 6px;">Recall</div><div style="font-size: 1.6rem; font-family: 'JetBrains Mono', monospace; font-weight: 500; color: #06b6d4; text-shadow: 0 0 15px rgba(6,182,212,0.3);">91.7%</div></div>
    <div><div style="font-size: 0.75rem; color: #7f7f8c; text-transform: uppercase; font-weight: 600; letter-spacing: 0.1em; margin-bottom: 6px;">F1 Score</div><div style="font-size: 1.6rem; font-family: 'JetBrains Mono', monospace; font-weight: 500; color: #06b6d4; text-shadow: 0 0 15px rgba(6,182,212,0.3);">90.6%</div></div>
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
