"""
MovieReviewIQ — AI-Powered Movie Sentiment Intelligence
Portfolio-Grade NLP SaaS Application powered by TF-IDF & Logistic Regression.
Refined UI directly aligned with the MovieReviewIQ Dark Navy Product Mockup.
"""

import os
import re
import pickle
import numpy as np
import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# ─────────────────────────────────────────────────────────────
# 1. Page Configuration & NLTK Setup
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MovieReviewIQ — AI Movie Sentiment Intelligence",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource(show_spinner=False)
def download_nltk_assets():
    for pkg in ['stopwords', 'wordnet', 'punkt', 'punkt_tab', 'omw-1.4']:
        nltk.download(pkg, quiet=True)

download_nltk_assets()

# ─────────────────────────────────────────────────────────────
# 2. Paths & Authoritative Metrics (Single Source of Truth)
# ─────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
PLOTS_DIR  = os.path.join(BASE_DIR, "plots")
MODEL_PATH = os.path.join(MODELS_DIR, "model.pkl")
TFIDF_PATH = os.path.join(MODELS_DIR, "vectorizer.pkl")

# Exact evaluated metrics from train.py on the 10,000 held-out test reviews (5k pos / 5k neg)
METRICS_EVAL = {
    "logistic_regression": {
        "name": "Logistic Regression",
        "accuracy": 90.4,
        "precision": 89.4,
        "recall": 91.7,
        "f1": 90.6,
        "tn": 4458,
        "fp": 542,
        "fn": 413,
        "tp": 4587
    },
    "naive_bayes": {
        "name": "Multinomial Naive Bayes",
        "accuracy": 88.1,
        "precision": 87.4,
        "recall": 89.1,
        "f1": 88.3,
        "tn": 4358,
        "fp": 642,
        "fn": 543,
        "tp": 4457
    }
}

# ─────────────────────────────────────────────────────────────
# 3. Design System & CSS (Navy Dark Theme matching Mockup)
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    -webkit-font-smoothing: antialiased;
}

/* Background & Surfaces (Amber/Gold & Deep Charcoal Design System) */
[data-testid="stAppViewContainer"] {
    background: #0D0D0C;
    color: #F5F5F0;
}
[data-testid="stHeader"] {
    background: transparent;
}
/* Hide default streamlit chrome */
#MainMenu, footer, [data-testid="stToolbar"] { display: none !important; }

/* Main Container: responsive SaaS workspace that automatically expands when sidebar collapses */
.block-container {
    width: 100% !important;
    max-width: min(1580px, 96vw) !important;
    padding: 1.5rem 2.2rem 3rem !important;
    margin: 0 auto !important;
    transition: max-width 0.25s ease, padding 0.25s ease !important;
}

/* ── Top App Bar (Header with Search & User Profile) ── */
.top-app-bar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding-bottom: 1.25rem;
    margin-bottom: 1.25rem;
    border-bottom: 1px solid #292820;
}
.app-brand-inline {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.78rem;
    color: #F59E0B;
    font-weight: 500;
}
.top-actions {
    display: flex;
    align-items: center;
    gap: 16px;
}
.mock-search {
    display: flex;
    align-items: center;
    gap: 8px;
    background: #171714;
    border: 1px solid #292820;
    border-radius: 8px;
    padding: 6px 14px;
    font-size: 0.8rem;
    color: #706F67;
    min-width: 240px;
}
.kbd-shortcut {
    background: #1B1A16;
    border: 1px solid #292820;
    border-radius: 4px;
    padding: 1px 6px;
    font-size: 0.68rem;
    color: #A3A196;
    font-family: 'JetBrains Mono', monospace;
    margin-left: auto;
}
.user-avatar {
    width: 32px;
    height: 32px;
    border-radius: 50%;
    background: #1B1A16;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.75rem;
    font-weight: 700;
    color: #FBBF24;
    border: 1px solid #F59E0B;
}

/* ── Hero Title Section ── */
.hero-header-row {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 1.5rem;
}
.main-heading {
    font-size: 2.1rem;
    font-weight: 800;
    color: #F5F5F0;
    letter-spacing: -0.03em;
    margin: 0 0 4px 0;
    line-height: 1.15;
}
.main-heading span {
    color: #FBBF24;
}
.main-subheading {
    font-size: 0.92rem;
    color: #A3A196;
    margin: 0;
}
.hero-quote {
    text-align: right;
    font-size: 0.82rem;
    color: #706F67;
    font-style: italic;
    line-height: 1.4;
}

/* ── Sidebar Styles (SaaS Navigation Rail) ── */
[data-testid="stSidebar"] {
    background-color: #11110F !important;
    border-right: 1px solid #292820 !important;
    min-width: 260px !important;
    max-width: 260px !important;
    width: 260px !important;
    height: 100vh !important;
    transition: min-width 0.25s cubic-bezier(0.4, 0, 0.2, 1),
                max-width 0.25s cubic-bezier(0.4, 0, 0.2, 1),
                transform 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
}

/* When sidebar is collapsed by user */
[data-testid="stSidebar"][aria-expanded="false"] {
    min-width: 0px !important;
    max-width: 0px !important;
    width: 0px !important;
    border-right: none !important;
    transform: translateX(-100%) !important;
}

[data-testid="stSidebar"] > div:first-child {
    padding: 1.5rem 0.9rem 1.25rem !important;
    background-color: #11110F !important;
    display: flex !important;
    flex-direction: column !important;
    gap: 0 !important;
    height: 100vh !important;
    overflow-x: hidden !important;
    overflow-y: auto !important;
    scrollbar-width: thin;
    scrollbar-color: transparent transparent;
}

/* Subtle scrollbar ONLY appears when content overflows and user hovers/scrolls */
[data-testid="stSidebar"] > div:first-child:hover {
    scrollbar-color: #292820 transparent;
}
[data-testid="stSidebar"] > div:first-child::-webkit-scrollbar {
    width: 4px;
}
[data-testid="stSidebar"] > div:first-child::-webkit-scrollbar-track {
    background: transparent;
}
[data-testid="stSidebar"] > div:first-child::-webkit-scrollbar-thumb {
    background: transparent;
    border-radius: 4px;
    transition: background 0.15s ease;
}
[data-testid="stSidebar"] > div:first-child:hover::-webkit-scrollbar-thumb {
    background: #292820;
}
[data-testid="stSidebar"] > div:first-child::-webkit-scrollbar-thumb:hover {
    background: #38372D;
}

/* Streamlit Collapsed Control Button (Arrow Rail) Styling */
[data-testid="stSidebarCollapsedControl"] {
    background: #11110F !important;
    border: 1px solid #292820 !important;
    border-radius: 8px !important;
    top: 14px !important;
    left: 14px !important;
    color: #F59E0B !important;
    transition: all 0.2s ease !important;
    z-index: 999999 !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.35) !important;
}
[data-testid="stSidebarCollapsedControl"]:hover {
    background: #1B1A16 !important;
    border-color: #F59E0B !important;
    color: #FBBF24 !important;
}
[data-testid="stSidebarCollapsedControl"] svg {
    stroke: #F59E0B !important;
    fill: #F59E0B !important;
}

/* 1. Brand Header */
.sb-brand-block {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 0 0.25rem 1.1rem;
    border-bottom: 1px solid #292820;
    margin-bottom: 1rem;
    flex-shrink: 0;
}
.sb-brand-icon {
    width: 32px;
    height: 32px;
    background: #F59E0B;
    border-radius: 6px;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
}
.sb-brand-title {
    font-size: 1rem; /* 16px */
    font-weight: 600;
    letter-spacing: -0.01em;
    color: #F5F5F0;
    line-height: 1.2;
    white-space: nowrap;
}
.sb-brand-sub {
    font-size: 0.68rem; /* 10.5px */
    color: #A3A196;
    margin-top: 2px;
    line-height: 1.2;
    white-space: nowrap;
}

/* 2. Navigation System (One Line, 38-40px height, No Emojis) */
[data-testid="stSidebar"] [data-testid="stRadio"] {
    margin-bottom: 1rem !important;
    flex-shrink: 0 !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] > label,
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(span:contains("Navigation")),
[data-testid="stSidebar"] [data-testid="stRadio"] > div:first-child:not([role="radiogroup"]) {
    display: none !important;
    height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
    visibility: hidden !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] > div {
    display: flex !important;
    flex-direction: column !important;
    gap: 3px !important;
    margin: 0 !important;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label {
    background: transparent !important;
    border: 1px solid transparent !important;
    border-radius: 7px !important;
    padding: 0 0.75rem !important;
    height: 38px !important;
    min-height: 38px !important;
    display: flex !important;
    align-items: center !important;
    cursor: pointer !important;
    transition: all 0.12s ease !important;
    margin: 0 !important;
    width: 100% !important;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    background: #1B1A16 !important;
}

/* Active Navigation Row */
[data-testid="stSidebar"] [data-testid="stRadio"] label[data-checked="true"],
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
    background: rgba(245, 158, 11, 0.10) !important;
    border: 1px solid rgba(245, 158, 11, 0.25) !important;
    border-left: 3px solid #F59E0B !important;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label span,
[data-testid="stSidebar"] [data-testid="stRadio"] label p {
    font-size: 0.85rem !important; /* 13.5px */
    font-weight: 500 !important;
    color: #A3A196 !important;
    margin: 0 !important;
    letter-spacing: -0.01em !important;
    display: flex !important;
    align-items: center !important;
    gap: 9px !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label[data-checked="true"] span,
[data-testid="stSidebar"] [data-testid="stRadio"] label[data-checked="true"] p,
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) span,
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) p {
    color: #F5F5F0 !important;
    font-weight: 600 !important;
}

/* Hide default radio circle */
[data-testid="stSidebar"] [data-testid="stRadio"] input[type="radio"] {
    display: none !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label > div:first-child {
    display: none !important;
}

/* Professional SVG Icons for Nav Items (Activity, BarChart, Database) */
[data-testid="stSidebar"] [data-testid="stRadio"] label span::before,
[data-testid="stSidebar"] [data-testid="stRadio"] label p::before {
    content: "" !important;
    display: inline-block !important;
    width: 15px !important;
    height: 15px !important;
    flex-shrink: 0 !important;
    background-size: contain !important;
    background-repeat: no-repeat !important;
    background-position: center !important;
    opacity: 0.70 !important;
    transition: opacity 0.12s ease !important;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label[data-checked="true"] span::before,
[data-testid="stSidebar"] [data-testid="stRadio"] label[data-checked="true"] p::before,
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) span::before,
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) p::before {
    opacity: 1 !important;
}

/* 1. Sentiment Analyzer: Activity / Spark Icon */
[data-testid="stSidebar"] [data-testid="stRadio"] label:nth-child(1) span::before,
[data-testid="stSidebar"] [data-testid="stRadio"] label:nth-child(1) p::before {
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23F59E0B' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpolyline points='22 12 18 12 15 21 9 3 6 12 2 12'/%3E%3C/svg%3E") !important;
}

/* 2. Model Performance: Bar Chart Icon */
[data-testid="stSidebar"] [data-testid="stRadio"] label:nth-child(2) span::before,
[data-testid="stSidebar"] [data-testid="stRadio"] label:nth-child(2) p::before {
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23F59E0B' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cline x1='18' y1='20' x2='18' y2='10'/%3E%3Cline x1='12' y1='20' x2='12' y2='4'/%3E%3Cline x1='6' y1='20' x2='6' y2='14'/%3E%3C/svg%3E") !important;
}

/* 3. Dataset Explorer: Database Icon */
[data-testid="stSidebar"] [data-testid="stRadio"] label:nth-child(3) span::before,
[data-testid="stSidebar"] [data-testid="stRadio"] label:nth-child(3) p::before {
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23F59E0B' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cellipse cx='12' cy='5' rx='9' ry='3'/%3E%3Cpath d='M21 12c0 1.66-4 3-9 3s-9-1.34-9-3'/%3E%3Cpath d='M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5'/%3E%3C/svg%3E") !important;
}

/* 3. Model Info Section */
.sb-divider {
    height: 1px;
    background: #292820;
    margin-bottom: 0.85rem;
    flex-shrink: 0;
}

.sb-label {
    font-size: 0.62rem; /* 10px */
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #A3A196;
    margin: 0 0 0.45rem 0.2rem;
    flex-shrink: 0;
}

.sb-info-card {
    background: #171714;
    border: 1px solid #292820;
    border-radius: 7px;
    padding: 0.75rem 0.85rem;
    display: flex;
    flex-direction: column;
    gap: 7px;
    margin-bottom: 0.85rem;
    flex-shrink: 0;
}
.sb-field {
    display: flex;
    flex-direction: column;
    gap: 1px;
}
.sb-f-label {
    font-size: 0.6rem; /* 9.5px */
    color: #706F67;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-weight: 700;
}
.sb-f-val {
    font-size: 0.8rem; /* 12.5px */
    color: #F5F5F0;
    font-weight: 500;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.sb-f-val.highlight {
    color: #FBBF24;
    font-size: 0.95rem;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700;
}

/* 4. Footer Section */
.sb-footer-links {
    display: flex;
    flex-direction: column;
    gap: 2px;
    padding-top: 0.75rem;
    border-top: 1px solid #292820;
    margin-top: auto;
    flex-shrink: 0;
}
.sb-footer-row {
    display: flex;
    align-items: center;
    gap: 9px;
    font-size: 0.82rem;
    color: #A3A196;
    font-weight: 500;
    padding: 6px 8px;
    border-radius: 6px;
    height: 32px;
    transition: all 0.12s ease;
    cursor: default;
}
.sb-footer-row:hover {
    background: #1B1A16;
    color: #F5F5F0;
}

.sb-version {
    font-size: 0.68rem;
    color: #706F67;
    font-family: 'JetBrains Mono', monospace;
    padding: 0.5rem 0.25rem 0;
}

/* ── Card Systems in Main Area ── */
.m-card {
    background: #171714;
    border: 1px solid #292820;
    border-radius: 12px;
    padding: 1.3rem 1.4rem;
    margin-bottom: 1.25rem;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35);
}

.m-card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.85rem;
}
.m-card-title {
    font-size: 0.88rem;
    font-weight: 700;
    color: #F5F5F0;
    display: flex;
    align-items: center;
    gap: 8px;
}
.m-char-counter {
    font-size: 0.74rem;
    color: #706F67;
    font-family: 'JetBrains Mono', monospace;
}

/* ── Custom Textarea ── */
textarea {
    background: #0D0D0C !important;
    border: 1px solid #292820 !important;
    border-radius: 8px !important;
    color: #F5F5F0 !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 0.92rem !important;
    line-height: 1.6 !important;
    padding: 0.95rem 1rem !important;
    min-height: 140px !important;
}
textarea::placeholder {
    color: #706F67 !important;
}
textarea:focus {
    border-color: #F59E0B !important;
    box-shadow: 0 0 0 2px rgba(245, 158, 11, 0.15) !important;
}

/* ── "Try a sample review..." Dropdown (Dark Charcoal + Amber Focus) ── */
div[data-testid="stSelectbox"] > div:first-child {
    border-radius: 8px !important;
}
div[data-testid="stSelectbox"] [data-baseweb="select"] {
    background-color: transparent !important;
    border-radius: 8px !important;
}
div[data-testid="stSelectbox"] [data-baseweb="select"] > div {
    background-color: #171714 !important;
    border: 1px solid #292820 !important;
    border-radius: 8px !important;
    color: #F5F5F0 !important;
    min-height: 40px !important;
    transition: all 0.15s ease !important;
    box-shadow: none !important;
}
div[data-testid="stSelectbox"] [data-baseweb="select"] > div:hover {
    background-color: #201E18 !important;
    border-color: #38372D !important;
}
div[data-testid="stSelectbox"] [data-baseweb="select"]:has(:focus) > div,
div[data-testid="stSelectbox"] [data-baseweb="select"] > div[aria-expanded="true"] {
    background-color: #171714 !important;
    border-color: #F59E0B !important;
    box-shadow: 0 0 0 2px rgba(245, 158, 11, 0.20) !important;
}
/* Selectbox text and placeholder */
div[data-testid="stSelectbox"] [data-baseweb="select"] [data-testid="stMarkdownContainer"] p,
div[data-testid="stSelectbox"] [data-baseweb="select"] span {
    color: #F5F5F0 !important;
    font-size: 0.86rem !important;
    font-weight: 500 !important;
}
div[data-testid="stSelectbox"] [data-baseweb="select"] [aria-placeholder] {
    color: #A3A196 !important;
}
/* Dropdown Arrow / SVG Icon */
div[data-testid="stSelectbox"] [data-baseweb="select"] svg {
    fill: #A3A196 !important;
    color: #A3A196 !important;
    transition: transform 0.15s ease, fill 0.15s ease !important;
}
div[data-testid="stSelectbox"] [data-baseweb="select"]:hover svg {
    fill: #F59E0B !important;
    color: #F59E0B !important;
}

/* Dropdown Popup Menu List */
ul[data-testid="main-menu-list"],
div[data-baseweb="popover"],
div[data-baseweb="menu"],
ul[role="listbox"] {
    background-color: #171714 !important;
    border: 1px solid #292820 !important;
    border-radius: 8px !important;
    box-shadow: 0 10px 25px rgba(0, 0, 0, 0.5) !important;
}
li[role="option"] {
    background-color: transparent !important;
    color: #F5F5F0 !important;
    font-size: 0.85rem !important;
    padding: 0.55rem 0.9rem !important;
    transition: background-color 0.12s ease !important;
}
li[role="option"]:hover,
li[role="option"][aria-selected="true"] {
    background-color: #201E18 !important;
    color: #F5F5F0 !important;
}
li[role="option"][aria-selected="true"] {
    border-left: 2px solid #F59E0B !important;
}

/* ── Action Buttons ── */
/* Target Streamlit primary buttons across all versions & testid patterns */
button[data-testid="baseButton-primary"],
button[data-testid="stBaseButton-primary"],
div[data-testid="stButton"] button[kind="primary"],
div[data-testid="stButton"] > button:first-child[kind="primary"],
div[data-testid="stButton"] > button.primary {
    background: #F59E0B !important;
    background-color: #F59E0B !important;
    color: #0D0D0C !important;
    border: 1px solid #F59E0B !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    font-size: 0.88rem !important;
    padding: 0.55rem 1.4rem !important;
    box-shadow: none !important;
    transition: all 0.15s ease !important;
}
button[data-testid="baseButton-primary"]:hover,
button[data-testid="stBaseButton-primary"]:hover,
div[data-testid="stButton"] button[kind="primary"]:hover,
div[data-testid="stButton"] > button:first-child[kind="primary"]:hover,
div[data-testid="stButton"] > button.primary:hover {
    background: #FBBF24 !important;
    background-color: #FBBF24 !important;
    border-color: #FBBF24 !important;
    color: #0D0D0C !important;
}
button[data-testid="baseButton-primary"]:active,
button[data-testid="stBaseButton-primary"]:active,
div[data-testid="stButton"] button[kind="primary"]:active,
div[data-testid="stButton"] > button:first-child[kind="primary"]:active,
div[data-testid="stButton"] > button.primary:active {
    background: #D97706 !important;
    background-color: #D97706 !important;
    border-color: #D97706 !important;
    color: #0D0D0C !important;
}
button[data-testid="baseButton-primary"]:focus,
button[data-testid="stBaseButton-primary"]:focus,
div[data-testid="stButton"] button[kind="primary"]:focus,
div[data-testid="stButton"] > button:first-child[kind="primary"]:focus,
div[data-testid="stButton"] > button.primary:focus {
    outline: none !important;
    box-shadow: 0 0 0 2px rgba(245, 158, 11, 0.4) !important;
    border-color: #F59E0B !important;
}

/* Fallback catch-all for any primary styled button inside review input */
div[data-testid="stButton"] > button[data-testid*="primary"] {
    background: #F59E0B !important;
    background-color: #F59E0B !important;
    color: #0D0D0C !important;
    border: 1px solid #F59E0B !important;
}
div[data-testid="stButton"] > button[data-testid*="primary"]:hover {
    background: #FBBF24 !important;
    background-color: #FBBF24 !important;
    border-color: #FBBF24 !important;
    color: #0D0D0C !important;
}
div[data-testid="stButton"] > button[data-testid*="primary"]:active {
    background: #D97706 !important;
    background-color: #D97706 !important;
    border-color: #D97706 !important;
    color: #0D0D0C !important;
}

button[data-testid="baseButton-secondary"],
button[data-testid="stBaseButton-secondary"],
div[data-testid="stButton"] button[kind="secondary"],
div[data-testid="stButton"] > button:first-child[kind="secondary"] {
    background: #1B1A16 !important;
    background-color: #1B1A16 !important;
    border: 1px solid #292820 !important;
    color: #A3A196 !important;
    border-radius: 7px !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    transition: all 0.15s ease !important;
}
button[data-testid="baseButton-secondary"]:hover,
button[data-testid="stBaseButton-secondary"]:hover,
div[data-testid="stButton"] button[kind="secondary"]:hover,
div[data-testid="stButton"] > button:first-child[kind="secondary"]:hover {
    background: #292820 !important;
    background-color: #292820 !important;
    color: #F5F5F0 !important;
    border-color: #38372D !important;
}

/* ── Sample Review Cards (3-Column Layout) ── */
.sample-box {
    background: #171714;
    border: 1px solid #292820;
    border-radius: 10px;
    padding: 1.1rem;
    height: 100%;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    transition: border-color 0.15s ease;
}
.sample-box.pos { border: 1px solid rgba(34, 197, 94, 0.35); }
.sample-box.neg { border: 1px solid rgba(244, 63, 94, 0.35); }
.sample-box.mix { border: 1px solid rgba(245, 158, 11, 0.35); }

.sample-head-tag {
    font-size: 0.74rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    gap: 6px;
}
.sample-head-tag.pos { color: #22C55E; }
.sample-head-tag.neg { color: #F43F5E; }
.sample-head-tag.mix { color: #F59E0B; }

.sample-body-quote {
    font-size: 0.82rem;
    color: #A3A196;
    line-height: 1.45;
    margin-bottom: 0.85rem;
}

/* ── Bottom Metric Stat Cards (4-Column) ── */
.metric-row-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-top: 1.25rem;
}
.metric-stat-card {
    background: #171714;
    border: 1px solid #292820;
    border-radius: 10px;
    padding: 1rem 1.1rem;
    display: flex;
    align-items: center;
    gap: 12px;
}
.metric-icon-circle {
    width: 38px;
    height: 38px;
    border-radius: 50%;
    background: #1B1A16;
    border: 1px solid #292820;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1rem;
    flex-shrink: 0;
}
.metric-stat-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.25rem;
    font-weight: 700;
    color: #FBBF24;
    line-height: 1.2;
}
.metric-stat-title {
    font-size: 0.72rem;
    color: #A3A196;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.metric-stat-sub {
    font-size: 0.68rem;
    color: #706F67;
}

/* ── Right Column: Prediction Result & Breakdown ── */
.panel-model-badge {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    background: #1B1A16;
    border: 1px solid #292820;
    padding: 2px 8px;
    border-radius: 4px;
    color: #F59E0B;
}

/* Empty State / Active Result Container */
.pred-display-box {
    text-align: center;
    padding: 1.5rem 0.5rem;
}
.pred-icon-circle {
    width: 52px;
    height: 52px;
    border-radius: 50%;
    background: #1B1A16;
    border: 1px solid #292820;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 1.3rem;
    color: #F59E0B;
    margin-bottom: 0.85rem;
}
.pred-ready-title {
    font-size: 0.95rem;
    font-weight: 700;
    color: #F5F5F0;
    margin-bottom: 0.35rem;
}
.pred-ready-sub {
    font-size: 0.78rem;
    color: #706F67;
    max-width: 280px;
    margin: 0 auto;
    line-height: 1.45;
}

.pred-three-stats {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    border-top: 1px solid #292820;
    padding-top: 1rem;
    margin-top: 1.2rem;
    text-align: center;
}
.p-stat-title {
    font-size: 0.72rem;
    font-weight: 600;
    color: #A3A196;
    margin-bottom: 2px;
}
.p-stat-sub {
    font-size: 0.68rem;
    color: #706F67;
}

/* Active Prediction Output */
.active-verdict {
    font-size: 2.2rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    line-height: 1;
    margin-bottom: 0.35rem;
}
.active-verdict.pos { color: #22C55E; }
.active-verdict.neg { color: #F43F5E; }

.active-prob {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.25rem;
    font-weight: 700;
    color: #F5F5F0;
}

/* Progress Breakdown Bars */
.conf-row {
    display: flex;
    align-items: center;
    gap: 12px;
    font-size: 0.78rem;
    margin-bottom: 10px;
}
.conf-label {
    width: 65px;
    color: #A3A196;
    font-weight: 500;
}
.conf-track {
    flex-grow: 1;
    height: 7px;
    background: #1B1A16;
    border: 1px solid #292820;
    border-radius: 4px;
    overflow: hidden;
}
.conf-fill-pos {
    height: 100%;
    background: #22C55E;
    border-radius: 4px;
}
.conf-fill-neg {
    height: 100%;
    background: #F43F5E;
    border-radius: 4px;
}
.conf-pct {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.76rem;
    width: 48px;
    text-align: right;
    color: #A3A196;
}

/* Contributing Word Signals (Filter Pills & Signal List) */
.sig-filter-row {
    display: flex;
    gap: 8px;
    margin-bottom: 10px;
}
.sig-btn-pos {
    background: rgba(34, 197, 94, 0.12);
    border: 1px solid rgba(34, 197, 94, 0.3);
    color: #22C55E;
    padding: 3px 10px;
    border-radius: 6px;
    font-size: 0.72rem;
    font-weight: 600;
}
.sig-btn-neg {
    background: rgba(244, 63, 94, 0.12);
    border: 1px solid rgba(244, 63, 94, 0.3);
    color: #F43F5E;
    padding: 3px 10px;
    border-radius: 6px;
    font-size: 0.72rem;
    font-weight: 600;
}

.word-contrib-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 4px 0;
    border-bottom: 1px solid #292820;
    font-size: 0.78rem;
}
.word-contrib-name {
    font-family: 'JetBrains Mono', monospace;
    color: #F5F5F0;
    width: 110px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.word-contrib-bar-track {
    flex-grow: 1;
    height: 5px;
    background: #1B1A16;
    border-radius: 3px;
    margin: 0 10px;
    overflow: hidden;
}
.word-contrib-fill-p {
    height: 100%;
    background: #22C55E;
    border-radius: 3px;
}
.word-contrib-fill-n {
    height: 100%;
    background: #F43F5E;
    border-radius: 3px;
}
.word-contrib-score {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.74rem;
    width: 44px;
    text-align: right;
}
.word-contrib-score.pos { color: #22C55E; }
.word-contrib-score.neg { color: #F43F5E; }

/* ── Bottom Quote & Metadata Banner (Footer) ── */
.bottom-banner {
    background: #171714;
    border: 1px solid #292820;
    border-radius: 8px;
    padding: 1rem 1.4rem;
    margin-top: 2rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.bottom-quote {
    font-size: 0.82rem;
    color: #A3A196;
    font-style: italic;
    display: flex;
    align-items: center;
    gap: 8px;
}
.bottom-tags {
    font-size: 0.72rem;
    color: #706F67;
    font-family: 'JetBrains Mono', monospace;
    display: flex;
    gap: 14px;
}

/* ── Tables & Matrix (in Performance / Explorer views) ── */
.data-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.85rem;
    margin: 0.75rem 0;
}
.data-table th {
    text-align: left;
    padding: 8px 12px;
    background: #1B1A16;
    color: #A3A196;
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    border-bottom: 1px solid #292820;
}
.data-table td {
    padding: 10px 12px;
    border-bottom: 1px solid #292820;
    color: #F5F5F0;
}
.data-table tr:hover td {
    background: #1B1A16;
}
.cell-lead {
    color: #FBBF24;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
}

.matrix-grid {
    display: grid;
    grid-template-columns: 105px 1fr 1fr;
    gap: 8px;
    font-size: 0.8rem;
    margin-top: 0.5rem;
}
.matrix-cell {
    background: #171714;
    border: 1px solid #292820;
    border-radius: 6px;
    padding: 10px;
    text-align: center;
}
.matrix-cell.primary-hit {
    background: rgba(34, 197, 94, 0.08);
    border-color: rgba(34, 197, 94, 0.35);
}
.matrix-cell.error-hit {
    background: rgba(244, 63, 94, 0.08);
    border-color: rgba(244, 63, 94, 0.35);
}
.matrix-num {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.15rem;
    font-weight: 700;
    color: #F5F5F0;
}
.matrix-sub {
    font-size: 0.68rem;
    color: #A3A196;
    margin-top: 2px;
}
.matrix-hdr {
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #A3A196;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# 4. Model Loading & Preprocessing
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model_artifacts():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(TFIDF_PATH):
        return None, None
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        with open(TFIDF_PATH, 'rb') as f:
            tfidf = pickle.load(f)
        return model, tfidf
    except Exception:
        return None, None

model, tfidf = load_model_artifacts()

_lem = WordNetLemmatizer()
_stops = set(stopwords.words('english'))
_custom_stops = {'movie', 'film', 'one', 'character', 'time', 'story', 'scene', 'people', 'watch', 'make', 'even', 'really'}
_stops.update(_custom_stops)

_HTML_RE = re.compile(r'<[^>]+>')
_URL_RE  = re.compile(r'http\S+|www\S+')
_PUNC_RE = re.compile(r'[^a-z\s]')

def preprocess_text(text: str) -> str:
    if not text or not isinstance(text, str):
        return ""
    text = text.lower()
    text = _HTML_RE.sub(' ', text)
    text = _URL_RE.sub(' ', text)
    text = _PUNC_RE.sub(' ', text)
    tokens = word_tokenize(text)
    cleaned = [
        _lem.lemmatize(t) for t in tokens
        if t not in _stops and len(t) > 2
    ]
    return ' '.join(cleaned)

def calculate_feature_attribution(text: str, tfidf_vec, lr_model):
    clean = preprocess_text(text)
    if not clean.strip():
        return [], []
    
    vec_x = tfidf_vec.transform([clean])
    feature_names = tfidf_vec.get_feature_names_out()
    coef = lr_model.coef_[0]
    
    row = vec_x.tocoo()
    signals = []
    for col, val in zip(row.col, row.data):
        feat = feature_names[col]
        c = coef[col]
        score = val * c
        signals.append((feat, score, val, c))
    
    pos_signals = [s for s in signals if s[1] > 0]
    neg_signals = [s for s in signals if s[1] < 0]
    
    pos_signals.sort(key=lambda x: x[1], reverse=True)
    neg_signals.sort(key=lambda x: x[1])
    return pos_signals, neg_signals

@st.cache_data(show_spinner=False)
def get_global_top_features():
    if model is None or tfidf is None:
        return [], []
    feature_names = np.array(tfidf.get_feature_names_out())
    coef = model.coef_[0]
    top_p = np.argsort(coef)[-8:][::-1]
    top_n = np.argsort(coef)[:8]
    pos_f = [(feature_names[i], coef[i]) for i in top_p]
    neg_f = [(feature_names[i], coef[i]) for i in top_n]
    return pos_f, neg_f

# ─────────────────────────────────────────────────────────────
# 5. Sidebar (Mockup Model Info, Navigation, and Footer Quote)
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sb-brand-block">
        <div class="sb-brand-icon">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#FFFFFF" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                <rect x="2" y="2" width="20" height="20" rx="4" />
                <path d="m9 12 2 2 4-4" />
            </svg>
        </div>
        <div>
            <div class="sb-brand-title">MovieReviewIQ</div>
            <div class="sb-brand-sub">AI Movie Sentiment Intelligence</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 2. Navigation (No emojis, single line, identical dimensions)
    active_view = st.radio(
        label="",
        options=["Sentiment Analyzer", "Model Performance", "Dataset Explorer"],
        index=0,
        label_visibility="collapsed"
    )

    # 3. Model Info Card
    st.markdown('<div class="sb-label">MODEL INFO</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sb-info-card">
        <div class="sb-field">
            <span class="sb-f-label">MODEL</span>
            <span class="sb-f-val">Logistic Regression</span>
        </div>
        <div class="sb-field">
            <span class="sb-f-label">VECTORIZER</span>
            <span class="sb-f-val">TF-IDF (50k n-grams)</span>
        </div>
        <div class="sb-field">
            <span class="sb-f-label">DATASET</span>
            <span class="sb-f-val">IMDb (50,000 reviews)</span>
        </div>
        <div class="sb-field">
            <span class="sb-f-label">EVALUATION SET</span>
            <span class="sb-f-val">10,000 reviews</span>
        </div>
        <div class="sb-field">
            <span class="sb-f-label">TEST ACCURACY</span>
            <span class="sb-f-val highlight">90.4%</span>
        </div>
    </div>

    <!-- 4. Footer -->
    <div class="sb-footer-links">
        <div class="sb-footer-row">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>
            About
        </div>
        <div class="sb-footer-row">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z"/><circle cx="12" cy="12" r="3"/></svg>
            Settings
        </div>
        <div class="sb-version">MovieReviewIQ v1.0</div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# 6. View: Sentiment Analyzer (Direct 2-Column Mockup Alignment)
# ─────────────────────────────────────────────────────────────
if active_view == "Sentiment Analyzer":
    # Mockup Top App Bar
    st.markdown("""
    <div class="top-app-bar">
        <div class="app-brand-inline">
            <svg width="10" height="10" viewBox="0 0 24 24" fill="#F59E0B" style="vertical-align:middle; margin-right:4px;"><circle cx="12" cy="12" r="10"/></svg>
            Real-time Inference Engine
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Hero Title with Right Quote (as in Mockup)
    st.markdown("""
    <div class="hero-header-row">
        <div>
            <h1 class="main-heading">Movie Review <span>Intelligence</span></h1>
            <p class="main-subheading">Analyze audience sentiment from movie reviews using TF-IDF-powered NLP classification.</p>
        </div>
        <div class="hero-quote">
            "Movies tell stories.<br>Our model listens."
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Sample reviews data
    SAMPLE_REVIEWS_MOCKUP = [
        {
            "tag": "pos",
            "title": "Positive Review",
            "text": "An absolutely wonderful film with brilliant acting and an amazing story. One of the best movies I've seen in years!"
        },
        {
            "tag": "neg",
            "title": "Negative Review",
            "text": "An incredibly boring and disappointing movie. The story was predictable and the acting felt forced. A complete waste of time."
        },
        {
            "tag": "mix",
            "title": "Mixed / Nuanced Review",
            "text": "The movie had some great visuals and a few interesting scenes, but the overall story felt weak and the pacing was inconsistent."
        }
    ]

    if 'input_review_text' not in st.session_state:
        st.session_state['input_review_text'] = ""

    def apply_sample(idx: int):
        st.session_state['input_review_text'] = SAMPLE_REVIEWS_MOCKUP[idx]["text"]

    def clear_review_input():
        st.session_state['input_review_text'] = ""

    # Two Main Columns matching Mockup Grid (Left: 65% Workspace, Right: 35% Results Panel)
    col_left, col_right = st.columns([62, 38], gap="medium")

    with col_left:
        # Card 1: Review Input Workspace
        st.markdown("""
        <div class="m-card" style="padding-bottom:0.75rem; margin-bottom:1rem;">
            <div class="m-card-header">
                <div class="m-card-title">
                    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#F59E0B" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle; margin-right:4px;"><path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/><path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/></svg>
                    Enter a Movie Review
                </div>
            </div>
        """, unsafe_allow_html=True)

        user_review = st.text_area(
            label="Review Text",
            placeholder="Paste or type a movie review here...",
            height=150,
            key="input_review_text",
            max_chars=5000,
            label_visibility="collapsed"
        )

        def on_sample_select():
            selected = st.session_state.get('sample_quick_pick', '')
            if selected == "Sample 1 (Positive)":
                st.session_state['input_review_text'] = SAMPLE_REVIEWS_MOCKUP[0]["text"]
            elif selected == "Sample 2 (Negative)":
                st.session_state['input_review_text'] = SAMPLE_REVIEWS_MOCKUP[1]["text"]
            elif selected == "Sample 3 (Mixed)":
                st.session_state['input_review_text'] = SAMPLE_REVIEWS_MOCKUP[2]["text"]

        col_sample_select, col_btn_analyze = st.columns([1.6, 1.2])
        with col_sample_select:
            st.selectbox(
                label="Try a sample review",
                options=["Try a sample review...", "Sample 1 (Positive)", "Sample 2 (Negative)", "Sample 3 (Mixed)"],
                index=0,
                key="sample_quick_pick",
                on_change=on_sample_select,
                label_visibility="collapsed"
            )

        with col_btn_analyze:
            analyze_trigger = st.button("Analyze Review →", type="primary", use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # Card 2: Try a Sample Review (3 Cards with Borders from Mockup)
        st.markdown("""
        <div style="margin: 1.25rem 0 0.65rem;">
            <div style="font-size:0.92rem; font-weight:700; color:#f1f5f9; display:flex; align-items:center; gap:8px;">
                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#F59E0B" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/></svg>
                Try a Sample Review
            </div>
            <div style="font-size:0.75rem; color:#64748b; margin-top:2px;">
                Click on any example to load it into the analyzer
            </div>
        </div>
        """, unsafe_allow_html=True)

        col_s1, col_s2, col_s3 = st.columns(3)
        for i, (c_col, s_data) in enumerate(zip([col_s1, col_s2, col_s3], SAMPLE_REVIEWS_MOCKUP)):
            with c_col:
                tag_col = "#22C55E" if s_data['tag'] == 'pos' else ("#F43F5E" if s_data['tag'] == 'neg' else "#F59E0B")
                st.markdown(f"""
                <div class="sample-box {s_data['tag']}">
                    <div>
                        <div class="sample-head-tag {s_data['tag']}">
                            <svg width="8" height="8" viewBox="0 0 24 24" fill="{tag_col}"><circle cx="12" cy="12" r="10"/></svg>
                            {s_data['title']}
                        </div>
                        <div class="sample-body-quote">
                            "{s_data['text']}"
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.button("Use This Review →", key=f"btn_card_s_{i}", on_click=apply_sample, args=(i,), use_container_width=True)

    with col_right:
        # Prediction Output Evaluation
        review_input_val = user_review.strip()
        has_run = analyze_trigger and bool(review_input_val)

        if has_run and model is not None and tfidf is not None:
            with st.spinner("Analyzing sentiment..."):
                cleaned = preprocess_text(review_input_val)
                vec = tfidf.transform([cleaned])
                pred_label = model.predict(vec)[0]
                
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(vec)[0]
                    p_neg = float(probs[0]) * 100
                    p_pos = float(probs[1]) * 100
                    confidence_pct = p_pos if pred_label == 1 else p_neg
                else:
                    p_neg = 0.0
                    p_pos = 100.0 if pred_label == 1 else 0.0
                    confidence_pct = 100.0

                is_positive = (pred_label == 1)
                verdict_str = "POSITIVE" if is_positive else "NEGATIVE"
                verdict_color = "pos" if is_positive else "neg"

                pos_words, neg_words = calculate_feature_attribution(review_input_val, tfidf, model)

            # Card 1: Prediction Result (Active State)
            st.markdown(f"""
            <div class="m-card">
                <div class="m-card-header">
                    <div class="m-card-title">
                        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#F59E0B" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2v4"/><path d="m4.93 10.93 2.83 2.83"/><path d="M2 18h4"/><path d="M20 18h2"/><path d="m19.07 10.93-2.83 2.83"/><path d="M22 22H2"/><path d="m8 6 4-4 4 4"/><path d="M16 18a4 4 0 0 0-8 0"/></svg>
                        Prediction Result
                    </div>
                    <span class="panel-model-badge">Logistic Regression</span>
                </div>
                <div class="pred-display-box">
                    <div class="active-verdict {verdict_color}">{verdict_str}</div>
                    <div class="active-prob">{confidence_pct:.1f}%</div>
                    <div style="font-size:0.75rem; color:#706F67; margin-top:2px;">Model Probability</div>
                </div>
                <div class="pred-three-stats">
                    <div>
                        <div class="p-stat-title">Sentiment</div>
                        <div class="p-stat-sub" style="color:{'#22C55E' if is_positive else '#F43F5E'}; font-weight:600;">{verdict_str}</div>
                    </div>
                    <div>
                        <div class="p-stat-title">Probability</div>
                        <div class="p-stat-sub">{confidence_pct:.1f}%</div>
                    </div>
                    <div>
                        <div class="p-stat-title">Key Insights</div>
                        <div class="p-stat-sub">{len(pos_words) + len(neg_words)} words</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Card 2: Model Confidence Breakdown (Active State)
            st.markdown(f"""
            <div class="m-card">
                <div class="m-card-header">
                    <div class="m-card-title">
                        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#F59E0B" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg>
                        Model Confidence Breakdown
                    </div>
                </div>
                <div class="conf-row">
                    <span class="conf-label">Positive</span>
                    <div class="conf-track"><div class="conf-fill-pos" style="width:{p_pos:.1f}%;"></div></div>
                    <span class="conf-pct">{p_pos:.1f}%</span>
                </div>
                <div class="conf-row" style="margin-bottom:0;">
                    <span class="conf-label">Negative</span>
                    <div class="conf-track"><div class="conf-fill-neg" style="width:{p_neg:.1f}%;"></div></div>
                    <span class="conf-pct">{p_neg:.1f}%</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Card 3: Top Contributing Words (Active State)
            st.markdown("""
            <div class="m-card">
                <div class="m-card-header">
                    <div class="m-card-title">
                        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#F59E0B" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
                        Top Contributing Words
                    </div>
                </div>
                <div class="sig-filter-row">
                    <span class="sig-btn-pos">+ Positive Signals</span>
                    <span class="sig-btn-neg">− Negative Signals</span>
                </div>
            """, unsafe_allow_html=True)

            # Render top active words
            all_signals = pos_words[:4] + neg_words[:4]
            if all_signals:
                max_sc = max([abs(s[1]) for s in all_signals]) if all_signals else 1.0
                for feat, score, val, c in all_signals:
                    pct = max(10, min(100, int((abs(score) / max_sc) * 100)))
                    is_p = score > 0
                    c_fill = "word-contrib-fill-p" if is_p else "word-contrib-fill-n"
                    c_val = "pos" if is_p else "neg"
                    sign = "+" if is_p else ""
                    st.markdown(f"""
                    <div class="word-contrib-row">
                        <div class="word-contrib-name">{sign}{feat}</div>
                        <div class="word-contrib-bar-track"><div class="{c_fill}" style="width:{pct}%;"></div></div>
                        <div class="word-contrib-score {c_val}">{sign}{score:.2f}</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("<div style='font-size:0.75rem; color:#706F67; padding:8px 0;'>No prominent signal words identified.</div>", unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

        else:
            # Card 1: Prediction Result (Empty State as in Mockup)
            st.markdown("""
            <div class="m-card">
                <div class="m-card-header">
                    <div class="m-card-title">
                        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#F59E0B" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2v4"/><path d="m4.93 10.93 2.83 2.83"/><path d="M2 18h4"/><path d="M20 18h2"/><path d="m19.07 10.93-2.83 2.83"/><path d="M22 22H2"/><path d="m8 6 4-4 4 4"/><path d="M16 18a4 4 0 0 0-8 0"/></svg>
                        Prediction Result
                    </div>
                    <span class="panel-model-badge">Logistic Regression</span>
                </div>
                <div class="pred-display-box">
                    <div class="pred-icon-circle">
                        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#F59E0B" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
                    </div>
                    <div class="pred-ready-title">Analyze a review to see the sentiment</div>
                    <div class="pred-ready-sub">Your prediction, probability and key insights will appear here.</div>
                </div>
                <div class="pred-three-stats">
                    <div>
                        <div class="p-stat-title">Sentiment</div>
                        <div class="p-stat-sub">Positive / Negative</div>
                    </div>
                    <div>
                        <div class="p-stat-title">Probability</div>
                        <div class="p-stat-sub">Model Confidence</div>
                    </div>
                    <div>
                        <div class="p-stat-title">Key Insights</div>
                        <div class="p-stat-sub">Important words</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Card 2: Model Confidence Breakdown (Empty State)
            st.markdown("""
            <div class="m-card">
                <div class="m-card-header">
                    <div class="m-card-title">
                        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#F59E0B" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg>
                        Model Confidence Breakdown
                    </div>
                </div>
                <div class="conf-row">
                    <span class="conf-label">Positive</span>
                    <div class="conf-track"><div class="conf-fill-pos" style="width:0%;"></div></div>
                    <span class="conf-pct">--%</span>
                </div>
                <div class="conf-row" style="margin-bottom:0;">
                    <span class="conf-label">Negative</span>
                    <div class="conf-track"><div class="conf-fill-neg" style="width:0%;"></div></div>
                    <span class="conf-pct">--%</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Card 3: Top Contributing Words (Empty State)
            st.markdown("""
            <div class="m-card">
                <div class="m-card-header">
                    <div class="m-card-title">
                        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#F59E0B" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
                        Top Contributing Words
                    </div>
                </div>
                <div class="sig-filter-row">
                    <span class="sig-btn-pos">+ Positive Signals</span>
                    <span class="sig-btn-neg">− Negative Signals</span>
                </div>
                <div style="text-align:center; padding:1.5rem 0.5rem; color:#706F67; font-size:0.78rem;">
                    <div style="margin-bottom:8px; display:inline-flex; align-items:center; justify-content:center;">
                        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#706F67" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg>
                    </div>
                    <div>Analyze a review to see top contributing words</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Bottom Row: 4 Metric Cards directly matching the reference Mockup
    st.markdown("""
    <div class="metric-row-grid">
        <div class="metric-stat-card">
            <div class="metric-icon-circle">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#F59E0B" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="3" width="20" height="14" rx="2"/><line x1="8" y1="21" x2="16" y2="21"/><line x1="12" y1="17" x2="12" y2="21"/></svg>
            </div>
            <div>
                <div class="metric-stat-val">50,000</div>
                <div class="metric-stat-title">Total Reviews</div>
                <div class="metric-stat-sub">IMDb Dataset</div>
            </div>
        </div>
        <div class="metric-stat-card">
            <div class="metric-icon-circle">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#22C55E" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3zM7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3"/></svg>
            </div>
            <div>
                <div class="metric-stat-val">25,000</div>
                <div class="metric-stat-title">Positive Reviews</div>
                <div class="metric-stat-sub">50.0% of total</div>
            </div>
        </div>
        <div class="metric-stat-card">
            <div class="metric-icon-circle">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#F43F5E" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10 15v4a3 3 0 0 0 3 3l4-9V2H5.72a2 2 0 0 0-2 1.7l-1.38 9a2 2 0 0 0 2 2.3zm7-13h3a2 2 0 0 1 2 2v7a2 2 0 0 1-2 2h-3"/></svg>
            </div>
            <div>
                <div class="metric-stat-val">25,000</div>
                <div class="metric-stat-title">Negative Reviews</div>
                <div class="metric-stat-sub">50.0% of total</div>
            </div>
        </div>
        <div class="metric-stat-card">
            <div class="metric-icon-circle">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#FBBF24" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg>
            </div>
            <div>
                <div class="metric-stat-val">10,000</div>
                <div class="metric-stat-title">Test Set Reviews</div>
                <div class="metric-stat-sub">20% held-out</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Bottom Quote Banner (Mockup Footer)
    st.markdown("""
    <div class="bottom-banner">
        <div class="bottom-quote">
            <span style="color:#F59E0B; font-size:1.2rem; font-family:serif;">“</span>
            Great movies stay with you. Now, you can measure why.
        </div>
        <div class="bottom-tags">
            <span>NLP</span>
            <span>●</span>
            <span>Machine Learning</span>
            <span>●</span>
            <span>Real Insights</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# 7. View: Model Performance (Navy Theme Alignment)
# ─────────────────────────────────────────────────────────────
elif active_view == "Model Performance":
    st.markdown("""
    <div class="top-app-bar">
        <div class="app-brand-inline">
            <span>●</span> Evaluated Test Set Benchmark
        </div>
    </div>
    <div class="hero-header-row">
        <div>
            <h1 class="main-heading">Model <span>Performance</span></h1>
            <p class="main-subheading">Rigorous evaluation on the 10,000 held-out IMDb test reviews (5,000 Positive / 5,000 Negative).</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    lr = METRICS_EVAL["logistic_regression"]
    nb = METRICS_EVAL["naive_bayes"]

    # 4 KPI Cards
    st.markdown(f"""
    <div class="metric-row-grid" style="margin-top:0; margin-bottom:1.5rem;">
        <div class="metric-stat-card">
            <div class="metric-icon-circle">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#F59E0B" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/></svg>
            </div>
            <div>
                <div class="metric-stat-val">{lr['accuracy']:.1f}%</div>
                <div class="metric-stat-title">Accuracy</div>
                <div class="metric-stat-sub">9,045 / 10,000 correct</div>
            </div>
        </div>
        <div class="metric-stat-card">
            <div class="metric-icon-circle">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#FBBF24" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="6 3 18 3 22 9 12 22 2 9 6 3"/></svg>
            </div>
            <div>
                <div class="metric-stat-val">{lr['precision']:.1f}%</div>
                <div class="metric-stat-title">Precision</div>
                <div class="metric-stat-sub">Positive predictive value</div>
            </div>
        </div>
        <div class="metric-stat-card">
            <div class="metric-icon-circle">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#F59E0B" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>
            </div>
            <div>
                <div class="metric-stat-val">{lr['recall']:.1f}%</div>
                <div class="metric-stat-title">Recall</div>
                <div class="metric-stat-sub">Sensitivity (True pos rate)</div>
            </div>
        </div>
        <div class="metric-stat-card">
            <div class="metric-icon-circle">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#FBBF24" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m16 16 3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1Z"/><path d="m2 16 3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1Z"/><path d="M7 21h10"/><path d="M12 3v18"/><path d="M3 7h2c2 0 5-1 7-2 2 1 5 2 7 2h2"/></svg>
            </div>
            <div>
                <div class="metric-stat-val">{lr['f1']:.1f}%</div>
                <div class="metric-stat-title">F1 Score</div>
                <div class="metric-stat-sub">Harmonic mean</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Comparison Table
    st.markdown("""
    <div class="m-card">
        <div class="m-card-header">
            <div class="m-card-title">
                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#F59E0B" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m16 16 3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1Z"/><path d="m2 16 3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1Z"/><path d="M7 21h10"/><path d="M12 3v18"/><path d="M3 7h2c2 0 5-1 7-2 2 1 5 2 7 2h2"/></svg>
                Model Comparison (Primary vs Baseline)
            </div>
            <span class="panel-model-badge">80/20 Stratified Test Split</span>
        </div>
        <table class="data-table">
            <thead>
                <tr>
                    <th>Model</th>
                    <th>Accuracy</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>F1 Score</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><strong>Logistic Regression (saga, C=5.0)</strong></td>
                    <td class="cell-lead">90.4%</td>
                    <td class="cell-lead">89.4%</td>
                    <td class="cell-lead">91.7%</td>
                    <td class="cell-lead">90.6%</td>
                    <td><span style="color:#F59E0B; font-size:0.72rem; font-weight:700; background:rgba(245,158,11,0.12); padding:2px 8px; border-radius:4px; border:1px solid rgba(245,158,11,0.3);">Primary Model</span></td>
                </tr>
                <tr>
                    <td>Multinomial Naive Bayes (&alpha;=0.1)</td>
                    <td style="font-family:'JetBrains Mono',monospace;">88.1%</td>
                    <td style="font-family:'JetBrains Mono',monospace;">87.4%</td>
                    <td style="font-family:'JetBrains Mono',monospace;">89.1%</td>
                    <td style="font-family:'JetBrains Mono',monospace;">88.3%</td>
                    <td><span style="color:#A3A196; font-size:0.72rem;">Baseline</span></td>
                </tr>
            </tbody>
        </table>
    </div>
    """, unsafe_allow_html=True)

    # Confusion Matrix & Global Insights
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        st.markdown(f"""
        <div class="m-card">
            <div class="m-card-header">
                <div class="m-card-title">
                    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#F59E0B" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2"/><line x1="3" y1="12" x2="21" y2="12"/><line x1="12" y1="3" x2="12" y2="21"/></svg>
                    Confusion Matrix
                </div>
                <span class="panel-model-badge">Logistic Regression</span>
            </div>
            <div class="matrix-grid">
                <div></div>
                <div class="matrix-hdr">Pred Negative</div>
                <div class="matrix-hdr">Pred Positive</div>
                <div class="matrix-hdr" style="justify-content:flex-start;">Actual Negative</div>
                <div class="matrix-cell primary-hit">
                    <div class="matrix-num">{lr['tn']:,}</div>
                    <div class="matrix-sub">True Negative</div>
                </div>
                <div class="matrix-cell error-hit">
                    <div class="matrix-num">{lr['fp']:,}</div>
                    <div class="matrix-sub">False Positive</div>
                </div>
                <div class="matrix-hdr" style="justify-content:flex-start;">Actual Positive</div>
                <div class="matrix-cell error-hit">
                    <div class="matrix-num">{lr['fn']:,}</div>
                    <div class="matrix-sub">False Negative</div>
                </div>
                <div class="matrix-cell primary-hit">
                    <div class="matrix-num">{lr['tp']:,}</div>
                    <div class="matrix-sub">True Positive</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_c2:
        pos_glob, neg_glob = get_global_top_features()
        st.markdown("""
        <div class="m-card">
            <div class="m-card-header">
                <div class="m-card-title">
                    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#F59E0B" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="2" y1="12" x2="22" y2="12"/><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/></svg>
                    Top Global Features
                </div>
                <span class="panel-model-badge">Learned Coefficients</span>
            </div>
        """, unsafe_allow_html=True)
        if pos_glob and neg_glob:
            st.markdown("<div style='font-size:0.74rem; font-weight:700; color:#22C55E; margin-bottom:6px;'>TOP POSITIVE DRIVERS</div>", unsafe_allow_html=True)
            for feat, coef_val in pos_glob[:4]:
                st.markdown(f"""
                <div class="word-contrib-row">
                    <div class="word-contrib-name">{feat}</div>
                    <div class="word-contrib-bar-track"><div class="word-contrib-fill-p" style="width:{int(coef_val/10.7*100)}%;"></div></div>
                    <div class="word-contrib-score pos">+{coef_val:.2f}</div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("<div style='font-size:0.74rem; font-weight:700; color:#F43F5E; margin:10px 0 6px;'>TOP NEGATIVE DRIVERS</div>", unsafe_allow_html=True)
            for feat, coef_val in neg_glob[:4]:
                st.markdown(f"""
                <div class="word-contrib-row">
                    <div class="word-contrib-name">{feat}</div>
                    <div class="word-contrib-bar-track"><div class="word-contrib-fill-n" style="width:{int(abs(coef_val)/15.0*100)}%;"></div></div>
                    <div class="word-contrib-score neg">{coef_val:.2f}</div>
                </div>
                """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Wordcloud Expander
    pos_wc = os.path.join(PLOTS_DIR, "wordcloud_positive.png")
    neg_wc = os.path.join(PLOTS_DIR, "wordcloud_negative.png")
    if os.path.exists(pos_wc) or os.path.exists(neg_wc):
        with st.expander("Explore WordClouds (Secondary Feature Visualizations)"):
            wc1, wc2 = st.columns(2)
            with wc1:
                if os.path.exists(pos_wc):
                    st.image(pos_wc, caption="Positive Signal WordCloud", use_column_width=True)
            with wc2:
                if os.path.exists(neg_wc):
                    st.image(neg_wc, caption="Negative Signal WordCloud", use_column_width=True)

# ─────────────────────────────────────────────────────────────
# 8. View: Dataset Explorer
# ─────────────────────────────────────────────────────────────
elif active_view == "Dataset Explorer":
    st.markdown("""
    <div class="top-app-bar">
        <div class="app-brand-inline">
            <span>●</span> Corpus & Architecture
        </div>
    </div>
    <div class="hero-header-row">
        <div>
            <h1 class="main-heading">Dataset <span>Explorer</span></h1>
            <p class="main-subheading">IMDb Large Movie Review dataset distribution and NLP engineering pipeline.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="metric-row-grid" style="margin-top:0; margin-bottom:1.5rem;">
        <div class="metric-stat-card">
            <div class="metric-icon-circle">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#F59E0B" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/></svg>
            </div>
            <div>
                <div class="metric-stat-val">50,000</div>
                <div class="metric-stat-title">Total Corpus</div>
                <div class="metric-stat-sub">IMDb Movie Reviews</div>
            </div>
        </div>
        <div class="metric-stat-card">
            <div class="metric-icon-circle">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#22C55E" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3zM7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3"/></svg>
            </div>
            <div>
                <div class="metric-stat-val">25,000</div>
                <div class="metric-stat-title">Positive Class</div>
                <div class="metric-stat-sub">50.0% of corpus</div>
            </div>
        </div>
        <div class="metric-stat-card">
            <div class="metric-icon-circle">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#F43F5E" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10 15v4a3 3 0 0 0 3 3l4-9V2H5.72a2 2 0 0 0-2 1.7l-1.38 9a2 2 0 0 0 2 2.3zm7-13h3a2 2 0 0 1 2 2v7a2 2 0 0 1-2 2h-3"/></svg>
            </div>
            <div>
                <div class="metric-stat-val">25,000</div>
                <div class="metric-stat-title">Negative Class</div>
                <div class="metric-stat-sub">50.0% of corpus</div>
            </div>
        </div>
        <div class="metric-stat-card">
            <div class="metric-icon-circle">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#FBBF24" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10 2v7.31"/><path d="M14 9.3V1.99"/><path d="M8.5 2h7"/><path d="M14 9.3a6.5 6.5 0 1 1-4 0"/><path d="M5.52 16h12.96"/></svg>
            </div>
            <div>
                <div class="metric-stat-val">10,000</div>
                <div class="metric-stat-title">Test Split</div>
                <div class="metric-stat-sub">20% held-out test</div>
            </div>
        </div>
    </div>

    <!-- Class Balance Visualization -->
    <div class="m-card" style="margin-bottom:1.5rem; padding:1.1rem 1.3rem;">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
            <div style="font-size:0.85rem; font-weight:700; color:#F5F5F0; display:flex; align-items:center; gap:8px;">
                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#F59E0B" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m16 16 3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1Z"/><path d="m2 16 3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1Z"/><path d="M7 21h10"/><path d="M12 3v18"/><path d="M3 7h2c2 0 5-1 7-2 2 1 5 2 7 2h2"/></svg>
                Corpus Class Balance (50 / 50 Balanced)
            </div>
            <div style="font-family:'JetBrains Mono',monospace; font-size:0.75rem; color:#A3A196;">50,000 total items</div>
        </div>
        <div style="width:100%; height:8px; background:#1B1A16; border-radius:4px; overflow:hidden; display:flex;">
            <div style="width:50%; background:#22C55E; height:100%;" title="Positive Reviews: 25,000 (50%)"></div>
            <div style="width:50%; background:#F43F5E; height:100%;" title="Negative Reviews: 25,000 (50%)"></div>
        </div>
        <div style="display:flex; justify-content:space-between; align-items:center; margin-top:8px; font-size:0.75rem; font-family:'JetBrains Mono',monospace;">
            <span style="color:#22C55E; font-weight:600;">● Positive: 25,000 (50.0%)</span>
            <span style="color:#F43F5E; font-weight:600;">● Negative: 25,000 (50.0%)</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="m-card">
        <div class="m-card-header">
            <div class="m-card-title">
                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#F59E0B" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>
                End-to-End NLP Pipeline Flow
            </div>
            <span class="panel-model-badge">NLP Pipeline</span>
        </div>
        <div style="display:grid; grid-template-columns:repeat(6, 1fr); gap:8px; text-align:center;">
            <div style="background:#1B1A16; border:1px solid #292820; border-radius:6px; padding:10px;">
                <div style="color:#FBBF24; font-family:'JetBrains Mono',monospace; font-size:0.72rem; font-weight:700;">01</div>
                <div style="font-size:0.8rem; font-weight:700; color:#F5F5F0; margin:2px 0;">IMDb Data</div>
                <div style="font-size:0.68rem; color:#A3A196;">50,000 reviews</div>
            </div>
            <div style="background:#1B1A16; border:1px solid #292820; border-radius:6px; padding:10px;">
                <div style="color:#FBBF24; font-family:'JetBrains Mono',monospace; font-size:0.72rem; font-weight:700;">02</div>
                <div style="font-size:0.8rem; font-weight:700; color:#F5F5F0; margin:2px 0;">Cleaning</div>
                <div style="font-size:0.68rem; color:#A3A196;">Regex & lemmatize</div>
            </div>
            <div style="background:#1B1A16; border:1px solid #292820; border-radius:6px; padding:10px;">
                <div style="color:#FBBF24; font-family:'JetBrains Mono',monospace; font-size:0.72rem; font-weight:700;">03</div>
                <div style="font-size:0.8rem; font-weight:700; color:#F5F5F0; margin:2px 0;">TF-IDF</div>
                <div style="font-size:0.68rem; color:#A3A196;">50k n-grams</div>
            </div>
            <div style="background:#1B1A16; border:1px solid #292820; border-radius:6px; padding:10px;">
                <div style="color:#FBBF24; font-family:'JetBrains Mono',monospace; font-size:0.72rem; font-weight:700;">04</div>
                <div style="font-size:0.8rem; font-weight:700; color:#F5F5F0; margin:2px 0;">Logistic Reg</div>
                <div style="font-size:0.68rem; color:#A3A196;">C=5.0, saga</div>
            </div>
            <div style="background:#1B1A16; border:1px solid #292820; border-radius:6px; padding:10px;">
                <div style="color:#FBBF24; font-family:'JetBrains Mono',monospace; font-size:0.72rem; font-weight:700;">05</div>
                <div style="font-size:0.8rem; font-weight:700; color:#F5F5F0; margin:2px 0;">Prediction</div>
                <div style="font-size:0.68rem; color:#A3A196;">Binary sentiment</div>
            </div>
            <div style="background:#1B1A16; border:1px solid #292820; border-radius:6px; padding:10px;">
                <div style="color:#FBBF24; font-family:'JetBrains Mono',monospace; font-size:0.72rem; font-weight:700;">06</div>
                <div style="font-size:0.8rem; font-weight:700; color:#F5F5F0; margin:2px 0;">Attribution</div>
                <div style="font-size:0.68rem; color:#A3A196;">TF-IDF &times; coef</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
