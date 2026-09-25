# ============================================================
# MOODMIX PDA — PROFESSIONAL REDESIGN v2
# Plotly interactive charts + AI hover-bot graph explainer
# ============================================================

import os
import math
from datetime import datetime
import warnings

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score,
    confusion_matrix, roc_curve, auc, classification_report
)

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================
st.set_page_config(
    page_title="MoodMix PDA",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🎵"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SONG_FILE               = os.path.join(BASE_DIR, "top10s.csv")
USERS_FILE              = os.path.join(BASE_DIR, "users.csv")
USER_PREF_FILE          = os.path.join(BASE_DIR, "user_preferences.csv")
HISTORY_FILE            = os.path.join(BASE_DIR, "listening_history.csv")
RATINGS_FILE            = os.path.join(BASE_DIR, "ratings.csv")
RECOMMENDATION_LOG_FILE = os.path.join(BASE_DIR, "recommendation_logs.csv")

TRACK_COL      = "title"
ARTIST_COL     = "artist"
GENRE_COL      = "top_genre"
POPULARITY_COL = "popularity"

NUMERIC_FEATURES = [
    "bpm", "nrgy", "dnce", "db", "live", "year",
    "val", "dur", "acous", "spch", "instrumentalness"
]

FEATURE_LABELS = {
    "bpm": "BPM (Tempo)", "nrgy": "Energy",
    "dnce": "Danceability", "db": "Loudness (dB)", "live": "Liveness",
    "val": "Valence", "dur": "Duration (s)", "acous": "Acousticness",
    "spch": "Speechiness", "popularity": "Popularity",
    "instrumentalness": "Instrumentalness", "Year": "Year"
}

# ============================================================
# PROFESSIONAL PLOTLY THEME
# ============================================================
PALETTE = {
    "bg":      "#0A0C12",
    "surface": "#10131C",
    "card":    "#161923",
    "border":  "#1E2235",
    "text":    "#E2E8F8",
    "muted":   "#5A6480",
    "accent1": "#7C6FFF",
    "accent2": "#FF5F7E",
    "accent3": "#3DEBA8",
    "accent4": "#FFB84D",
    "accent5": "#4DC8F5",
}

CHART_COLORS = [
    "#7C6FFF", "#FF5F7E", "#3DEBA8", "#FFB84D", "#4DC8F5",
    "#B06FFF", "#FF8C5A", "#5BFFCD", "#FFD97A", "#7FD8FF",
]

MOOD_COLORS = {
    "Fresh": "#3DEBA8", "Focus": "#7C6FFF", "Relax": "#4DC8F5",
    "Happy": "#FFB84D", "Party": "#FF5F7E", "Sad": "#9CA3AF",
    "Balanced": "#C5CCFF",
}
TIME_COLORS = {"Morning": "#FFB84D", "Afternoon": "#7C6FFF", "Evening": "#FF5F7E", "Night": "#4DC8F5"}

def get_adaptive_colors(labels, mode="default"):
    if mode == "mood":
        return [MOOD_COLORS.get(str(label), CHART_COLORS[i % len(CHART_COLORS)]) for i, label in enumerate(labels)]
    if mode == "time":
        return [TIME_COLORS.get(str(label), CHART_COLORS[i % len(CHART_COLORS)]) for i, label in enumerate(labels)]
    return [CHART_COLORS[i % len(CHART_COLORS)] for i, _ in enumerate(labels)]

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(16,19,28,0.88)",
    font=dict(family="'Sora', 'DM Sans', sans-serif", color="#EAF0FF", size=13),
    title_font=dict(family="'Sora', sans-serif", size=16, color="#FFFFFF"),
    legend=dict(
        bgcolor="rgba(22,25,35,0.95)",
        bordercolor="#3A4168",
        borderwidth=1,
        font=dict(size=13, color="#FFFFFF"),
    ),
    xaxis=dict(
        gridcolor="#2B3150",
        gridwidth=0.7,
        linecolor="#3A4168",
        tickfont=dict(size=12, color="#FFFFFF"),
        title_font=dict(size=13, color="#FFFFFF"),
        zeroline=False,
    ),
    yaxis=dict(
        gridcolor="#2B3150",
        gridwidth=0.7,
        linecolor="#3A4168",
        tickfont=dict(size=12, color="#FFFFFF"),
        title_font=dict(size=13, color="#FFFFFF"),
        zeroline=False,
    ),
    margin=dict(l=62, r=35, t=58, b=55),
    hoverlabel=dict(
        bgcolor="#161923",
        bordercolor="#7C6FFF",
        font=dict(family="'DM Sans', sans-serif", size=14, color="#FFFFFF"),
    ),
)

def apply_layout(fig, title="", height=400, **kwargs):
    layout = dict(PLOTLY_LAYOUT)
    layout.update(kwargs)
    if title:
        layout["title"] = dict(text=title, x=0.01, xanchor="left", font=dict(size=16, color="#FFFFFF"))
    layout["height"] = height
    fig.update_layout(**layout)
    fig.update_xaxes(
    tickfont=dict(color="#FFFFFF"),
    title_font=dict(color="#FFFFFF")
    )

    fig.update_yaxes(
    tickfont=dict(color="#FFFFFF"),
    title_font=dict(color="#FFFFFF")
    )
    return fig

def show_chart(fig, chart_id, description):
    """Render a Plotly chart with an AI-bot hover overlay."""
    fig.update_layout(
        modebar_remove=["lasso2d", "select2d", "autoScale2d"],
        modebar_bgcolor="rgba(0,0,0,0)",
        modebar_color=PALETTE["muted"],
        modebar_activecolor=PALETTE["accent1"],
    )
    # Store chart descriptions for the bot
    if "chart_descriptions" not in st.session_state:
        st.session_state["chart_descriptions"] = {}
    st.session_state["chart_descriptions"][chart_id] = description

    # Premium chart polish
    fig.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(16,19,28,0.88)",
    font=dict(color="#FFFFFF", size=13),
    )

    fig.update_xaxes(
    tickfont=dict(color="#FFFFFF", size= 13),
    title_font=dict(color="#FFFFFF", size=13)
    )

    fig.update_yaxes(
    tickfont=dict(color="#FFFFFF", size=13),
    title_font=dict(color="#FFFFFF", size=13)
    )

    st.markdown('<div class="chart-shell">', unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True, key=chart_id, config={
        "modeBarButtonsToRemove": [
            "zoom2d",   # removes box zoom
            "pan2d",    # removes pan
            "select2d",
            "lasso2d"
        ]
    })
    st.markdown('</div>', unsafe_allow_html=True)

    # Render AI bot explanation panel beneath chart
    render_ai_bot_panel(chart_id, description)


def render_ai_bot_panel(chart_id, description):
    """Renders a sleek inline AI explanation card below each chart."""
    bot_html = f"""
    <div class="ai-bot-panel" id="bot-{chart_id}">
        <div class="bot-avatar">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
                <circle cx="12" cy="12" r="10" fill="#7C6FFF" opacity="0.2"/>
                <path d="M9 9h.01M15 9h.01M8 13s1.5 2 4 2 4-2 4-2" stroke="#7C6FFF" stroke-width="2" stroke-linecap="round"/>
                <circle cx="12" cy="12" r="10" stroke="#7C6FFF" stroke-width="1.5"/>
            </svg>
        </div>
        <div class="bot-content">
            <div class="bot-label">Chart Insight</div>
            <div class="bot-text">{description}</div>
        </div>
        <div class="bot-glow"></div>
    </div>
    """
    st.markdown(bot_html, unsafe_allow_html=True)


# ============================================================
# CSS — full professional redesign
# ============================================================
def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700;800&family=DM+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {
        --bg:      #0A0C12;
        --surface: #10131C;
        --card:    #161923;
        --border:  #1E2235;
        --text:    #E2E8F8;
        --muted:   #5A6480;
        --accent1: #7C6FFF;
        --accent2: #FF5F7E;
        --accent3: #3DEBA8;
        --accent4: #FFB84D;
        --accent5: #4DC8F5;
        --radius:  14px;
    }

    html, body, [data-testid="stAppViewContainer"], .stApp,
    [data-testid="stHeader"] {
        background: var(--bg) !important;
        color: var(--text) !important;
        font-family: 'DM Sans', sans-serif !important;
    }

    #MainMenu, footer {
    visibility: hidden;
}

/* Do NOT hide Streamlit toolbar/sidebar controls */
[data-testid="stToolbar"] {
    visibility: visible !important;
}

    .block-container {
        padding: 1.2rem 2.2rem 3rem !important;
        max-width: 1440px !important;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: var(--surface) !important;
        border-right: 1px solid var(--border) !important;
    }
    [data-testid="stSidebar"] * { color: var(--text) !important; }

    /* ── Brand ── */
    .brand-name {
        font-family: 'Sora', sans-serif;
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.3em;
        text-transform: uppercase;
        background: linear-gradient(90deg, var(--accent1), var(--accent2));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .page-title {
        font-family: 'Sora', sans-serif;
        font-size: 2.5rem;
        font-weight: 800;
        color: var(--text);
        line-height: 1.1;
        margin: 2px 0 4px;
        letter-spacing: -0.02em;
    }

    .page-subtitle {
        font-size: 1rem;
        color: #C5CCFF;
        margin-bottom: 0;
    }

    .divider {
        height: 1px;
        background: linear-gradient(90deg, var(--accent1) 0%, var(--accent2) 40%, transparent 100%);
        border: none;
        margin: 10px 0 20px;
        opacity: 0.6;
    }

    /* ── Metric Cards ── */
    .metric-grid { display: flex; gap: 12px; margin-bottom: 20px; flex-wrap: wrap; }

    .metric-card {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 18px 20px;
        flex: 1;
        min-width: 110px;
        position: relative;
        overflow: hidden;
        transition: border-color 0.2s, transform 0.2s;
    }

    .metric-card:hover {
        border-color: var(--accent1);
        transform: translateY(-2px);
    }

    .metric-card::after {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, var(--accent1), var(--accent2));
    }

    .metric-label {
        font-size: 0.72rem;
        color: var(--muted);
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin-bottom: 8px;
    }

    .metric-value {
        font-family: 'Sora', sans-serif;
        font-size: 1.9rem;
        font-weight: 800;
        color: var(--text);
        line-height: 1;
    }

    .metric-delta {
        font-size: 0.76rem;
        color: var(--accent3);
        margin-top: 4px;
    }

    /* ── Glass Cards ── */
    .glass-card {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 20px 22px;
        margin-bottom: 14px;
        position: relative;
        overflow: hidden;
    }

    .glass-card::before {
        content: '';
        position: absolute;
        inset: 0;
        background: linear-gradient(135deg, rgba(124,111,255,0.05) 0%, transparent 60%);
        pointer-events: none;
    }

    /* ── AI Bot Panel ── */
    .ai-bot-panel {
        display: flex;
        align-items: flex-start;
        gap: 12px;
        background: linear-gradient(135deg, rgba(124,111,255,0.07) 0%, rgba(61,235,168,0.04) 100%);
        border: 1px solid rgba(124,111,255,0.22);
        border-radius: 12px;
        padding: 12px 16px;
        margin: -8px 0 18px;
        position: relative;
        overflow: hidden;
        animation: botFadeIn 0.4s ease forwards;
    }

    @keyframes botFadeIn {
        from { opacity: 0; transform: translateY(6px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    .ai-bot-panel::before {
        content: '';
        position: absolute;
        left: 0; top: 0; bottom: 0;
        width: 3px;
        background: linear-gradient(180deg, var(--accent1), var(--accent3));
        border-radius: 3px 0 0 3px;
    }

    .bot-glow {
        position: absolute;
        width: 120px; height: 120px;
        background: radial-gradient(circle, rgba(124,111,255,0.08) 0%, transparent 70%);
        top: -40px; right: -30px;
        pointer-events: none;
    }

    .bot-avatar {
        flex-shrink: 0;
        width: 32px; height: 32px;
        background: rgba(124,111,255,0.12);
        border: 1px solid rgba(124,111,255,0.3);
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-top: 2px;
    }

    .bot-content { flex: 1; }

    .bot-label {
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--accent1);
        margin-bottom: 4px;
    }

    .bot-text {
        font-size: 0.84rem;
        color: #A8B2D8;
        line-height: 1.6;
    }

    /* ── Section Heading ── */
    .section-heading {
        font-family: 'Sora', sans-serif;
        font-size: 1.05rem;
        font-weight: 700;
        color: var(--text);
        margin: 22px 0 12px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .section-heading::after {
        content: '';
        flex: 1;
        height: 1px;
        background: var(--border);
        margin-left: 8px;
    }

    /* ── Song Cards ── */
    .song-card {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 16px 18px;
        margin-bottom: 10px;
        transition: border-color 0.2s, box-shadow 0.2s;
    }

    .song-card:hover {
        border-color: var(--accent1);
        box-shadow: 0 4px 24px rgba(124,111,255,0.12);
    }

    .song-title {
        font-family: 'Sora', sans-serif;
        font-size: 1.02rem;
        font-weight: 700;
        color: var(--text);
        margin-bottom: 2px;
    }

    .song-artist { font-size: 0.9rem; font-weight: 600; color: var(--accent1); margin-bottom: 6px; }
    .song-meta   { font-size: 0.8rem; color: var(--muted); font-family: 'JetBrains Mono', monospace; }

    .reason-box {
        background: rgba(124,111,255,0.07);
        border: 1px solid rgba(124,111,255,0.2);
        border-radius: 8px;
        padding: 9px 13px;
        margin-top: 8px;
        font-size: 0.83rem;
        color: #B0B8D8;
        line-height: 1.5;
    }

    /* ── Badges & Pills ── */
    .badge { display: inline-block; padding: 3px 10px; border-radius: 99px; font-size: 0.72rem; font-weight: 700; margin-left: 6px; }
    .badge-high { background: rgba(61,235,168,0.12); color: #3DEBA8; border: 1px solid rgba(61,235,168,0.28); }
    .badge-mid  { background: rgba(255,184,77,0.12);  color: #FFB84D; border: 1px solid rgba(255,184,77,0.28); }
    .badge-low  { background: rgba(255,95,126,0.12);  color: #FF5F7E; border: 1px solid rgba(255,95,126,0.28); }

    .pill {
        display: inline-block;
        background: rgba(124,111,255,0.1);
        color: #A49DFF;
        border: 1px solid rgba(124,111,255,0.22);
        padding: 4px 12px; border-radius: 99px;
        font-size: 0.78rem; font-weight: 600;
        margin: 3px 4px 3px 0;
    }

    /* ── Hero ── */
    .hero {
        background: linear-gradient(135deg, rgba(124,111,255,0.12), rgba(255,95,126,0.07), rgba(61,235,168,0.06));
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 28px 30px;
        margin-bottom: 20px;
        position: relative;
        overflow: hidden;
    }

    .hero::before {
        content: '';
        position: absolute;
        width: 280px; height: 280px;
        background: radial-gradient(circle, rgba(124,111,255,0.1) 0%, transparent 70%);
        top: -100px; right: -50px;
        pointer-events: none;
    }

    .hero-title { font-family: 'Sora', sans-serif; font-size: 1.75rem; font-weight: 800; margin-bottom: 6px; }
    .hero-sub {
    font-size: 1.00rem;              /* slightly bigger */
    color: #C5CCFF;                 /* brighter bluish-white */
    font-weight: 500;
    line-height: 1.6;
}

    /* ── Formula Box ── */
    .formula-box {
        background: rgba(124,111,255,0.05);
        border: 1px solid rgba(124,111,255,0.18);
        border-radius: 10px;
        padding: 14px 18px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        color: #A49DFF;
        line-height: 1.8;
    }

    /* ── Buttons ── */
    .stButton > button, .stFormSubmitButton > button {
        background: linear-gradient(135deg, var(--accent1), #9B5CF6) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.55rem 1.2rem !important;
        font-family: 'DM Sans', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        box-shadow: 0 4px 18px rgba(124,111,255,0.28) !important;
        transition: all 0.2s !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 28px rgba(124,111,255,0.42) !important;
    }

/* ── Inputs (FIXED VISIBILITY) ── */
.stTextInput label,
.stTextInput div[data-baseweb="input"] label,
.stTextInput label span {
    color: #FFFFFF !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
}

/* Input box text */
.stTextInput > div > div > input {
    background: var(--card) !important;
    border: 1px solid #444 !important;
    color: #FFFFFF !important;
    font-size: 1rem !important;
    padding: 10px !important;
    border-radius: 10px !important;
}

/* Placeholder text */
.stTextInput input::placeholder {
    color: #A0A8C0 !important;  /* brighter than muted */
    opacity: 1 !important;
    font-size: 0.95rem !important;
}

/* Email + Username field border glow on focus */
.stTextInput input:focus {
    border: 1px solid #7C6FFF !important;
    box-shadow: 0 0 0 1px #7C6FFF !important;
    outline: none !important;
}

/* Select / dropdown */
.stSelectbox > div > div,
.stMultiSelect > div > div {
    background: var(--card) !important;
    border: 1px solid #444 !important;
    color: #FFFFFF !important;
    font-size: 0.95rem !important;
}

    /* ── Expanders ── */
    .streamlit-expanderHeader {
        background: var(--card) !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
        font-family: 'Sora', sans-serif !important;
        font-weight: 600 !important;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        background: var(--surface) !important;
        border-radius: 12px !important;
        padding: 4px !important;
        gap: 2px !important;
        border: 1px solid var(--border) !important;
    }

    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        border-radius: 8px !important;
        color: var(--muted) !important;
        font-family: 'Sora', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.82rem !important;
        padding: 8px 14px !important;
    }

    .stTabs [aria-selected="true"] {
        background: var(--card) !important;
        color: var(--text) !important;
        border: 1px solid var(--border) !important;
    }

    .stDataFrame { border: 1px solid var(--border) !important; border-radius: 10px !important; }

    /* ===== PREMIUM MUSIC-APP THEME FIXES ===== */
    .page-subtitle {
        font-size: 0.78rem !important;
        color: #AAB2FF !important;
        font-weight: 500 !important;
        line-height: 1.3 !important;
        letter-spacing: 0.02em !important;
    }
    .hero-sub {
        font-size: 1.03rem !important;
        color: #DDE4FF !important;
        font-weight: 500 !important;
        line-height: 1.65 !important;
    }
    .glass-card, .metric-card, .song-card, .music-card { box-shadow: 0 0 22px rgba(124,111,255,0.10) !important; }
    .metric-card:hover, .glass-card:hover, .song-card:hover, .music-card:hover {
        border-color: #8F83FF !important;
        box-shadow: 0 0 30px rgba(124,111,255,0.25) !important;
    }
    .chart-shell {
        border: 1px solid #2B3150;
        border-radius: 16px;
        padding: 10px 10px 2px 10px;
        background: rgba(16,19,28,0.38);
        box-shadow: 0 0 28px rgba(124,111,255,0.16);
        margin-bottom: 12px;
    }
    [data-testid="stExpander"] {
        border: 1px solid #2B3150 !important;
        border-radius: 14px !important;
        background: rgba(16,19,28,0.55) !important;
        overflow: hidden !important;
    }
    [data-testid="stExpander"] summary, .streamlit-expanderHeader {
        background: #161923 !important;
        color: #FFFFFF !important;
        border: 1px solid #2B3150 !important;
        border-radius: 12px !important;
        font-size: 0.95rem !important;
        font-weight: 700 !important;
    }
    [data-testid="stExpander"] summary p, [data-testid="stExpander"] summary span {
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }
    [data-testid="stSidebar"] {
        background: #10131C !important;
        border-right: 1px solid #2B3150 !important;
    }
    [data-testid="stSidebar"] * { color: #FFFFFF !important; }
    [data-testid="stSidebar"] .stRadio label { font-size: 0.95rem !important; font-weight: 650 !important; }
    .stTabs [data-baseweb="tab"] { font-size: 0.82rem !important; color: #AAB2FF !important; }
    .stTabs [aria-selected="true"] { color: #FFFFFF !important; border-bottom: 2px solid #FF5F7E !important; }
    input, textarea { color: #FFFFFF !important; font-size: 1rem !important; }
    input::placeholder, textarea::placeholder { color: #C5CCFF !important; opacity: 1 !important; }
    label { color: #FFFFFF !important; font-size: 1rem !important; font-weight: 600 !important; }
    .music-row-title {
        font-family: 'Sora', sans-serif;
        font-size: 1.12rem;
        font-weight: 800;
        color: #FFFFFF;
        margin: 20px 0 4px;
    }
    .music-row-subtitle { color: #AAB2FF; font-size: 0.82rem; margin-bottom: 10px; }
    .music-card {
        background: linear-gradient(145deg, #181B28, #10131C);
        border: 1px solid #2B3150;
        border-radius: 18px;
        padding: 15px 15px 14px;
        min-height: 176px;
        transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
        position: relative;
        overflow: hidden;
    }
    .music-card::before {
        content: '';
        position: absolute;
        width: 120px;
        height: 120px;
        right: -45px;
        top: -45px;
        background: radial-gradient(circle, rgba(124,111,255,0.20), transparent 70%);
    }
    .music-card:hover { transform: translateY(-4px); }
    .music-title {
        color: #FFFFFF;
        font-size: 0.96rem;
        font-weight: 800;
        line-height: 1.25;
        margin-bottom: 5px;
        position: relative;
    }
    .music-artist {
        color: #B8C0FF;
        font-size: 0.82rem;
        font-weight: 600;
        margin-bottom: 8px;
        position: relative;
    }
    .music-pill {
        display: inline-block;
        color: #FFFFFF;
        background: rgba(124,111,255,0.18);
        border: 1px solid rgba(124,111,255,0.35);
        border-radius: 999px;
        font-size: 0.68rem;
        font-weight: 700;
        padding: 3px 8px;
        margin: 0 4px 8px 0;
        position: relative;
    }
    .music-reason { color: #DDE4FF; font-size: 0.75rem; line-height: 1.4; position: relative; }
    .bot-text { font-size: 0.78rem !important; color: #C9D1FF !important; }

    /* ===== FIX FILTER BY GENRE / MULTISELECT VISIBILITY ===== */
    div[data-testid="stMultiSelect"] label,
    div[data-testid="stMultiSelect"] label p {
        color: #FFFFFF !important;
        font-size: 1rem !important;
        font-weight: 700 !important;
    }

    div[data-testid="stMultiSelect"] [data-baseweb="select"] > div {
        background-color: #161923 !important;
        border: 1px solid #3A4168 !important;
        border-radius: 12px !important;
        color: #FFFFFF !important;
        min-height: 44px !important;
        box-shadow: 0 0 0 1px rgba(124,111,255,0.12) !important;
    }

    div[data-testid="stMultiSelect"] [data-baseweb="select"] input,
    div[data-testid="stMultiSelect"] [data-baseweb="select"] span,
    div[data-testid="stMultiSelect"] [data-baseweb="select"] div {
        color: #FFFFFF !important;
        opacity: 1 !important;
        caret-color: #FFFFFF !important;
    }

    div[data-testid="stMultiSelect"] [data-baseweb="select"] input::placeholder {
        color: #DDE4FF !important;
        opacity: 1 !important;
    }

    [data-baseweb="popover"] [data-baseweb="menu"],
    [data-baseweb="menu"] {
        background-color: #161923 !important;
        border: 1px solid #3A4168 !important;
        border-radius: 12px !important;
        box-shadow: 0 12px 35px rgba(0,0,0,0.55) !important;
    }

    [data-baseweb="menu"] li,
    [data-baseweb="menu"] div,
    [role="option"],
    [role="option"] * {
        color: #FFFFFF !important;
        background-color: #161923 !important;
        font-size: 0.95rem !important;
    }

    [data-baseweb="menu"] li:hover,
    [role="option"]:hover,
    [aria-selected="true"] {
        background-color: #7C6FFF !important;
        color: #FFFFFF !important;
    }

    /* Keep selected genre chips readable */
    div[data-testid="stMultiSelect"] span[data-baseweb="tag"] {
        background: rgba(124,111,255,0.24) !important;
        border: 1px solid rgba(124,111,255,0.42) !important;
        color: #FFFFFF !important;
    }

    </style>
    """, unsafe_allow_html=True)


inject_css()

# ============================================================
# SESSION STATE
# ============================================================
def init_session():
    defaults = {"logged_in": False, "user_id": None, "username": None, "email": None, "page": "Welcome", "explore_page_num": 1}
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()

# ============================================================
# CSV BACKEND
# ============================================================
def initialize_backend_files():
    schemas = {
        USERS_FILE:       ["user_id", "username", "email", "created_at"],
        USER_PREF_FILE:   ["user_id", "preferred_genres", "created_at", "updated_at"],
        HISTORY_FILE:     ["history_id", "user_id", "title", "artist", "top_genre", "time_period", "mood", "listened_at"],
        RATINGS_FILE:     ["rating_id", "user_id", "title", "artist", "top_genre", "rating", "liked", "time_period", "rated_at"],
        RECOMMENDATION_LOG_FILE: ["log_id", "user_id", "title", "artist", "top_genre", "time_period", "final_score", "reason", "recommended_at"],
    }
    for path, cols in schemas.items():
        if not os.path.exists(path):
            pd.DataFrame(columns=cols).to_csv(path, index=False)

initialize_backend_files()

def read_csv_safe(path, columns):
    if not os.path.exists(path):
        df = pd.DataFrame(columns=columns); df.to_csv(path, index=False); return df
    try: df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        df = pd.DataFrame(columns=columns); df.to_csv(path, index=False); return df
    changed = False
    for col in columns:
        if col not in df.columns: df[col] = pd.NA; changed = True
    df = df[[c for c in columns] + [c for c in df.columns if c not in columns]]
    id_cols = [c for c in columns if c.endswith("_id") and c != "user_id"]
    for id_col in id_cols:
        if id_col in df.columns:
            if df[id_col].isna().all():
                df[id_col] = range(1, len(df)+1); changed = True
            else:
                df[id_col] = pd.to_numeric(df[id_col], errors="coerce")
                mask = df[id_col].isna()
                if mask.any():
                    cur_max = int(df.loc[~mask, id_col].max()) if (~mask).any() else 0
                    df.loc[mask, id_col] = range(cur_max+1, cur_max+1+mask.sum()); changed = True
    if changed: df.to_csv(path, index=False)
    return df

def next_numeric_id(df, id_col):
    if df.empty or id_col not in df.columns: return 1
    v = pd.to_numeric(df[id_col], errors="coerce")
    return 1 if v.dropna().empty else int(v.max()) + 1

load_users                = lambda: read_csv_safe(USERS_FILE, ["user_id","username","email","created_at"])
save_users                = lambda df: df.to_csv(USERS_FILE, index=False)
load_user_preferences     = lambda: read_csv_safe(USER_PREF_FILE, ["user_id","preferred_genres","created_at","updated_at"])
save_user_preferences     = lambda df: df.to_csv(USER_PREF_FILE, index=False)
load_history              = lambda: read_csv_safe(HISTORY_FILE, ["history_id","user_id","title","artist","top_genre","time_period","mood","listened_at"])
save_history              = lambda df: df.to_csv(HISTORY_FILE, index=False)
load_ratings              = lambda: read_csv_safe(RATINGS_FILE, ["rating_id","user_id","title","artist","top_genre","rating","liked","time_period","rated_at"])
save_ratings              = lambda df: df.to_csv(RATINGS_FILE, index=False)
load_recommendation_logs  = lambda: read_csv_safe(RECOMMENDATION_LOG_FILE, ["log_id","user_id","title","artist","top_genre","time_period","final_score","reason","recommended_at"])
save_recommendation_logs  = lambda df: df.to_csv(RECOMMENDATION_LOG_FILE, index=False)

# ============================================================
# HELPERS
# ============================================================
def sx(v): return str(v).replace("<","&lt;").replace(">","&gt;")

def get_time_period():
    h = datetime.now().hour
    if 5<=h<12: return "Morning"
    if 12<=h<17: return "Afternoon"
    if 17<=h<21: return "Evening"
    return "Night"

def get_default_mood(tp):
    return {"Morning":"Fresh","Afternoon":"Focus","Evening":"Relax","Night":"Party"}.get(tp,"Balanced")

def split_genres(v):
    if pd.isna(v) or str(v).strip()=="": return []
    return [g.strip() for g in str(v).split(";") if g.strip()]

def score_badge(score):
    pct = round(float(score)*100)
    cls = "badge-high" if score>=0.75 else ("badge-mid" if score>=0.5 else "badge-low")
    return f'<span class="badge {cls}">{pct}% match</span>'

def render_header(title, subtitle=""):
    st.markdown(f'<div class="brand-name">MoodMix PDA</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="page-title">{title}</div>', unsafe_allow_html=True)
    if subtitle: st.markdown(f'<div class="page-subtitle">{subtitle}</div>', unsafe_allow_html=True)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

def metric_card(label, value, delta=""):
    delta_html = f'<div class="metric-delta">↑ {delta}</div>' if delta else ""
    return f"""<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div>{delta_html}</div>"""

def render_metrics(*cards):
    if not cards: return
    cols = st.columns(len(cards))
    for col, card in zip(cols, cards):
        with col: st.markdown(card, unsafe_allow_html=True)

def render_song_card(title, artist, meta_lines=None, reason=None, score=None):
    meta_lines = meta_lines or []
    badge = score_badge(score) if score is not None else ""
    meta = " &nbsp;|&nbsp; ".join(sx(l) for l in meta_lines)
    reason_html = f'<div class="reason-box">💡 {sx(reason)}</div>' if reason else ""
    st.markdown(f"""
    <div class="song-card">
        <div class="song-title">🎵 {sx(title)} {badge}</div>
        <div class="song-artist">🎤 {sx(artist)}</div>
        <div class="song-meta">{meta}</div>
        {reason_html}
    </div>""", unsafe_allow_html=True)

def render_music_card_grid(rows, row_title, row_subtitle='', max_cards=4, reason_prefix='Because it matches your listening pattern'):
    """Render compact Spotify/Netflix-style song cards without removing analytics content."""
    if rows is None or len(rows) == 0:
        return
    st.markdown(f'<div class="music-row-title">{sx(row_title)}</div>', unsafe_allow_html=True)
    if row_subtitle:
        st.markdown(f'<div class="music-row-subtitle">{sx(row_subtitle)}</div>', unsafe_allow_html=True)
    cols = st.columns(min(max_cards, len(rows)))
    for col, (_, row) in zip(cols, rows.head(max_cards).iterrows()):
        title = sx(row.get(TRACK_COL, 'Unknown Song'))
        artist = sx(row.get(ARTIST_COL, 'Unknown Artist'))
        genre = sx(row.get(GENRE_COL, 'Genre'))
        pop = row.get(POPULARITY_COL, None)
        pop_text = f'{float(pop):.0f}% popularity' if pd.notna(pop) else 'Recommended'
        reason = sx(reason_prefix)
        with col:
            st.markdown(f'''
            <div class="music-card">
                <div class="music-title">🎵 {title}</div>
                <div class="music-artist">{artist}</div>
                <span class="music-pill">{genre}</span>
                <span class="music-pill">{pop_text}</span>
                <div class="music-reason">{reason}</div>
            </div>
            ''', unsafe_allow_html=True)

def section_heading(label, emoji=""):
    st.markdown(f'<div class="section-heading">{emoji} {label}</div>', unsafe_allow_html=True)

# ============================================================
# DATA + NORMALIZATION
# ============================================================
def normalize_feature_columns(df):
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower().str.replace(" ","_",regex=False)
    for c in list(df.columns):
        if c.startswith("unnamed"): df = df.drop(columns=[c])
    rename_map = {
        "track_name":"title","artists":"artist","track_genre":"top_genre",
        "tempo":"bpm","energy":"nrgy","danceability":"dnce","loudness":"db",
        "liveness":"live","valence":"val","duration_ms":"dur",
        "acousticness":"acous","speechiness":"spch","pop":"popularity","album_name":"album"
    }
    for old_col, new_col in rename_map.items():
        if old_col in df.columns and new_col not in df.columns:
            df = df.rename(columns={old_col: new_col})
    fraction_cols = ["dnce","nrgy","live","val","acous","spch","instrumentalness"]
    for col in fraction_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            if df[col].dropna().max() <= 1.5: df[col] = df[col] * 100
    if "dur" in df.columns:
        df["dur"] = pd.to_numeric(df["dur"], errors="coerce")
        if df["dur"].dropna().median() > 1000: df["dur"] = df["dur"] / 1000
    if "db" in df.columns:
        df["db"] = pd.to_numeric(df["db"], errors="coerce")
        db_min, db_max = df["db"].min(), df["db"].max()
        if db_min < 0: df["db"] = ((df["db"] - db_min) / (db_max - db_min + 1e-9)) * 100
    if "bpm" in df.columns: df["bpm"] = pd.to_numeric(df["bpm"], errors="coerce")
    df = df.drop_duplicates()
    if TRACK_COL in df.columns and ARTIST_COL in df.columns:
        df = df.drop_duplicates(subset=[TRACK_COL, ARTIST_COL])
    return df

@st.cache_data(show_spinner=False)
def load_song_data():
    df = pd.read_csv(SONG_FILE, encoding="latin1")
    df = normalize_feature_columns(df)
    required = [TRACK_COL, ARTIST_COL, GENRE_COL, POPULARITY_COL]
    missing = [c for c in required if c not in df.columns]
    if missing: raise ValueError("Missing cols: " + ", ".join(missing))
    available = [c for c in NUMERIC_FEATURES if c in df.columns]
    if not available: raise ValueError("No numeric audio feature columns found.")
    for col in available + [POPULARITY_COL]: df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=[POPULARITY_COL]).reset_index(drop=True)
    MAX_ROWS = 6000
    if len(df) > MAX_ROWS: df = df.sample(MAX_ROWS, random_state=42).reset_index(drop=True)
    return df, available

@st.cache_resource(show_spinner=False)
def train_models_cached(df, numeric_features):
    """Train all models used in the frontend analytics dashboard.

    This keeps the Streamlit frontend aligned with the notebook analysis:
    regression, user-preference classification, and genre classification.
    """
    numeric_features = list(numeric_features)
    X = df[numeric_features]
    y = df[POPULARITY_COL]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.30, random_state=42)

    def regression_metrics(y_true, y_pred):
        return {
            "MAE": float(mean_absolute_error(y_true, y_pred)),
            "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "R2": float(r2_score(y_true, y_pred)),
        }

    # ---------------- Regression models ----------------
    reg_models = {
        "Linear Regression": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("model", LinearRegression())
        ]),
        "Polynomial Regression": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("model", LinearRegression())
        ]),
        "Decision Tree Regressor": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("model", DecisionTreeRegressor(max_depth=3, min_samples_leaf=2, min_samples_split=2, random_state=42))
        ]),
        "Random Forest Regressor": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(n_estimators=200, max_depth=5, min_samples_leaf=2,
                                            min_samples_split=5, random_state=42, n_jobs=-1))
        ]),
        "Log-Transformed Random Forest": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(n_estimators=300, max_depth=5, min_samples_leaf=2,
                                            min_samples_split=5, random_state=42, n_jobs=-1))
        ]),
    }

    regression_results = {}
    fitted_reg_models = {}
    for name, model in reg_models.items():
        if name == "Log-Transformed Random Forest":
            model.fit(X_tr, np.log1p(y_tr))
            pred = np.expm1(model.predict(X_te))
        else:
            model.fit(X_tr, y_tr)
            pred = model.predict(X_te)
        pred = np.clip(pred, 0, 100)
        regression_results[name] = {**regression_metrics(y_te, pred), "pred": pred}
        fitted_reg_models[name] = model

    best_reg_name = min(regression_results, key=lambda n: regression_results[n]["RMSE"])
    best_reg_model = fitted_reg_models[best_reg_name]

    # ---------------- User preference classification ----------------
    # In this project, liked_proxy provides a consistent binary preference label from popularity
    # for the global dashboard. Real ratings are still captured in ratings.csv for personalization.
    temp = df.copy()
    temp["liked_proxy"] = np.where(temp[POPULARITY_COL] >= temp[POPULARITY_COL].median(), 1, 0)
    Xc, yc = temp[numeric_features], temp["liked_proxy"]
    Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(Xc, yc, test_size=0.30, random_state=42, stratify=yc)

    clf_models = {
        "Logistic Reg": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, random_state=42))
        ]),
        "Decision Tree": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("model", DecisionTreeClassifier(max_depth=8, random_state=42))
        ]),
        "Random Forest": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(n_estimators=120, random_state=42, n_jobs=-1))
        ]),
        "SVM": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("model", SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=42))
        ]),
    }

    classification_results = {}
    fitted_clf_models = {}
    for name, clf in clf_models.items():
        clf.fit(Xc_tr, yc_tr)
        pred = clf.predict(Xc_te)
        prob = clf.predict_proba(Xc_te)[:, 1]
        fpr, tpr, _ = roc_curve(yc_te, prob)
        classification_results[name] = {
            "Accuracy": float(accuracy_score(yc_te, pred)),
            "Precision": float(precision_score(yc_te, pred, zero_division=0)),
            "Recall": float(recall_score(yc_te, pred, zero_division=0)),
            "F1": float((2 * precision_score(yc_te, pred, zero_division=0) * recall_score(yc_te, pred, zero_division=0)) /
                        (precision_score(yc_te, pred, zero_division=0) + recall_score(yc_te, pred, zero_division=0) + 1e-9)),
            "AUC": float(auc(fpr, tpr)),
            "fpr": fpr, "tpr": tpr, "pred": pred, "prob": prob,
            "cm": confusion_matrix(yc_te, pred).tolist(),
        }
        fitted_clf_models[name] = clf

    # ---------------- Genre classification ----------------
    genre_results = {}
    fitted_genre_models = {}
    genre_payload = {"available": False}
    try:
        genre_df = df.dropna(subset=[GENRE_COL]).copy()
        genre_counts = genre_df[GENRE_COL].value_counts()
        valid_genres = genre_counts[genre_counts >= 2].index
        genre_df = genre_df[genre_df[GENRE_COL].isin(valid_genres)].copy()
        if genre_df[GENRE_COL].nunique() >= 2 and len(genre_df) >= 20:
            Xg = genre_df[numeric_features]
            le = LabelEncoder()
            yg = le.fit_transform(genre_df[GENRE_COL].astype(str))
            Xg_tr, Xg_te, yg_tr, yg_te = train_test_split(
                Xg, yg, test_size=0.30, random_state=42, stratify=yg
            )
            genre_models = {
                "Logistic Reg": Pipeline([
                    ("imp", SimpleImputer(strategy="median")),
                    ("sc", StandardScaler()),
                    ("model", LogisticRegression(max_iter=1500, random_state=42))
                ]),
                "Decision Tree": Pipeline([
                    ("imp", SimpleImputer(strategy="median")),
                    ("model", DecisionTreeClassifier(max_depth=8, random_state=42))
                ]),
                "Random Forest": Pipeline([
                    ("imp", SimpleImputer(strategy="median")),
                    ("model", RandomForestClassifier(n_estimators=140, random_state=42, n_jobs=-1))
                ]),
                "SVM": Pipeline([
                    ("imp", SimpleImputer(strategy="median")),
                    ("sc", StandardScaler()),
                    ("model", SVC(kernel="rbf", C=1.0, gamma="scale", probability=False, random_state=42))
                ]),
            }
            for name, gm in genre_models.items():
                gm.fit(Xg_tr, yg_tr)
                gp = gm.predict(Xg_te)
                genre_results[name] = {
                    "Accuracy": float(accuracy_score(yg_te, gp)),
                    "Precision": float(precision_score(yg_te, gp, average="weighted", zero_division=0)),
                    "Recall": float(recall_score(yg_te, gp, average="weighted", zero_division=0)),
                    "F1": float(classification_report(yg_te, gp, output_dict=True, zero_division=0)["weighted avg"]["f1-score"]),
                    "pred": gp,
                }
                fitted_genre_models[name] = gm
            best_genre_name = max(genre_results, key=lambda n: genre_results[n]["F1"])
            genre_payload = {
                "available": True,
                "results": genre_results,
                "models": fitted_genre_models,
                "best_name": best_genre_name,
                "y_test": yg_te,
                "labels": le.classes_,
            }
    except Exception as e:
        genre_payload = {"available": False, "error": str(e)}

    # Random Forest used for feature importance and recommendation popularity signal
    feature_model = fitted_reg_models.get("Random Forest Regressor")

    return {
        "best_reg_model": best_reg_model,
        "best_reg_name": best_reg_name,
        "regression_results": regression_results,
        "regression_models": fitted_reg_models,
        "feature_model": feature_model,
        "pref_model": fitted_clf_models["Logistic Reg"],
        "classification_results": classification_results,
        "classification_models": fitted_clf_models,
        "genre_classification": genre_payload,
        "X_te": X_te,
        "y_te": y_te,
        "lin_pred": regression_results["Linear Regression"]["pred"],
        "poly_pred": regression_results["Polynomial Regression"]["pred"],
        "lin_r2": regression_results["Linear Regression"]["R2"],
        "poly_r2": regression_results["Polynomial Regression"]["R2"],
        "Xc_te": Xc_te,
        "yc_te": yc_te,
    }

# ============================================================
# USER LOGIC
# ============================================================
def get_user_preferred_genres(uid):
    prefs = load_user_preferences(); r = prefs[prefs["user_id"]==uid]
    return [] if r.empty else split_genres(r.iloc[0]["preferred_genres"])

def has_user_preferences(uid):
    return len(get_user_preferred_genres(uid)) > 0

def save_preferred_genres(uid, genres):
    prefs = load_user_preferences(); text = ";".join(genres)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    idx = prefs.index[prefs["user_id"]==uid].tolist()
    if idx:
        prefs.loc[idx[0],"preferred_genres"] = text; prefs.loc[idx[0],"updated_at"] = now
    else:
        prefs = pd.concat([prefs, pd.DataFrame([{"user_id":uid,"preferred_genres":text,"created_at":now,"updated_at":now}])], ignore_index=True)
    save_user_preferences(prefs)

def create_or_login_user(username, email):
    users = load_users(); username, email = username.strip(), email.strip().lower()
    if not username or not email: return {"success":False,"message":"Username and email are required."}
    existing = users[users["email"].astype(str).str.lower()==email]
    if not existing.empty:
        r = existing.iloc[0]
        return {"success":True,"message":"Welcome back!","user_id":int(r["user_id"]),"username":r["username"],"email":r["email"]}
    new_id = 1 if users.empty else int(users["user_id"].max())+1
    users = pd.concat([users, pd.DataFrame([{"user_id":new_id,"username":username,"email":email,"created_at":datetime.now().strftime("%Y-%m-%d %H:%M:%S")}])], ignore_index=True)
    save_users(users)
    return {"success":True,"message":"New account created!","user_id":new_id,"username":username,"email":email}

def logout_user():
    for k in ["logged_in","user_id","username","email"]:
        st.session_state[k] = None if k!="logged_in" else False
    for k in ["nav_page", "latest_recs", "latest_recs_tp", "latest_recs_time_genres", "latest_recs_pref_genres"]:
        st.session_state.pop(k, None)
    st.session_state["page"] = "Welcome"

def add_listening_history(uid, row, mood=None):
    history = load_history(); tp = get_time_period(); mood = mood or get_default_mood(tp)
    new_row = pd.DataFrame([{"history_id":next_numeric_id(history,"history_id"),"user_id":uid,"title":row[TRACK_COL],"artist":row[ARTIST_COL],"top_genre":row[GENRE_COL],"time_period":tp,"mood":mood,"listened_at":datetime.now().strftime("%Y-%m-%d %H:%M:%S")}])
    save_history(pd.concat([history, new_row], ignore_index=True))

def add_rating(uid, row, rating):
    ratings = load_ratings(); tp = get_time_period(); liked = 1 if int(rating)>=4 else 0
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    title, artist, genre = row[TRACK_COL], row[ARTIST_COL], row[GENRE_COL]
    existing = ratings[(ratings["user_id"].astype(int)==int(uid))&(ratings["title"].astype(str)==str(title))&(ratings["artist"].astype(str)==str(artist))]
    if not existing.empty:
        idx = existing.index[-1]
        ratings.loc[idx, ["rating","liked","time_period","rated_at","top_genre"]] = [int(rating),liked,tp,now,genre]
        ratings = ratings.drop(index=existing.index[:-1])
    else:
        new_id = 1 if ratings.empty else int(pd.to_numeric(ratings["rating_id"],errors="coerce").max())+1
        ratings = pd.concat([ratings, pd.DataFrame([{"rating_id":new_id,"user_id":uid,"title":title,"artist":artist,"top_genre":genre,"rating":int(rating),"liked":liked,"time_period":tp,"rated_at":now}])], ignore_index=True)
    save_ratings(ratings)

def get_top_user_genres_for_time(uid, tp, fallback=None):
    history, ratings, fallback = load_history(), load_ratings(), fallback or []
    h = history[(history["user_id"]==uid)&(history["time_period"]==tp)]
    r = ratings[(ratings["user_id"]==uid)&(ratings["time_period"]==tp)]
    frames = []
    if not h.empty: frames.append(h[[GENRE_COL]].rename(columns={GENRE_COL:"genre"}))
    if not r.empty:
        weighted = r.copy(); weighted["weight"] = np.where(weighted["liked"]==1,2,1)
        exp = []
        for _,row in weighted.iterrows():
            for _ in range(int(row["weight"])): exp.append({"genre":row[GENRE_COL]})
        if exp: frames.append(pd.DataFrame(exp))
    if frames:
        combined = pd.concat(frames, ignore_index=True)
        counts = combined["genre"].value_counts()
        if not counts.empty: return counts.head(3).index.tolist()
    return fallback[:3]

def build_user_profile(uid, songs_df, numeric_features, tp):
    history, ratings = load_history(), load_ratings()
    ur = ratings[(ratings["user_id"]==uid)&(ratings["liked"]==1)]
    sh = history[(history["user_id"]==uid)&(history["time_period"]==tp)]
    rows = []
    if not ur.empty: rows.append(ur[[TRACK_COL,ARTIST_COL]])
    if not sh.empty: rows.append(sh[[TRACK_COL,ARTIST_COL]])
    if not rows: return None
    keys = pd.concat(rows, ignore_index=True).drop_duplicates()
    liked = songs_df.merge(keys, on=[TRACK_COL,ARTIST_COL], how="inner")
    return None if liked.empty else liked[numeric_features].mean()

def get_time_feature_score(row, tp):
    bpm_s = min(float(row.get("bpm",0))/200, 1)*100
    if tp=="Morning":   s = 0.30*row.get("acous",0)+0.25*(100-row.get("nrgy",0))+0.25*row.get("val",0)+0.20*row.get("dnce",0)
    elif tp=="Afternoon": s = 0.30*row.get("dnce",0)+0.25*row.get("nrgy",0)+0.20*row.get("val",0)+0.15*bpm_s+0.10*(100-row.get("acous",0))
    elif tp=="Evening":   s = 0.35*row.get("acous",0)+0.25*(100-row.get("nrgy",0))+0.20*row.get("val",0)+0.20*(100-bpm_s)
    else:                 s = 0.35*row.get("nrgy",0)+0.30*row.get("dnce",0)+0.20*bpm_s+0.15*row.get("val",0)
    return max(0,min(100,s))/100

def create_recommendation_reason(row, tp, user_genres, has_profile):
    reasons = []
    if row["genre_match"]==1: reasons.append(f"matches your preferred genre for {tp.lower()}")
    if row["time_genre_match"]==1: reasons.append(f"you often listen to {row[GENRE_COL]} during {tp.lower()}")
    if has_profile and row["user_similarity"]>=0.75: reasons.append("audio features closely match songs you've liked")
    if row["preference_probability"]>=0.70: reasons.append("classification model predicts high preference probability")
    if row["predicted_pop"]>=70: reasons.append("regression model predicts strong popularity")
    if row.get("dnce",0)>=70: reasons.append("high danceability score")
    if row.get("nrgy",0)>=75: reasons.append("high energy track")
    if row.get("acous",0)>=45 and tp in ["Morning","Evening"]: reasons.append("acoustic feel suits this time of day")
    if row.get("val",0)>=60: reasons.append("positive valence/mood")
    if row["time_score"]>=0.70: reasons.append(f"audio profile suits {tp.lower()} listening")
    if not reasons: reasons.append("balanced recommendation from popularity, features, and time context")
    return "Recommended because " + "; ".join(reasons) + "."

def log_recommendations(uid, recs, tp):
    logs = load_recommendation_logs(); start = next_numeric_id(logs,"log_id")
    rows = [{"log_id":start+i,"user_id":uid,"title":row[TRACK_COL],"artist":row[ARTIST_COL],"top_genre":row[GENRE_COL],"time_period":tp,"final_score":round(float(row["final_score"]),4),"reason":row["recommendation_reason"],"recommended_at":datetime.now().strftime("%Y-%m-%d %H:%M:%S")} for i,(_,row) in enumerate(recs.iterrows())]
    if rows: save_recommendation_logs(pd.concat([logs, pd.DataFrame(rows)], ignore_index=True))

def recommend_for_user(uid, songs_df, numeric_features, models, top_n=10):
    temp = songs_df.copy()
    tp = get_time_period()

    pref_genres = get_user_preferred_genres(uid)
    time_genres = get_top_user_genres_for_time(uid, tp, fallback=pref_genres)
    all_genres = list(dict.fromkeys(time_genres + pref_genres))

    profile = build_user_profile(uid, songs_df, numeric_features, tp)
    has_profile = profile is not None

    fm = temp[numeric_features].fillna(temp[numeric_features].median())

    if has_profile:
        sc = StandardScaler()
        sf = sc.fit_transform(fm)
        sp = sc.transform(pd.DataFrame([profile], columns=numeric_features))
        temp["user_similarity"] = (cosine_similarity(sf, sp).flatten() + 1) / 2
    else:
        temp["user_similarity"] = 0.50

    temp["predicted_pop"] = models["best_reg_model"].predict(temp[numeric_features]).clip(0, 100)
    temp["preference_probability"] = models["pref_model"].predict_proba(temp[numeric_features])[:, 1]

    temp["genre_match"] = temp[GENRE_COL].isin(pref_genres).astype(int)
    temp["time_genre_match"] = temp[GENRE_COL].isin(time_genres).astype(int)
    temp["time_score"] = temp.apply(lambda r: get_time_feature_score(r, tp), axis=1)

    # Collaborative filtering signals
    # Frontend-safe approximation aligned with analytics scoring structure
    temp["cf_user_match"] = temp["user_similarity"]
    temp["cf_item_match"] = temp["genre_match"]

    history, ratings = load_history(), load_ratings()

    listened = history[history["user_id"] == uid][[TRACK_COL, ARTIST_COL]]
    rated = ratings[ratings["user_id"] == uid][[TRACK_COL, ARTIST_COL]]

    seen_df = pd.concat([listened, rated], ignore_index=True).drop_duplicates()
    seen = set(zip(seen_df[TRACK_COL], seen_df[ARTIST_COL]))

    if seen:
        temp = temp[~temp.apply(lambda r: (r[TRACK_COL], r[ARTIST_COL]) in seen, axis=1)]

    # Final weighted recommendation score
    temp["final_score"] = (
        0.20 * temp["user_similarity"] +
        0.18 * temp["preference_probability"] +
        0.16 * (temp["predicted_pop"] / 100) +
        0.14 * temp["time_score"] +
        0.12 * temp["cf_user_match"] +
        0.10 * temp["cf_item_match"] +
        0.06 * temp["time_genre_match"] +
        0.04 * temp["genre_match"]
    )

    temp["recommendation_reason"] = temp.apply(
        lambda r: create_recommendation_reason(r, tp, all_genres, has_profile),
        axis=1
    )

    recs = temp.sort_values("final_score", ascending=False).head(top_n)

    log_recommendations(uid, recs, tp)

    return recs, tp, time_genres, pref_genres

# ============================================================
# PLOTLY CHART BUILDERS
# ============================================================

def make_bar_chart(x, y, title, xlabel="", ylabel="", color=None, horizontal=False, height=400):
    color = color or CHART_COLORS[0]
    if horizontal:
        fig = go.Figure(go.Bar(
            x=y, y=x, orientation="h",
            marker=dict(color=color if isinstance(color, list) else color,
                        line=dict(width=0)),
            hovertemplate="<b>%{y}</b><br>Value: %{x}<extra></extra>",
        ))
    else:
        fig = go.Figure(go.Bar(
            x=x, y=y,
            marker=dict(color=color if isinstance(color, list) else color,
                        line=dict(width=0)),
            hovertemplate="<b>%{x}</b><br>Value: %{y}<extra></extra>",
        ))
    apply_layout(fig, title=title, height=height)
    fig.update_layout(
        xaxis_title=xlabel, yaxis_title=ylabel,
        showlegend=False,
        bargap=0.35,
    )
    return fig


def make_donut_chart(labels, values, title, height=380):
    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        hole=0.55,
        marker=dict(colors=get_adaptive_colors(labels, mode="time"), line=dict(color=PALETTE["bg"], width=2)),
        textinfo="label+percent",
        textfont=dict(size=14, color="#FFFFFF", family="'DM Sans', sans-serif"),
        hovertemplate="<b>%{label}</b><br>%{value} (%{percent})<extra></extra>",
        insidetextorientation="radial",
    ))
    apply_layout(fig, title=title, height=height)
    fig.update_layout(showlegend=True, legend=dict(orientation="v", x=1.05))
    return fig


def make_scatter_chart(x, y, title, xlabel="", ylabel="", color=None, trendline=False, height=420):
    traces = [go.Scatter(
        x=x, y=y, mode="markers", name="Songs",
        marker=dict(color=color or CHART_COLORS[0], size=5, opacity=1,
                    line=dict(width=0)),
        hovertemplate=f"{xlabel}: %{{x:.1f}}<br>{ylabel}: %{{y:.1f}}<extra></extra>",
    )]
    if trendline:
        valid = (~np.isnan(np.array(x, float))) & (~np.isnan(np.array(y, float)))
        xv, yv = np.array(x, float)[valid], np.array(y, float)[valid]
        if len(xv) > 2:
            z = np.polyfit(xv, yv, 1); p = np.poly1d(z)
            xl = np.linspace(xv.min(), xv.max(), 80)
            traces.append(go.Scatter(
                x=xl, y=p(xl), mode="lines",
                line=dict(color=PALETTE["accent2"], width=2, dash="dash"),
                name="Trend", hoverinfo="skip",
            ))
    fig = go.Figure(traces)
    apply_layout(fig, title=title, height=height)
    fig.update_layout(xaxis_title=xlabel, yaxis_title=ylabel)
    return fig


def make_histogram(data, title, xlabel="", color=None, height=380):
    fig = go.Figure(go.Histogram(
        x=data, nbinsx=28,
        marker=dict(color=color or CHART_COLORS[0], line=dict(color=PALETTE["bg"], width=0.5)),
        hovertemplate="Range: %{x}<br>Count: %{y}<extra></extra>",
        opacity=0.9,
    ))
    apply_layout(fig, title=title, height=height)
    fig.update_layout(xaxis_title=xlabel, yaxis_title="Count", showlegend=False, bargap=0.05)
    mean_val = np.nanmean(data)
    fig.add_vline(x=mean_val, line=dict(color=PALETTE["accent2"], width=2, dash="dot"),
                  annotation_text=f"Mean {mean_val:.1f}", annotation_font=dict(color=PALETTE["accent2"], size=10))
    return fig


def make_heatmap(matrix, xlabels, ylabels, title, height=720):  # 🔥 bigger

    colorscale = [
        [0.00, "#E85D75"],
        [0.50, "#151A26"],
        [1.00, "#7C6FFF"],
    ]

    percent_text = [[f"{v * 100:.0f}%" for v in row] for row in matrix]

    fig = go.Figure(go.Heatmap(
        z=matrix,
        x=xlabels,
        y=ylabels,
        text=percent_text,
        texttemplate="%{text}",
        textfont=dict(size=11, color="#FFFFFF"),
        colorscale=colorscale,
        zmin=-1,
        zmax=1,
        zmid=0,
        colorbar=dict(
            title=dict(text="Correlation (%)", font=dict(color="#FFFFFF", size=13)),
            tickvals=[-1, -0.5, 0, 0.5, 1],
            ticktext=["-100%", "-50%", "0%", "50%", "100%"],
            tickfont=dict(color="#FFFFFF", size=11),
            thickness=20,
            len=0.9
        ),
    ))

    apply_layout(fig, title=title, height=height)

    fig.update_layout(
        autosize=True,
        margin=dict(l=200, r=140, t=80, b=200),  # 🔥 more breathing space
        xaxis=dict(
            tickangle=-45,
            tickfont=dict(size=12),
            automargin=True
        ),
        yaxis=dict(
            autorange="reversed",
            tickfont=dict(size=12),
            automargin=True
        ),
    )

    return fig


def make_roc_chart(fpr_list, tpr_list, auc_list, labels, title, height=420):
    fig = go.Figure()
    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                  line=dict(color=PALETTE["muted"], width=1.5, dash="dash"))
    for fpr, tpr, auc_val, label, color in zip(fpr_list, tpr_list, auc_list, labels, CHART_COLORS):
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines",
            name=f"{label} (AUC={auc_val:.3f})",
            line=dict(color=color, width=2.5),
            hovertemplate="FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>",
            fill="tozeroy" if len(fpr_list)==1 else None,
            fillcolor="rgba(124,111,255,0.08)" if len(fpr_list)==1 else None,
        ))
    apply_layout(fig, title=title, height=height)
    fig.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
                      xaxis_range=[-0.02,1.02], yaxis_range=[-0.02,1.02])
    return fig


def make_confusion_matrix_chart(cm, title, height=380):
    labels = ["Not Like", "Like"]
    colorscale = [[0, PALETTE["surface"]], [1, PALETTE["accent1"]]]
    fig = go.Figure(go.Heatmap(
        z=cm, x=labels, y=labels,
        colorscale=colorscale,
        text=[[f"{cm[i][j]:,}" for j in range(2)] for i in range(2)],
        texttemplate="<b>%{text}</b>",
        textfont=dict(size=22, color=PALETTE["text"]),
        showscale=True,
        hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
    ))
    apply_layout(fig, title=title, height=height)
    fig.update_layout(xaxis_title="Predicted", yaxis_title="Actual", yaxis=dict(autorange="reversed"))
    return fig


def make_multi_bar_comparison(categories, series_dict, title, height=420):
    fig = go.Figure()
    for i, (name, vals) in enumerate(series_dict.items()):
        fig.add_trace(go.Bar(
            name=name, x=categories, y=vals,
            marker=dict(color=CHART_COLORS[i], line=dict(width=0)),
            hovertemplate=f"<b>{name}</b><br>%{{x}}: %{{y:.3f}}<extra></extra>",
        ))
    apply_layout(fig, title=title, height=height)
    fig.update_layout(barmode="group", bargap=0.2, bargroupgap=0.05, yaxis_range=[0, 1.15])
    return fig


def make_feature_importance_bar(features, importances, title, height=420):
    idx = np.argsort(importances)
    fig = go.Figure(go.Bar(
        x=importances[idx], y=[features[i] for i in idx],
        orientation="h",
        marker=dict(
            color=importances[idx],
            colorscale=[[0, PALETTE["accent1"]], [1, PALETTE["accent3"]]],
            showscale=False,
            line=dict(width=0),
        ),
        hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
    ))
    apply_layout(fig, title=title, height=height)
    fig.update_layout(xaxis_title="Relative Importance", showlegend=False)
    return fig


def make_residual_chart(predicted, residuals, title, height=380):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=predicted, y=residuals, mode="markers",
        marker=dict(color=PALETTE["accent1"], size=5, opacity=0.5, line=dict(width=0)),
        hovertemplate="Predicted: %{x:.1f}<br>Residual: %{y:.1f}<extra></extra>",
        name="Residuals",
    ))
    fig.add_hline(y=0, line=dict(color=PALETTE["accent2"], width=2, dash="dot"))
    apply_layout(fig, title=title, height=height)
    fig.update_layout(xaxis_title="Predicted Popularity", yaxis_title="Residual", showlegend=False)
    return fig

# ============================================================
# PAGES
# ============================================================
def welcome_page():
    render_header("🎵 Predictive Music Recommendation", "Powered by ML — personalized recommendations that explain themselves")
    left, right = st.columns([1.2, 0.8])
    with left:
        st.markdown("""
        <div class="hero">
            <div class="hero-title">Welcome to MoodMix PDA 👋</div>
            <div class="hero-sub">A full predictive analytics system that learns from your listening patterns,
            ratings, and time-of-day behavior to recommend music — with clear explanations for every suggestion.</div>
        </div>""", unsafe_allow_html=True)
        section_heading("What this system does", "⚡")
        cols = st.columns(2)
        features = [
            ("🎛️","Multi-model ML","Linear, Polynomial, Logistic Regression + Random Forest for popularity & preference prediction"),
            ("⏰","Time-aware","Detects Morning / Afternoon / Evening / Night and adjusts recommendations accordingly"),
            ("📊","Explainable AI","Every recommendation comes with a clear, readable reason driven by model outputs"),
            ("📁","CSV backend","Stores real user history, ratings, logs — no database required"),
        ]
        for i, (icon, title, desc) in enumerate(features):
            with cols[i%2]:
                st.markdown(f"""<div class="glass-card"><div style="font-size:1.4rem;margin-bottom:6px">{icon}</div>
                <div style="font-family:'Sora',sans-serif;font-weight:700;font-size:0.95rem;color:#E2E8F8;margin-bottom:4px">{title}</div>
                <div style="font-size:0.84rem;color:#C9D1FF;line-height:1.45">{desc}</div></div>""", unsafe_allow_html=True)
    with right:
        st.markdown("""<div class="glass-card"><div style="font-family:'Sora',sans-serif;font-weight:700;font-size:1.1rem;margin-bottom:16px">🔐 Login or Sign Up</div></div>""", unsafe_allow_html=True)
        with st.form("login_form"):
            username = st.text_input("Username", placeholder="e.g. music_lover")
            email    = st.text_input("Email", placeholder="you@example.com")
            submitted = st.form_submit_button("Continue →", use_container_width=True)
        if submitted:
            result = create_or_login_user(username, email)
            if result["success"]:
                st.session_state.update({"logged_in":True,"user_id":result["user_id"],"username":result["username"],"email":result["email"],"page":"Dashboard","nav_page":"Dashboard"})
                st.success(result["message"]); st.rerun()
            else: st.error(result["message"])


def preference_setup_page(df):
    render_header("🎛️ Genre Preferences", "Tell us what you like — we'll personalize everything from here")
    genres = sorted(df[GENRE_COL].dropna().astype(str).unique().tolist())
    current = get_user_preferred_genres(st.session_state["user_id"])
    selected = st.multiselect("Choose one or more genres you prefer", options=genres, default=current)
    if st.button("Save Preferences & Continue →"):
        if not selected: st.warning("Please select at least one genre.")
        else:
            save_preferred_genres(st.session_state["user_id"], selected)
            st.success("Preferences saved!"); st.session_state["page"] = "Dashboard"; st.rerun()


def dashboard_page(df):
    uid = st.session_state["user_id"]; tp = get_time_period()
    render_header(f"Hello, {st.session_state['username']} 👋", f"Current listening period: {tp}")
    history = load_history(); ratings = load_ratings(); logs = load_recommendation_logs(); users = load_users()
    uh = history[history["user_id"]==uid]; ur = ratings[ratings["user_id"]==uid]; ul = logs[logs["user_id"]==uid]
    render_metrics(
        metric_card("Total Songs",f"{len(df):,}"), metric_card("Your Listens",f"{len(uh):,}"),
        metric_card("Your Ratings",f"{len(ur):,}"), metric_card("Recommendations",f"{len(ul):,}"),
        metric_card("Registered Users",f"{users['user_id'].nunique():,}"),
    )

    # Music-app style recommendation previews. Detailed graphs remain below and in Analytics.
    pref_genres = get_user_preferred_genres(uid)
    if pref_genres:
        liked_preview = df[df[GENRE_COL].isin(pref_genres)].sort_values(POPULARITY_COL, ascending=False).head(4)
        render_music_card_grid(
            liked_preview,
            f"Because you liked {pref_genres[0]}",
            "Personalized picks based on your saved genre preferences.",
            reason_prefix="Because this matches your preferred genre and has strong popularity signals."
        )

    time_preview = df.copy()
    time_preview["_time_score"] = time_preview.apply(lambda r: get_time_feature_score(r, tp), axis=1)
    time_preview = time_preview.sort_values(["_time_score", POPULARITY_COL], ascending=False).head(4)
    render_music_card_grid(
        time_preview,
        f"{tp} Session Picks",
        "Songs selected using time-of-day behavior, energy, danceability, and mood fit.",
        reason_prefix=f"Because its audio profile fits your {tp.lower()} listening session."
    )

    c1, c2 = st.columns(2)
    with c1:
        section_heading("Your Preferred Genres","🎼")
        genres = get_user_preferred_genres(uid)
        if genres: st.markdown("".join(f'<span class="pill">{sx(g)}</span>' for g in genres), unsafe_allow_html=True)
        else: st.info("No preferences saved yet.")
        section_heading("Listening Mood Distribution","🌙")
        if not uh.empty and "mood" in uh.columns:
            mood_counts = uh["mood"].value_counts()
            fig = make_bar_chart(mood_counts.index.tolist(), mood_counts.values.tolist(),
                                 "Listens by Mood", ylabel="Count",
                                 color=get_adaptive_colors(mood_counts.index.tolist(), mode="mood"),
                                 horizontal=True, height=340)
            show_chart(fig, "dash_mood",
                "This horizontal bar chart shows how your listening activity breaks down by mood. "
                "Taller bars mean you listen more in that emotional state. Use this to understand your music personality.")
        else: st.info("No listening history yet.")
    with c2:
        section_heading("Listening Pattern by Time","⏰")
        if not uh.empty:
            order = ["Morning","Afternoon","Evening","Night"]
            tc = uh["time_period"].value_counts().reindex(order).fillna(0)
            tc = tc[tc > 0]
            if tc.empty: st.info("No listening history yet.")
            else:
                fig = make_donut_chart(tc.index.tolist(), tc.values.tolist(), "Time Period Distribution", height=340)
                show_chart(fig, "dash_time",
                    "This donut chart reveals when you listen to music most. "
                    "The system uses this data to auto-detect your session period and adjust recommendations accordingly.")
        else: st.info("No listening history yet.")
        section_heading("Top Listened Genres","🎵")
        if not uh.empty and GENRE_COL in uh.columns:
            gc = uh[GENRE_COL].value_counts().head(6)
            fig = make_bar_chart(gc.index.tolist(), gc.values.tolist(), "Your Top Genres", ylabel="Listens", height=320)
            show_chart(fig, "dash_genres",
                "Top genres by total listen count from your personal history. "
                "The recommendation engine weights these genres more heavily when building your playlist.")
        else: st.info("No genre data yet.")


def explore_songs_page(df):
    render_header("🔍 Explore Songs", "Browse, listen, and rate songs from the dataset")
    c1, c2 = st.columns([2,1])
    with c1:
        search = st.text_input("Search by song or artist", placeholder="Type to search…")
    with c2:
        genre_filter = st.multiselect("Filter by genre", sorted(df[GENRE_COL].dropna().astype(str).unique().tolist()))

    filtered = df.copy()
    if search:
        s = search.lower()
        filtered = filtered[
            filtered[TRACK_COL].astype(str).str.lower().str.contains(s, na=False) |
            filtered[ARTIST_COL].astype(str).str.lower().str.contains(s, na=False)
        ]
    if genre_filter:
        filtered = filtered[filtered[GENRE_COL].isin(genre_filter)]

    # Pagination replaces the old broken "Show" slider.
    PAGE_SIZE = 30
    filter_signature = f"{search}|{','.join(sorted(map(str, genre_filter)))}|{len(filtered)}"
    if st.session_state.get("explore_filter_signature") != filter_signature:
        st.session_state["explore_filter_signature"] = filter_signature
        st.session_state["explore_page_num"] = 1

    total_songs = len(filtered)
    total_pages = max(1, math.ceil(total_songs / PAGE_SIZE))
    st.session_state["explore_page_num"] = min(max(1, st.session_state.get("explore_page_num", 1)), total_pages)
    start_idx = (st.session_state["explore_page_num"] - 1) * PAGE_SIZE
    end_idx = start_idx + PAGE_SIZE
    page_df = filtered.iloc[start_idx:end_idx]

    if total_songs:
        st.caption(
            f"Showing songs **{start_idx + 1}-{min(end_idx, total_songs)}** of **{total_songs:,}** "
            f"• Page **{st.session_state['explore_page_num']}** of **{total_pages}** "
            f"• current period: **{get_time_period()}**"
        )
    else:
        st.warning("No songs match your search/filter.")

    for idx, row in page_df.iterrows():
        safe_key = f"{idx}_{abs(hash(str(row[TRACK_COL]) + str(row[ARTIST_COL])))}"
        render_song_card(row[TRACK_COL], row[ARTIST_COL],
            [f"Genre: {row[GENRE_COL]}", f"Pop: {row.get(POPULARITY_COL,'N/A')}", f"Energy: {row.get('nrgy','N/A')} | Dance: {row.get('dnce','N/A')} | BPM: {row.get('bpm','N/A')}"])
        cc1, cc2, cc3 = st.columns([1,1,1.2])
        with cc1:
            mood = st.selectbox("Mood", ["Fresh","Focus","Relax","Happy","Party","Sad","Balanced"],
                index=["Fresh","Focus","Relax","Happy","Party","Sad","Balanced"].index(get_default_mood(get_time_period())), key=f"mood_{safe_key}")
        with cc2:
            rating = st.slider("Rate", 1, 5, 3, key=f"rating_{safe_key}")
        with cc3:
            st.write("")
            if st.button("🎧 Listened", key=f"listen_{safe_key}", use_container_width=True):
                add_listening_history(st.session_state["user_id"], row, mood)
                st.success("Saved to My History!")
                st.rerun()
            if st.button("⭐ Rate", key=f"rate_{safe_key}", use_container_width=True):
                add_rating(st.session_state["user_id"], row, rating)
                st.success("Rating saved to My History!")
                st.rerun()
        st.markdown('<div style="border-top:1px solid var(--border);margin:10px 0"></div>', unsafe_allow_html=True)

    if total_pages > 1:
        col_prev, col_page, col_next = st.columns([1, 2, 1])
        with col_prev:
            if st.button("⬅ Previous", disabled=st.session_state["explore_page_num"] <= 1, use_container_width=True):
                st.session_state["explore_page_num"] -= 1
                st.rerun()
        with col_page:
            st.markdown(
                f"<div style='text-align:center;padding-top:8px;color:var(--text);font-weight:700;'>Page {st.session_state['explore_page_num']} of {total_pages}</div>",
                unsafe_allow_html=True
            )
        with col_next:
            if st.button("Next ➡", disabled=st.session_state["explore_page_num"] >= total_pages, use_container_width=True):
                st.session_state["explore_page_num"] += 1
                st.rerun()

def my_history_page():
    render_header("📋 My History", "Your listening history, ratings, and recommendation logs")
    uid = st.session_state["user_id"]
    uh = load_history(); ur = load_ratings(); ul = load_recommendation_logs()
    uh = uh[uh["user_id"]==uid]; ur = ur[ur["user_id"]==uid]; ul = ul[ul["user_id"]==uid]
    tab1, tab2, tab3 = st.tabs(["🎧 Listening History", "⭐ Ratings", "💡 Recommendation Logs"])
    with tab1:
        if uh.empty: st.info("No listening history yet.")
        else: st.dataframe(uh.sort_values("listened_at",ascending=False), use_container_width=True)
    with tab2:
        if ur.empty: st.info("No ratings yet.")
        else: st.dataframe(ur.sort_values("rated_at",ascending=False), use_container_width=True)
    with tab3:
        if ul.empty: st.info("No recommendation logs yet.")
        else: st.dataframe(ul.sort_values("recommended_at",ascending=False), use_container_width=True)


def recommendations_page(df, numeric_features, models):
    render_header("✨ Personalized Recommendations", "ML-powered songs chosen for you right now")
    uid = st.session_state["user_id"]
    c1, c2 = st.columns([3,1])
    with c2:
        top_n = st.slider("# Recommendations", 5, 20, 10)

    if st.button("🎲 Generate Recommendations"):
        with st.spinner("Building your personalized playlist…"):
            recs, tp, time_genres, pref_genres = recommend_for_user(uid, df, numeric_features, models, top_n=top_n)
        st.session_state["latest_recs"] = recs.to_dict("records")
        st.session_state["latest_recs_tp"] = tp
        st.session_state["latest_recs_time_genres"] = time_genres
        st.session_state["latest_recs_pref_genres"] = pref_genres
        st.success(f"Generated **{len(recs)}** recommendations for your **{tp}** session.")

    if "latest_recs" not in st.session_state or not st.session_state["latest_recs"]:
        st.info("Click the button above to generate your personalized playlist.")
        return

    recs = pd.DataFrame(st.session_state["latest_recs"])
    tp = st.session_state.get("latest_recs_tp", get_time_period())
    time_genres = st.session_state.get("latest_recs_time_genres", [])
    pref_genres = st.session_state.get("latest_recs_pref_genres", [])

    section_heading("Genres Used", "🎼")
    if time_genres:
        st.markdown("**Time-period genres:** " + ", ".join(f'<span class="pill">{sx(g)}</span>' for g in time_genres), unsafe_allow_html=True)
    elif pref_genres:
        st.markdown("**Preferred genres:** " + ", ".join(f'<span class="pill">{sx(g)}</span>' for g in pref_genres), unsafe_allow_html=True)
    st.markdown("""<div class="formula-box">Final Score =
0.20 × User Similarity
+ 0.18 × Preference Probability
+ 0.16 × Predicted Popularity
+ 0.14 × Time Score
+ 0.12 × CF User Match
+ 0.10 × CF Item Match
+ 0.06 × Time Genre Match
+ 0.04 × Preferred Genre Match</div>""", unsafe_allow_html=True)

    section_heading("Your Playlist", "🎵")
    st.caption("Use 🎧 Played or ⭐ Rate here")
    for idx, row in recs.iterrows():
        safe_key = f"rec_{idx}_{abs(hash(str(row[TRACK_COL]) + str(row[ARTIST_COL])))}"
        render_song_card(row[TRACK_COL], row[ARTIST_COL],
            [f"Genre: {row[GENRE_COL]}", f"Score: {row['final_score']:.3f}", f"Similarity: {row['user_similarity']:.3f}", f"Pref Prob: {row['preference_probability']:.3f}", f"Pred Pop: {row['predicted_pop']:.1f}"],
            reason=row["recommendation_reason"], score=row["final_score"])

        cc1, cc2, cc3 = st.columns([1, 1, 1.2])
        with cc1:
            mood = st.selectbox("Mood", ["Fresh","Focus","Relax","Happy","Party","Sad","Balanced"],
                index=["Fresh","Focus","Relax","Happy","Party","Sad","Balanced"].index(get_default_mood(get_time_period())),
                key=f"rec_mood_{safe_key}")
        with cc2:
            rating = st.slider("Rate", 1, 5, 3, key=f"rec_rating_{safe_key}")
        with cc3:
            st.write("")
            if st.button("🎧 Played", key=f"rec_played_{safe_key}", use_container_width=True):
                add_listening_history(uid, row, mood)
                st.success("Added to My History!")
                st.rerun()
            if st.button("⭐ Save Rating", key=f"rec_rate_{safe_key}", use_container_width=True):
                add_rating(uid, row, rating)
                st.success("Rating updated in My History!")
                st.rerun()
        st.markdown('<div style="border-top:1px solid var(--border);margin:10px 0"></div>', unsafe_allow_html=True)

# ANALYTICS PAGE — Professional Plotly Charts + AI Bot
# ============================================================
def analytics_page(df, numeric_features, models):
    render_header("📊 Analytics Dashboard", "Comprehensive EDA, model evaluation, and system insights")
    history = load_history(); ratings = load_ratings()
    users = load_users(); prefs = load_user_preferences(); logs = load_recommendation_logs()
    render_metrics(
        metric_card("Registered Users", f"{users['user_id'].nunique():,}" if not users.empty else "0"),
        metric_card("Listening Records", f"{len(history):,}"),
        metric_card("Ratings", f"{len(ratings):,}"),
        metric_card("Songs in Dataset", f"{len(df):,}"),
        metric_card("Unique Genres", f"{df[GENRE_COL].nunique()}"),
        metric_card("Recommendation Logs", f"{len(logs):,}"),
    )

    tabs = st.tabs(["🔬 EDA","📈 Regression","🎯 Classification","🏆 Feature Importance","👤 User Behavior","💡 Recommendation System"])

    # ── TAB 1: EDA ──────────────────────────────────────────
    with tabs[0]:
        st.markdown("### 🔬 Exploratory Data Analysis")

        with st.expander("Q1 · Dataset overview", expanded=True):
            st.caption("**Answer:** Overview of song count, genre diversity, artists, and average popularity.")
            render_metrics(
                metric_card("Songs", f"{len(df):,}"), metric_card("Genres", str(df[GENRE_COL].nunique())),
                metric_card("Artists", str(df[ARTIST_COL].nunique())), metric_card("Avg Popularity", f"{df[POPULARITY_COL].mean():.1f}"),
            )
            st.dataframe(df[[TRACK_COL,ARTIST_COL,GENRE_COL,POPULARITY_COL]+numeric_features].head(8), use_container_width=True)

        with st.expander("Q2 · Which genres dominate the dataset?", expanded=True):
            gc = df[GENRE_COL].value_counts().head(15)
            fig = make_bar_chart(gc.index.tolist(), gc.values.tolist(),
                                 "Top 15 Genres by Track Count", ylabel="Song Count", height=420)
            show_chart(fig, "eda_genre_dist",
                "Bar chart ranking the 15 most common genres. Dominant genres can bias ML models — "
                "if pop-dance has 3× more songs than jazz, the model will have more training signal for popular patterns. "
                "Hover bars to see exact counts.")

        with st.expander("Q3 · Popularity distribution", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                fig = make_histogram(df[POPULARITY_COL].dropna().values, "Popularity Distribution",
                                     xlabel="Popularity Score", color=CHART_COLORS[0], height=400)
                show_chart(fig, "eda_pop_hist",
                    "Histogram of song popularity scores (0–100). The dashed line marks the mean. "
                    "The distribution is slightly left-skewed, meaning most tracks cluster at moderate to higher popularity values, with a few lower-popularity outliers."
                    "This provides a reasonably balanced target for the regression model.")
            with col2:
                top_g = df[GENRE_COL].value_counts().head(7).index.tolist()
                fig2 = go.Figure()
                for i, g in enumerate(top_g):
                    vals = df[df[GENRE_COL]==g][POPULARITY_COL].dropna()
                    fig2.add_trace(go.Box(y=vals, name=g, marker=dict(color=CHART_COLORS[i]),
                                          line=dict(color=CHART_COLORS[i]), boxmean=True))
                apply_layout(fig2, "Popularity by Genre (Box)", height=400)
                fig2.update_layout(yaxis_title="Popularity", showlegend=False)
                show_chart(fig2, "eda_pop_box",
                    "Box plots show popularity spread per genre. The horizontal line is the median, "
                    "the dot is the mean, and whiskers show range. Wide boxes = high variance. "
                    "Genres with high medians are strong signals for the recommendation model.")

        with st.expander("Q4 · Audio feature distributions", expanded=False):
            n = len(numeric_features)
            cols_in_row = min(3, n)
            for row_start in range(0, n, cols_in_row):
                cols = st.columns(cols_in_row)
                for j, col_idx in enumerate(range(row_start, min(row_start+cols_in_row, n))):
                    feat = numeric_features[col_idx]
                    with cols[j]:
                        data = pd.to_numeric(df[feat], errors="coerce").dropna().values
                        fig = make_histogram(data, FEATURE_LABELS.get(feat,feat),
                                             xlabel=feat, color=CHART_COLORS[col_idx % len(CHART_COLORS)], height=300)
                        show_chart(fig, f"eda_feat_{feat}",
                            f"Distribution of {FEATURE_LABELS.get(feat,feat)}. "
                            f"Mean ≈ {np.nanmean(data):.1f}, Std ≈ {np.nanstd(data):.1f}. "
                            "Skewed features may benefit from log-transform before modeling.")

        with st.expander("Q5 · Feature correlation with popularity", expanded=True):
            corr_cols = [c for c in numeric_features + [POPULARITY_COL] if c in df.columns]
            corr = df[corr_cols].corr(numeric_only=True)
            col1, col2 = st.columns([2.1, 1])
            with col1:
                labels = [FEATURE_LABELS.get(c,c) for c in corr.columns]
                fig = make_heatmap(corr.values.tolist(), labels, labels, "Full Correlation Matrix", height=480)
                show_chart(fig, "eda_corr_heat",
                    "Correlation matrix: values near +1 (blue) mean features move together; "
                    "near −1 (red) they move inversely; near 0 means no linear relationship. "
                    "Strong correlations with Popularity highlight the best predictive features.")
            with col2:
                pop_corr = corr[POPULARITY_COL].drop(POPULARITY_COL).sort_values()
                colors = [PALETTE["accent2"] if v<0 else PALETTE["accent1"] for v in pop_corr.values]
                fig2 = go.Figure(go.Bar(
                    x=pop_corr.values, y=[FEATURE_LABELS.get(c,c) for c in pop_corr.index],
                    orientation="h",
                    marker=dict(color=colors, line=dict(width=0)),
                    hovertemplate="<b>%{y}</b><br>Pearson r = %{x:.3f}<extra></extra>",
                ))
                apply_layout(fig2, "Correlation with Popularity", height=480)
                fig2.update_layout(xaxis_title="Pearson r", showlegend=False)
                fig2.add_vline(x=0, line=dict(color=PALETTE["border"], width=1))
                show_chart(fig2, "eda_pop_corr",
                    "Bar chart of Pearson correlation between each audio feature and popularity. "
                    "Blue bars are positive predictors; red bars are negative. "
                    "The longer the bar, the stronger the linear relationship.")

        with st.expander("Q6 · Pairwise scatter: key features", expanded=False):
            top_f = [c for c in ["nrgy","dnce","val","acous",POPULARITY_COL] if c in df.columns]
            sample = df[top_f].dropna().sample(min(800, len(df)), random_state=42)
            for i, fi in enumerate(top_f):
                for j, fj in enumerate(top_f):
                    pass
            c1, c2 = st.columns(2)
            pairs = [(top_f[a], top_f[b]) for a in range(len(top_f)) for b in range(a+1, len(top_f))][:4]
            for idx_p, (fa, fb) in enumerate(pairs):
                with (c1 if idx_p%2==0 else c2):
                    fig = make_scatter_chart(
                        sample[fb].values, sample[fa].values,
                        f"{FEATURE_LABELS.get(fa,fa)} vs {FEATURE_LABELS.get(fb,fb)}",
                        xlabel=FEATURE_LABELS.get(fb,fb), ylabel=FEATURE_LABELS.get(fa,fa),
                        color=CHART_COLORS[idx_p], trendline=True, height=340)
                    show_chart(fig, f"eda_scatter_{fa}_{fb}",
                        f"Scatter plot of {FEATURE_LABELS.get(fa,fa)} vs {FEATURE_LABELS.get(fb,fb)}. "
                        "The dashed trend line shows the linear direction. Clustered dots suggest correlated features; "
                        "spread-out dots suggest independence — both matter for model design.")

    # ── TAB 2: REGRESSION ────────────────────────────────────
    with tabs[1]:
        st.markdown("### 📈 Song Popularity Prediction (Regression)")
        yte = models["y_te"]
        reg_results = models["regression_results"]
        reg_df = pd.DataFrame([
            {"Model": name, "MAE": vals["MAE"], "RMSE": vals["RMSE"], "R²": vals["R2"]}
            for name, vals in reg_results.items()
        ]).sort_values("RMSE", ascending=True)
        best_name = models["best_reg_name"]
        best_pred = reg_results[best_name]["pred"]
        residuals = yte.values - best_pred

        with st.expander("Q7 · How accurately can we predict popularity?", expanded=True):
            best_row = reg_df.iloc[0]
            render_metrics(
                metric_card("Best Model", best_name),
                metric_card("Best RMSE", f"{best_row['RMSE']:.2f}"),
                metric_card("Best MAE", f"{best_row['MAE']:.2f}"),
                metric_card("Best R²", f"{best_row['R²']:.3f}"),
            )
            st.markdown("**Regression Model Comparison**")
            display_df = reg_df.copy()
            display_df["MAE"] = display_df["MAE"].map(lambda x: f"{x:.3f}")
            display_df["RMSE"] = display_df["RMSE"].map(lambda x: f"{x:.3f}")
            display_df["R²"] = display_df["R²"].map(lambda x: f"{x:.3f}")
            st.dataframe(display_df, use_container_width=True, hide_index=True)

            c1, c2 = st.columns(2)
            with c1:
                series = {
                    "MAE": reg_df.set_index("Model")["MAE"].tolist(),
                    "RMSE": reg_df.set_index("Model")["RMSE"].tolist(),
                }
                fig = make_multi_bar_comparison(reg_df["Model"].tolist(), series,
                                                "Regression Error Comparison", height=430)
                fig.update_layout(yaxis_title="Error", xaxis_tickangle=-25, yaxis_range=None)
                show_chart(fig, "reg_all_error_compare",
                    "Comparison of MAE and RMSE across all regression models. Lower values are better. "
                    f"{best_name} has the lowest RMSE, so it is the strongest popularity prediction model in this analysis.")
            with c2:
                fig = make_bar_chart(reg_df["Model"].tolist(), reg_df["R²"].tolist(),
                                     "Regression R² Comparison", xlabel="Model", ylabel="R² Score", height=430)
                fig.update_layout(xaxis_tickangle=-25)
                show_chart(fig, "reg_all_r2_compare",
                    "R² compares how much popularity variance each model explains. Higher is better; negative R² means the model performs worse than predicting the average popularity.")

            c1, c2 = st.columns(2)
            mn, mx = float(yte.min()), float(yte.max())
            with c1:
                fig = make_scatter_chart(yte.values, best_pred, f"Best Model: {best_name}",
                                         "Actual Popularity", "Predicted Popularity",
                                         color=CHART_COLORS[2], height=400)
                fig.add_trace(go.Scatter(x=[mn,mx], y=[mn,mx], mode="lines",
                                         line=dict(color="white", width=1.5, dash="dash"), name="Perfect fit", hoverinfo="skip"))
                show_chart(fig, "reg_best_scatter",
                    f"Actual vs predicted popularity for {best_name}. Points closer to the dashed diagonal line are more accurate predictions. "
                    "This view shows how well the best model generalizes across the popularity range.")
            with c2:
                baseline_pred = reg_results["Linear Regression"]["pred"]
                fig = make_scatter_chart(yte.values, baseline_pred, "Baseline: Linear Regression",
                                         "Actual Popularity", "Predicted Popularity",
                                         color=CHART_COLORS[0], height=400)
                fig.add_trace(go.Scatter(x=[mn,mx], y=[mn,mx], mode="lines",
                                         line=dict(color="white", width=1.5, dash="dash"), name="Perfect fit", hoverinfo="skip"))
                show_chart(fig, "reg_linear_baseline_scatter",
                    "Linear Regression is shown as the baseline model. Comparing it with the best model shows whether added model complexity improves prediction accuracy.")

        with st.expander("Q8 · Are prediction errors random or structured?", expanded=True):
            c1, c2, c3 = st.columns(3)
            with c1:
                fig = make_residual_chart(best_pred, residuals, f"{best_name}: Residuals vs Predicted", height=380)
                show_chart(fig, "reg_resid_scatter",
                    "Residuals vs predicted values for the best model. A healthy pattern is mostly random scatter around zero, meaning the model is not making one-sided systematic errors.")
            with c2:
                fig = make_histogram(residuals, "Residual Distribution", xlabel="Residual", color=CHART_COLORS[2], height=380)
                show_chart(fig, "reg_resid_hist",
                    "Distribution of residuals (actual − predicted). Centering near zero suggests low average bias; wider spread indicates remaining prediction uncertainty.")
            with c3:
                fig = make_bar_chart(reg_df["Model"].tolist(), reg_df["R²"].tolist(),
                                     "Model R² Comparison", ylabel="R² Score", height=380)
                fig.update_layout(xaxis_tickangle=-35)
                show_chart(fig, "reg_r2_compare_all",
                    "Side-by-side R² scores for all regression models. This confirms whether non-linear and ensemble models add predictive value beyond the baseline.")

        with st.expander("Q9 · Prediction error by popularity range", expanded=True):
            bins_edges = [0,20,40,60,80,100]
            bin_labels = ["0–20","20–40","40–60","60–80","80–100"]
            df_res = pd.DataFrame({"actual":yte.values,"pred":best_pred,"residual":residuals})
            df_res["pop_bin"] = pd.cut(df_res["actual"], bins=bins_edges, labels=bin_labels, include_lowest=True)
            bin_rmse = df_res.groupby("pop_bin", observed=False)["residual"].apply(lambda x: np.sqrt((x**2).mean()) if len(x)>0 else np.nan)
            fig = make_bar_chart(bin_labels, [0 if pd.isna(v) else float(v) for v in bin_rmse.values],
                                 f"RMSE by Popularity Range — {best_name}", xlabel="Popularity Range", ylabel="RMSE", height=420)
            show_chart(fig, "reg_rmse_bins",
                "RMSE broken down by actual popularity bucket for the best regression model. Lower error in common buckets means the model learns typical songs better; higher error in rare buckets shows where additional data or features would help.")

    # ── TAB 3: CLASSIFICATION ────────────────────────────────
    with tabs[2]:
        st.markdown("### 🎯 Classification Models")
        yc_te = models["yc_te"]
        clf_results = models["classification_results"]

        with st.expander("Q10 · How well does Logistic Regression classify user preferences?", expanded=True):
            lr = clf_results["Logistic Reg"]
            render_metrics(
                metric_card("Accuracy", f"{lr['Accuracy']:.3f}"),
                metric_card("Precision", f"{lr['Precision']:.3f}"),
                metric_card("Recall", f"{lr['Recall']:.3f}"),
                metric_card("F1", f"{lr['F1']:.3f}"),
            )
            c1, c2 = st.columns(2)
            with c1:
                fig = make_confusion_matrix_chart(lr["cm"], "Confusion Matrix — Logistic Regression", height=400)
                show_chart(fig, "clf_confusion",
                    "Confusion matrix for user preference classification. The diagonal cells are correct predictions; off-diagonal cells show songs the model misclassified.")
            with c2:
                fig = make_roc_chart([lr["fpr"]], [lr["tpr"]], [lr["AUC"]], ["Logistic Reg"], "ROC Curve", height=400)
                show_chart(fig, "clf_roc_lr",
                    f"ROC curve for Logistic Regression. AUC = {lr['AUC']:.3f}; values above the diagonal indicate better-than-random separation between liked and not-liked songs.")

        with st.expander("Q11 · How do multiple user-preference classifiers compare?", expanded=True):
            clf_df = pd.DataFrame([
                {"Model": name, "Accuracy": r["Accuracy"], "Precision": r["Precision"], "Recall": r["Recall"], "F1": r["F1"], "AUC": r["AUC"]}
                for name, r in clf_results.items()
            ]).sort_values("F1", ascending=False)
            best_clf = clf_df.iloc[0]["Model"]
            render_metrics(
                metric_card("Best Classifier", best_clf),
                metric_card("Best F1", f"{clf_df.iloc[0]['F1']:.3f}"),
                metric_card("Best Recall", f"{clf_df['Recall'].max():.3f}"),
                metric_card("Models Compared", str(len(clf_df))),
            )
            st.dataframe(clf_df.assign(**{
                "Accuracy": clf_df["Accuracy"].map(lambda x: f"{x:.3f}"),
                "Precision": clf_df["Precision"].map(lambda x: f"{x:.3f}"),
                "Recall": clf_df["Recall"].map(lambda x: f"{x:.3f}"),
                "F1": clf_df["F1"].map(lambda x: f"{x:.3f}"),
                "AUC": clf_df["AUC"].map(lambda x: f"{x:.3f}"),
            }), use_container_width=True, hide_index=True)
            c1, c2 = st.columns(2)
            with c1:
                series = {n: [r["Accuracy"], r["Precision"], r["Recall"], r["F1"]] for n, r in clf_results.items()}
                fig = make_multi_bar_comparison(["Accuracy","Precision","Recall","F1"], series,
                                                "Classifier Metric Comparison", height=430)
                show_chart(fig, "clf_comparison",
                    "Grouped bar chart comparing Logistic Regression, Decision Tree, Random Forest, and SVM. This keeps the frontend aligned with the notebook and shows which model balances precision and recall best.")
            with c2:
                fprs = [r["fpr"] for r in clf_results.values()]
                tprs = [r["tpr"] for r in clf_results.values()]
                aucs = [r["AUC"] for r in clf_results.values()]
                fig = make_roc_chart(fprs, tprs, aucs, list(clf_results.keys()), "ROC Curves — All User Preference Classifiers", height=430)
                show_chart(fig, "clf_roc_all",
                    "ROC curves compare how well each classifier separates liked from not-liked songs across thresholds. The highest AUC has the strongest separation ability.")

        # Q12 Genre Classification removed from dashboard display to keep presentation aligned with PPT focus.

    # ── TAB 4: FEATURE IMPORTANCE ─────────────────────────────
    with tabs[3]:
        st.markdown("### 🏆 Feature Importance Analysis")
        with st.expander("Q13 · Which features drive popularity predictions?", expanded=True):
            try:
                rf_model = models["feature_model"].named_steps["model"]
                importances = rf_model.feature_importances_
                feat_labels = [FEATURE_LABELS.get(f,f) for f in numeric_features]
                c1, c2 = st.columns([1.2,0.8])
                with c1:
                    fig = make_feature_importance_bar(feat_labels, importances, "Random Forest Feature Importance", height=440)
                    show_chart(fig, "fi_rf_bar",
                        "Horizontal bars show how much each audio feature reduces prediction error in the Random Forest. "
                        "Longer bars = more important. Features at the top are the strongest predictors of popularity — "
                        "they carry the most weight in the recommendation scoring formula.")
                with c2:
                    top6_idx = np.argsort(importances)[-6:][::-1]
                    pie_labels = [feat_labels[i] for i in top6_idx] + ["Others"]
                    pie_vals = [importances[i] for i in top6_idx] + [importances[np.argsort(importances)[:-6]].sum()]
                    fig2 = make_donut_chart(pie_labels, pie_vals, "Top 6 Features' Share", height=400)
                    show_chart(fig2, "fi_rf_pie",
                        "Donut chart showing what share of total importance the top 6 features hold. "
                        "A dominant slice means the model relies heavily on that single feature — "
                        "which can be a strength or a vulnerability if the feature is noisy.")
                imp_df = pd.DataFrame({"Feature":feat_labels,"Importance":importances}).sort_values("Importance",ascending=False)
                st.dataframe(imp_df.reset_index(drop=True), use_container_width=True)
            except Exception as e:
                st.info(f"Feature importance unavailable: {e}")


    # ── TAB 5: USER BEHAVIOR ──────────────────────────────────
    with tabs[4]:
        st.markdown("### 👤 Real User Behavior Analysis")
        with st.expander("Q14 · How are users interacting with the app?", expanded=True):
            c1, c2 = st.columns(2)
            with c1:
                if not ratings.empty and "rating" in ratings.columns:
                    rc = pd.to_numeric(ratings["rating"], errors="coerce").value_counts().sort_index()
                    star_colors = [PALETTE["accent2"],PALETTE["accent4"],PALETTE["accent4"],PALETTE["accent3"],PALETTE["accent3"]]
                    fig = make_bar_chart(rc.index.astype(str).tolist(), rc.values.tolist(),
                                         "User Ratings Distribution", xlabel="Star Rating", ylabel="Count",
                                         color=star_colors[:len(rc)], height=380)
                    show_chart(fig, "ub_ratings",
                        "Distribution of star ratings (1–5) submitted by all users. "
                        "A spike at 4–5 stars means users tend to rate songs they already like (selection bias). "
                        "The 3-star peak often represents ambivalence. This data trains the preference classifier.")
                else: st.info("No ratings yet.")
            with c2:
                if not history.empty and "time_period" in history.columns:
                    order = ["Morning","Afternoon","Evening","Night"]
                    tc = history["time_period"].value_counts().reindex(order).fillna(0)
                    fig = make_bar_chart(tc.index.tolist(), tc.values.tolist(),
                                         "Listening Activity by Time", xlabel="Period", ylabel="Listens",
                                         color=[PALETTE["accent4"],PALETTE["accent1"],PALETTE["accent2"],PALETTE["accent3"]], height=380)
                    show_chart(fig, "ub_time",
                        "Total listening sessions split by time of day across all users. "
                        "The tallest bar shows peak usage. The time-aware scoring formula "
                        "weights these periods differently when computing recommendation scores.")
                else: st.info("No listening records yet.")

        with st.expander("Q15 · Genre engagement analysis", expanded=True):
            c1, c2 = st.columns(2)
            with c1:
                if not history.empty and GENRE_COL in history.columns:
                    glc = history[GENRE_COL].value_counts().head(10)
                    fig = make_bar_chart(glc.index.tolist(), glc.values.tolist(),
                                         "Top Listened Genres", ylabel="Listen Count", height=400)
                    show_chart(fig, "ub_listened_genres",
                        "Top 10 genres by cumulative listen count across all users. "
                        "This is revealed preference data — what people actually played, not just said they liked. "
                        "The system uses this to build the time-period genre model.")
                else: st.info("No listening history yet.")
            with c2:
                if not prefs.empty and "preferred_genres" in prefs.columns:
                    pref_list = []
                    for val in prefs["preferred_genres"].dropna(): pref_list.extend(split_genres(val))
                    if pref_list:
                        ps = pd.Series(pref_list).value_counts().head(10)
                        fig = make_bar_chart(ps.index.tolist(), ps.values.tolist(),
                                             "Preferred Genres (Setup)", ylabel="User Count", height=400)
                        show_chart(fig, "ub_pref_genres",
                            "Genres users selected during onboarding setup. "
                            "Comparing this to the listened-genres chart reveals whether stated preferences "
                            "match actual listening behavior — gaps here are interesting for future cold-start work.")
                    else: st.info("No preferred genres saved yet.")
                else: st.info("No preferences yet.")

        with st.expander("Q16 · Liked vs. Not-Liked audio features", expanded=True):
            if not ratings.empty and "liked" in ratings.columns:
                rated_songs = ratings.merge(df[[TRACK_COL,ARTIST_COL]+numeric_features], on=[TRACK_COL,ARTIST_COL], how="inner")
                if not rated_songs.empty:
                    liked_avg = rated_songs[rated_songs["liked"]==1][numeric_features].mean()
                    nlked_avg = rated_songs[rated_songs["liked"]==0][numeric_features].mean()
                    compare_df = pd.DataFrame({"Liked":liked_avg,"Not Liked":nlked_avg}).dropna()
                    if not compare_df.empty:
                        feat_names = [FEATURE_LABELS.get(c,c) for c in compare_df.index]
                        fig = go.Figure()
                        fig.add_trace(go.Bar(name="Liked", x=feat_names, y=compare_df["Liked"].values,
                                             marker=dict(color=PALETTE["accent3"],line=dict(width=0)),
                                             hovertemplate="<b>%{x}</b><br>Liked avg: %{y:.1f}<extra></extra>"))
                        fig.add_trace(go.Bar(name="Not Liked", x=feat_names, y=compare_df["Not Liked"].values,
                                             marker=dict(color=PALETTE["accent2"],line=dict(width=0),opacity=0.85),
                                             hovertemplate="<b>%{x}</b><br>Not Liked avg: %{y:.1f}<extra></extra>"))
                        apply_layout(fig, "Avg Audio Features: Liked vs Not Liked", height=420)
                        fig.update_layout(barmode="group", bargap=0.2, yaxis_title="Average Value")
                        show_chart(fig, "ub_liked_compare",
                            "Grouped bars compare average audio feature values between liked (green) and not-liked (red) songs. "
                            "Large gaps between bars on a feature indicate it's a strong preference signal. "
                            "These gaps are what the Logistic Regression classifier learns to exploit.")
                    else: st.info("Not enough rated songs to compare.")
            else: st.info("No ratings data yet.")

    # ── TAB 6: RECOMMENDATION SYSTEM ─────────────────────────
    with tabs[5]:
        st.markdown("### 💡 Recommendation System Explainability")
        with st.expander("Q17 · How is the final score computed?", expanded=True):
            st.markdown("""<div class="formula-box"><b>Final Score =</b><br><br>
0.20 × User Similarity
(cosine similarity to liked song profile) +<br>
 0.18 × Preference Probability
(logistic regression output) + <br>
0.16 × Predicted Popularity
(regression model, normalized 0–1) + <br>
0.14 × Time Score
(audio feature fit for current period) + <br>
0.12 × CF User Match
(user-based collaborative filtering signal) +<br>
0.10 × CF Item Match
(item-based collaborative filtering similarity) + <br>
0.06 × Time Genre Match
(genre listened to during this time period) + <br>
0.04 × Preferred Genre Match
(user setup genre preferences) 
</div>""", unsafe_allow_html=True)
            weights = {"User Similarity": 0.20,
    "Pref Probability": 0.18,
    "Pred Popularity": 0.16,
    "Time Score": 0.14,
    "CF User Match": 0.12,
    "CF Item Match": 0.10,
    "Time Genre Match": 0.06,
    "Pref Genre Match": 0.04
}
            fig = make_donut_chart(list(weights.keys()), list(weights.values()), "Score Weight Distribution", height=420)
            show_chart(fig, "rec_weights",
                "Donut chart showing how much each component contributes to the final recommendation score. "
                "User Similarity dominates (28%) — the system prioritises matching your audio taste profile. "
                "Genre signals (10%+6%) ensure genre diversity isn't completely overridden by feature vectors.")

        with st.expander("Q18 · Recommendation log patterns", expanded=True):
            if logs.empty:
                st.info("No recommendation logs yet. Generate recommendations first.")
            else:
                c1, c2 = st.columns(2)
                with c1:
                    scores = pd.to_numeric(logs["final_score"], errors="coerce").dropna()
                    fig = make_histogram(scores.values, "Recommendation Score Distribution",
                                         xlabel="Final Score", color=PALETTE["accent1"], height=380)
                    show_chart(fig, "rec_score_dist",
                        "Histogram of final recommendation scores across all logged recommendations. "
                        f"Mean score ≈ {scores.mean():.3f}. Higher scores indicate stronger confidence. "
                        "A narrow distribution means the system discriminates well; wide = low confidence overall.")
                with c2:
                    if GENRE_COL in logs.columns:
                        rg = logs[GENRE_COL].value_counts().head(10)
                        fig = make_bar_chart(
                            x=rg.index.tolist(),
                            y=rg.values.tolist(),
                            title="Most Recommended Genres",
                            xlabel="Times Recommended",
                            ylabel="Genre",
                            horizontal=True,
                            height=380
                        )
                        fig.update_layout(yaxis=dict(autorange="reversed"))
                        show_chart(fig, "rec_genres",
                            "Top genres that appear in recommendation logs. "
                            "If this closely mirrors the Preferred Genres chart, the system is personalizing well. "
                            "A mismatch suggests model features are overriding stated genre preferences.")
                st.markdown("**Recent Recommendation Logs**")
                st.dataframe(logs.sort_values("recommended_at",ascending=False).head(12)
                             [["user_id","title","artist",GENRE_COL,"time_period","final_score","recommended_at"]], use_container_width=True)

        with st.expander("📁 Raw CSV Tables", expanded=False):
            for label, frame in [("users.csv",users),("user_preferences.csv",prefs),
                                  ("listening_history.csv",history),("ratings.csv",ratings),("recommendation_logs.csv",logs)]:
                st.markdown(f"**{label}**"); st.dataframe(frame, use_container_width=True)


def about_page():
    render_header("ℹ️ About MoodMix PDA", "Project overview, architecture, and technical details")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""<div class="glass-card"><div style="font-family:'Sora',sans-serif;font-weight:700;font-size:1rem;margin-bottom:10px">📌 Project Overview</div>
        <div style="font-size:0.88rem;color:#9CA3AF;line-height:1.65">MoodMix PDA is a Predictive Analytics music recommendation system built without a database.
        It uses a file-based backend and a multi-model ML pipeline to generate personalized, explainable song recommendations.</div></div>""", unsafe_allow_html=True)
        st.markdown("""<div class="glass-card"><div style="font-family:'Sora',sans-serif;font-weight:700;font-size:1rem;margin-bottom:10px">🗂️ Backend Files</div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:0.82rem;color:#5A6480;line-height:1.9">
        users.csv<br>user_preferences.csv<br>listening_history.csv<br>ratings.csv<br>recommendation_logs.csv</div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="glass-card"><div style="font-family:'Sora',sans-serif;font-weight:700;font-size:1rem;margin-bottom:10px">🤖 ML Models Used</div>
        <div style="font-size:0.88rem;color:#9CA3AF;line-height:1.8">
        • <b>Linear Regression</b> — baseline popularity prediction<br>
        • <b>Polynomial Regression</b> — nonlinear popularity modeling<br>
        • <b>Decision Tree Regressor</b> — interpretable popularity prediction<br>
        • <b>Random Forest Regressor</b> — best ensemble popularity prediction + feature importance<br>
        • <b>Logistic Regression</b> — user preference classification<br>
        • <b>Decision Tree / Random Forest / SVM</b> — preference classification and model comparison<br>
        • <b>Cosine Similarity</b> — user audio profile matching</div></div>""", unsafe_allow_html=True)
        st.markdown("""<div class="glass-card"><div style="font-family:'Sora',sans-serif;font-weight:700;font-size:1rem;margin-bottom:10px">👥 Team 14</div>
        <div style="font-size:0.88rem;color:#9CA3AF;line-height:1.8">
        Halima Taju Brmaji (2095075)<br>Hamza Ahmed Shad (2395933)<br>Harshita Ganesh Devadiga (2440413)<br>Shreyashree Mondal (2447509)</div></div>""", unsafe_allow_html=True)


# ============================================================
# MAIN
# ============================================================
def main():
    try:
        with st.spinner("Loading dataset and training ML models — please wait…"):
            df, numeric_features = load_song_data()
            models = train_models_cached(df, tuple(numeric_features))
    except Exception as e:
        st.error("Could not load top10s.csv. Make sure it is in the same folder as final.py.")
        st.error(str(e)); return

    with st.sidebar:
        st.markdown("""<div style="font-family:'Sora',sans-serif;font-size:1.2rem;font-weight:800;
        background:linear-gradient(90deg,#7C6FFF,#FF5F7E);-webkit-background-clip:text;
        -webkit-text-fill-color:transparent;background-clip:text;margin-bottom:4px">🎵 MoodMix PDA</div>
        <div style="font-size:0.75rem;color:#C5CCFF;margin-bottom:14px">Predictive Music Recommendation</div>""", unsafe_allow_html=True)


        if st.session_state["logged_in"]:
            st.markdown(f"""<div style="background:#161923;border:1px solid #1E2235;border-radius:10px;padding:10px 12px;margin-bottom:14px">
            <div style="font-size:0.78rem;color:#5A6480;margin-bottom:2px">Logged in as</div>
            <div style="font-weight:700;color:#E2E8F8">{st.session_state['username']}</div>
            <div style="font-size:0.78rem;color:#7C6FFF">{get_time_period()} Session</div></div>""", unsafe_allow_html=True)
            pages = ["Dashboard","Explore Songs","My History","Recommendations","Analytics Dashboard","About Project"]
            if st.session_state.get("page") not in pages:
                st.session_state["page"] = "Dashboard"

            # One-click navigation without using key="page" directly.
            current_index = pages.index(st.session_state.get("page", "Dashboard")) if st.session_state.get("page") in pages else 0
            selected_page = st.radio("Navigation", pages, index=current_index, key="nav_page")
            st.session_state["page"] = selected_page
            st.markdown("---")
            if st.button("Logout", use_container_width=True): logout_user(); st.rerun()
        else:
            st.session_state["page"] = "Welcome"; st.info("Please log in to continue.")

    if not st.session_state["logged_in"]: welcome_page(); return
    if not has_user_preferences(st.session_state["user_id"]) and st.session_state["page"] != "About Project":
        preference_setup_page(df); return

    page = st.session_state["page"]
    if page=="Dashboard":             dashboard_page(df)
    elif page=="Explore Songs":       explore_songs_page(df)
    elif page=="My History":          my_history_page()
    elif page=="Recommendations":     recommendations_page(df, numeric_features, models)
    elif page=="Analytics Dashboard": analytics_page(df, numeric_features, models)
    elif page=="About Project":       about_page()

if __name__ == "__main__":
    main()
