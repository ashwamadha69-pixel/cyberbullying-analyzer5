"""
Cyberbullying Impact Analyzer - Streamlit App
------------------------------------------------
Loads the trained TF-IDF + Logistic Regression model and:
  1. Classifies input text into a cyberbullying category
     (age, ethnicity, gender, religion, other_cyberbullying, not_cyberbullying)
  2. Estimates a psychological "Impact / Severity Score" using a
     transparent, explainable heuristic that combines:
       - model confidence
       - category weight (based on research on identity-based harassment
         being associated with greater psychological harm)
       - presence of intensifier / threat language
  3. Displays a category probability breakdown and simple guidance.

Visual design: bold violet -> fuchsia -> pink gradient, dark modern SaaS
aesthetic, glass-panel cards, a severity-reactive gauge, and a drifting
gradient "aura" backdrop.

Run:
    streamlit run app.py
"""

import re
import base64
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from PIL import Image

MODEL_DIR = "model"
LOGO_PATH = Path(__file__).parent / "assets" / "robo_mascot.png"


@st.cache_data
def get_logo_base64() -> str:
    """Base64-encode the mascot logo so it can be inlined in the HTML hero."""
    with open(LOGO_PATH, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# ---------------------------------------------------------------------
# Config: category weights reflect that identity-based bullying
# (ethnicity, religion, gender) is associated in the literature with
# deeper/longer-lasting psychological impact than generic insults.
# These are presented as an illustrative, adjustable heuristic -
# not a clinical diagnostic tool.
# ---------------------------------------------------------------------
CATEGORY_WEIGHTS = {
    "not_cyberbullying": 0.0,
    "other_cyberbullying": 0.55,
    "age": 0.6,
    "gender": 0.75,
    "religion": 0.85,
    "ethnicity": 0.9,
}

INTENSIFIER_WORDS = [
    "kill", "die", "hate", "ugly", "worthless", "stupid", "idiot",
    "disgusting", "pathetic", "loser", "never", "everyone", "always",
]

# ---------------------------------------------------------------------
# Design tokens - bold violet -> fuchsia -> pink gradient system.
# Severity colors ramp from cool violet (low concern) to hot ember
# (severe), so the palette itself communicates escalating intensity.
# ---------------------------------------------------------------------
COLOR_VOID = "#0B0714"
COLOR_SURFACE = "rgba(255,255,255,0.045)"
COLOR_BORDER = "rgba(255,255,255,0.09)"
COLOR_VIOLET = "#7C3AED"
COLOR_FUCHSIA = "#D946EF"
COLOR_PINK = "#EC4899"
COLOR_EMBER = "#F43F5E"
COLOR_INK_HI = "#F5F3FF"
COLOR_INK_LO = "#A79FC7"

TIER_STYLES = {
    "No significant concern detected": {"color": "#22D3B6", "glow": "34,211,182"},
    "Low": {"color": COLOR_VIOLET, "glow": "124,58,237"},
    "Moderate": {"color": COLOR_FUCHSIA, "glow": "217,70,239"},
    "High": {"color": COLOR_PINK, "glow": "236,72,153"},
    "Severe": {"color": COLOR_EMBER, "glow": "244,63,94"},
}


@st.cache_resource
def load_artifacts():
    clf = joblib.load(f"{MODEL_DIR}/model.pkl")
    vectorizer = joblib.load(f"{MODEL_DIR}/vectorizer.pkl")
    le = joblib.load(f"{MODEL_DIR}/label_encoder.pkl")
    return clf, vectorizer, le


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_impact_score(label, confidence, raw_text):
    """Return an impact score 0-100, a severity tier, and matched intensifier words."""
    base = CATEGORY_WEIGHTS.get(label, 0.5)
    text_lower = raw_text.lower()
    matched = [w for w in INTENSIFIER_WORDS if w in text_lower]
    intensity_boost = min(len(matched) * 0.05, 0.2)

    score = (base * 0.7 + confidence * 0.2 + intensity_boost) * 100
    score = float(np.clip(score, 0, 100))

    if label == "not_cyberbullying":
        tier = "No significant concern detected"
    elif score < 40:
        tier = "Low"
    elif score < 65:
        tier = "Moderate"
    elif score < 85:
        tier = "High"
    else:
        tier = "Severe"

    return round(score, 1), tier, matched


# =======================================================================
# STYLING
# =======================================================================
def inject_css():
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@500;600&display=swap');

        html, body, [class*="css"] {{
            font-family: 'Inter', sans-serif;
        }}

        #MainMenu, header, footer {{ visibility: hidden; }}

        .stApp {{
            background: {COLOR_VOID};
        }}

        .block-container {{
            max-width: 920px;
            padding-top: 1.6rem;
            padding-bottom: 2.5rem;
        }}

        div[data-testid="stVerticalBlock"] {{
            gap: 0.6rem;
        }}
        div[data-testid="stElementContainer"] {{
            margin-bottom: 0;
        }}

        /* ---------- aurora mesh backdrop: drifting blobs + panning dot-grid ---------- */
        .bg-aura {{
            position: fixed;
            inset: 0;
            z-index: -1;
            overflow: hidden;
            background: {COLOR_VOID};
        }}
        .blob {{
            position: absolute;
            border-radius: 50%;
            filter: blur(120px);
            will-change: transform;
        }}
        .blob-a {{
            width: 640px; height: 640px;
            background: {COLOR_VIOLET};
            top: -220px; left: -180px;
            opacity: 0.38;
            animation: driftA 26s ease-in-out infinite;
        }}
        .blob-b {{
            width: 560px; height: 560px;
            background: {COLOR_PINK};
            bottom: -220px; right: -160px;
            opacity: 0.34;
            animation: driftB 21s ease-in-out infinite;
        }}
        .blob-c {{
            width: 460px; height: 460px;
            background: {COLOR_FUCHSIA};
            top: 28%; right: 8%;
            opacity: 0.22;
            animation: driftC 29s ease-in-out infinite;
        }}
        .blob-d {{
            width: 380px; height: 380px;
            background: {COLOR_EMBER};
            bottom: 10%; left: 5%;
            opacity: 0.16;
            animation: driftD 33s ease-in-out infinite;
        }}
        @keyframes driftA {{
            0%, 100% {{ transform: translate(0, 0) scale(1); }}
            33% {{ transform: translate(90px, 50px) scale(1.12); }}
            66% {{ transform: translate(40px, -60px) scale(0.94); }}
        }}
        @keyframes driftB {{
            0%, 100% {{ transform: translate(0, 0) scale(1); }}
            50% {{ transform: translate(-80px, -60px) scale(1.15); }}
        }}
        @keyframes driftC {{
            0%, 100% {{ transform: translate(0, 0) rotate(0deg); }}
            50% {{ transform: translate(-60px, 70px) rotate(18deg); }}
        }}
        @keyframes driftD {{
            0%, 100% {{ transform: translate(0, 0) scale(1); }}
            50% {{ transform: translate(50px, -40px) scale(1.2); }}
        }}
        .grid-overlay {{
            position: absolute;
            inset: -10%;
            background-image: radial-gradient(rgba(255,255,255,0.075) 1px, transparent 1.2px);
            background-size: 36px 36px;
            -webkit-mask-image: radial-gradient(ellipse 70% 55% at 50% 28%, black 0%, transparent 72%);
            mask-image: radial-gradient(ellipse 70% 55% at 50% 28%, black 0%, transparent 72%);
            animation: panGrid 50s linear infinite;
        }}
        @keyframes panGrid {{
            from {{ background-position: 0 0; }}
            to   {{ background-position: 360px 360px; }}
        }}
        @media (prefers-reduced-motion: reduce) {{
            .blob, .grid-overlay {{ animation: none; }}
        }}

        /* ---------- hero ---------- */
        .hero-eyebrow {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.78rem;
            letter-spacing: 0.18em;
            color: {COLOR_PINK};
            text-transform: uppercase;
            font-weight: 600;
            margin-bottom: 0.6rem;
        }}
        .hero-title {{
            font-family: 'Space Grotesk', sans-serif;
            font-weight: 700;
            font-size: 3rem;
            line-height: 1.1;
            margin: 0 0 0.6rem 0;
            background: linear-gradient(135deg, {COLOR_VIOLET} 0%, {COLOR_FUCHSIA} 55%, {COLOR_PINK} 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}
        .hero-sub {{
            font-family: 'Inter', sans-serif;
            font-size: 1.02rem;
            color: {COLOR_INK_LO};
            max-width: 640px;
            line-height: 1.55;
        }}

        /* ---------- glass card ---------- */
        .glass-card {{
            background: {COLOR_SURFACE};
            border: 1px solid {COLOR_BORDER};
            border-radius: 22px;
            padding: 1.4rem 1.7rem;
            backdrop-filter: blur(18px);
            box-shadow: 0 8px 32px rgba(124, 58, 237, 0.12);
            margin-bottom: 1rem;
        }}
        .section-label {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.72rem;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            color: {COLOR_INK_LO};
            font-weight: 600;
            margin-bottom: 0.9rem;
        }}

        /* ---------- reveal animation ---------- */
        @keyframes fadeInUp {{
            from {{ opacity: 0; transform: translateY(18px); }}
            to   {{ opacity: 1; transform: translateY(0); }}
        }}
        .reveal {{ animation: fadeInUp 0.55s ease forwards; }}
        .reveal-delay-1 {{ animation-delay: 0.08s; opacity: 0; animation-fill-mode: forwards; }}
        .reveal-delay-2 {{ animation-delay: 0.16s; opacity: 0; animation-fill-mode: forwards; }}
        .reveal-delay-3 {{ animation-delay: 0.24s; opacity: 0; animation-fill-mode: forwards; }}

        /* ---------- stat pills ---------- */
        .stat-row {{ display: flex; gap: 1rem; }}
        .stat-pill {{
            flex: 1;
            background: rgba(255,255,255,0.03);
            border: 1px solid {COLOR_BORDER};
            border-radius: 16px;
            padding: 1rem 1.2rem;
        }}
        .stat-pill-label {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.68rem;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            color: {COLOR_INK_LO};
            margin-bottom: 0.35rem;
        }}
        .stat-pill-value {{
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.5rem;
            font-weight: 700;
            color: {COLOR_INK_HI};
        }}

        /* ---------- severity badge ---------- */
        .severity-badge {{
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.45rem 1.1rem;
            border-radius: 999px;
            font-family: 'Space Grotesk', sans-serif;
            font-weight: 600;
            font-size: 0.92rem;
        }}
        .severity-dot {{
            width: 9px; height: 9px; border-radius: 50%;
        }}

        /* ---------- chips ---------- */
        .chip {{
            display: inline-block;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.74rem;
            padding: 0.28rem 0.7rem;
            border-radius: 999px;
            background: rgba(236, 72, 153, 0.12);
            border: 1px solid rgba(236, 72, 153, 0.35);
            color: {COLOR_PINK};
            margin: 0.15rem 0.3rem 0.15rem 0;
        }}

        /* ---------- native widget overrides ---------- */
        div[data-testid="stTextArea"] textarea {{
            background: rgba(255,255,255,0.035);
            border: 1px solid {COLOR_BORDER};
            border-radius: 16px;
            color: {COLOR_INK_HI};
            font-family: 'Inter', sans-serif;
            font-size: 0.98rem;
            padding: 1rem;
        }}
        div[data-testid="stTextArea"] textarea:focus {{
            border-color: {COLOR_PINK};
            box-shadow: 0 0 0 3px rgba(236, 72, 153, 0.18);
        }}
        div[data-testid="stTextArea"] label {{
            font-family: 'Space Grotesk', sans-serif;
            color: {COLOR_INK_HI};
            font-weight: 600;
        }}

        div[data-testid="stButton"] > button {{
            background: linear-gradient(135deg, {COLOR_VIOLET}, {COLOR_PINK});
            border: none;
            border-radius: 999px;
            color: white;
            font-family: 'Space Grotesk', sans-serif;
            font-weight: 600;
            font-size: 0.98rem;
            padding: 0.7rem 2.2rem;
            box-shadow: 0 6px 24px rgba(236, 72, 153, 0.35);
            transition: transform 0.25s ease, box-shadow 0.25s ease;
        }}
        div[data-testid="stButton"] > button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 10px 32px rgba(236, 72, 153, 0.5);
            color: white;
        }}
        div[data-testid="stButton"] > button:active {{
            transform: translateY(0px);
        }}

        div[data-testid="stAlert"] {{
            background: rgba(255,255,255,0.04);
            border: 1px solid {COLOR_BORDER};
            border-radius: 16px;
        }}

        div[data-testid="stExpander"] {{
            background: rgba(255,255,255,0.03);
            border: 1px solid {COLOR_BORDER};
            border-radius: 16px;
        }}

        hr {{ border-color: {COLOR_BORDER}; }}

        ::-webkit-scrollbar {{ width: 10px; }}
        ::-webkit-scrollbar-track {{ background: {COLOR_VOID}; }}
        ::-webkit-scrollbar-thumb {{
            background: linear-gradient(180deg, {COLOR_VIOLET}, {COLOR_PINK});
            border-radius: 10px;
        }}
        </style>
        <div class="bg-aura">
            <div class="blob blob-a"></div>
            <div class="blob blob-b"></div>
            <div class="blob blob-c"></div>
            <div class="blob blob-d"></div>
            <div class="grid-overlay"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_gauge(score: float, tier: str):
    style = TIER_STYLES[tier]
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={
                "suffix": "",
                "font": {"size": 48, "color": COLOR_INK_HI, "family": "Space Grotesk"},
            },
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickcolor": COLOR_INK_LO,
                    "tickfont": {"color": COLOR_INK_LO, "size": 11},
                },
                "bar": {"color": style["color"], "thickness": 0.32},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 40], "color": "rgba(124,58,237,0.18)"},
                    {"range": [40, 65], "color": "rgba(217,70,239,0.22)"},
                    {"range": [65, 85], "color": "rgba(236,72,153,0.26)"},
                    {"range": [85, 100], "color": "rgba(244,63,94,0.3)"},
                ],
                "threshold": {
                    "line": {"color": style["color"], "width": 3},
                    "thickness": 0.9,
                    "value": score,
                },
            },
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": COLOR_INK_HI},
        height=230,
        margin=dict(l=20, r=20, t=25, b=5),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_probability_chart(prob_df: pd.DataFrame):
    n = len(prob_df)
    # violet -> pink gradient across the sorted bars, regardless of value,
    # so the chart itself carries the brand gradient
    colors = []
    for i in range(n):
        t = i / max(n - 1, 1)
        colors.append(
            f"rgba({int(124 + t * (236 - 124))},{int(58 + t * (72 - 58))},{int(237 + t * (153 - 237))},0.9)"
        )

    fig = go.Figure(
        go.Bar(
            x=prob_df["Probability"],
            y=prob_df["Category"],
            orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
            text=[f"{v:.0%}" for v in prob_df["Probability"]],
            textposition="outside",
            textfont=dict(color=COLOR_INK_HI, family="JetBrains Mono", size=12),
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": COLOR_INK_LO, "family": "Inter"},
        xaxis=dict(range=[0, 1.08], showgrid=True, gridcolor="rgba(255,255,255,0.06)", tickformat=".0%"),
        yaxis=dict(showgrid=False),
        height=230,
        margin=dict(l=10, r=10, t=25, b=5),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def main():
    st.set_page_config(
        page_title="Cyberbullying Impact Analyzer",
        page_icon=Image.open(LOGO_PATH),
        layout="centered",
    )
    inject_css()

    # ---------- Hero ----------
    logo_b64 = get_logo_base64()
    st.markdown(
        f"""
        <div class="reveal" style="margin-bottom: 1.4rem;">
            <img src="data:image/png;base64,{logo_b64}" alt="Robo mascot logo"
                 style="height: 64px; width: auto; display: block; margin-bottom: 0.9rem;" />
            <div class="hero-eyebrow">AI SAFETY &middot; FORENSIC PSYCHOLOGY PROJECT</div>
            <div class="hero-title">Cyberbullying Impact<br/>Analyzer</div>
            <div class="hero-sub">
                Paste any tweet, comment, or message. A trained ML classifier detects
                cyberbullying and its category, then a transparent heuristic estimates
                the potential psychological impact on a recipient.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    clf, vectorizer, le = load_artifacts()

    st.markdown('<div class="glass-card reveal">', unsafe_allow_html=True)
    text_input = st.text_area(
        "Message to analyze",
        height=120,
        placeholder='e.g. "You are so stupid, nobody wants you here"',
        label_visibility="visible",
    )
    analyze = st.button("Analyze Message", type="primary")
    st.markdown("</div>", unsafe_allow_html=True)

    if analyze and text_input.strip():
        cleaned = clean_text(text_input)
        vec = vectorizer.transform([cleaned])

        probs = clf.predict_proba(vec)[0]
        pred_idx = int(np.argmax(probs))
        label = le.classes_[pred_idx]
        confidence = float(probs[pred_idx])

        score, tier, matched = compute_impact_score(label, confidence, text_input)
        style = TIER_STYLES[tier]

        # ---------- classification stat row ----------
        st.markdown(
            f"""
            <div class="glass-card reveal reveal-delay-1">
                <div class="section-label">Classification Result</div>
                <div class="stat-row">
                    <div class="stat-pill">
                        <div class="stat-pill-label">Predicted Category</div>
                        <div class="stat-pill-value">{label.replace('_', ' ').title()}</div>
                    </div>
                    <div class="stat-pill">
                        <div class="stat-pill-label">Model Confidence</div>
                        <div class="stat-pill-value">{confidence:.1%}</div>
                    </div>
                </div>
                <span class="severity-badge" style="margin-top: 0.9rem; background: rgba({style['glow']},0.14); border: 1px solid rgba({style['glow']},0.4); color: {style['color']};">
                    <span class="severity-dot" style="background:{style['color']};"></span>
                    {tier}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ---------- gauge + probability breakdown, side by side in one card ----------
        prob_df = pd.DataFrame(
            {
                "Category": [c.replace("_", " ").title() for c in le.classes_],
                "Probability": probs,
            }
        ).sort_values("Probability", ascending=True)

        st.markdown('<div class="glass-card reveal reveal-delay-2">', unsafe_allow_html=True)
        col_gauge, col_chart = st.columns([1, 1.15], gap="medium")
        with col_gauge:
            st.markdown('<div class="section-label">Impact Score (0-100)</div>', unsafe_allow_html=True)
            render_gauge(score, tier)
            if matched:
                chips = "".join(f'<span class="chip">{w}</span>' for w in matched)
                st.markdown(
                    f'<div style="margin-top: -0.6rem;"><span class="stat-pill-label">Intensifier words detected</span><br/>{chips}</div>',
                    unsafe_allow_html=True,
                )
        with col_chart:
            st.markdown('<div class="section-label">Category Probability Breakdown</div>', unsafe_allow_html=True)
            render_probability_chart(prob_df)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="glass-card reveal reveal-delay-3">', unsafe_allow_html=True)
        if label != "not_cyberbullying":
            st.warning(
                f"This message shows characteristics of **{label.replace('_', ' ')}**-based "
                f"cyberbullying with an estimated **{tier.lower()}** potential impact on a "
                "recipient. Repeated exposure to identity-based harassment is associated in "
                "research with higher psychological distress."
            )
        else:
            st.success("This message does not show characteristics of cyberbullying.")
        st.markdown("</div>", unsafe_allow_html=True)

        with st.expander("How is the Impact Score calculated?"):
            st.markdown(
                """
                The score is a transparent heuristic (not a clinical measure) combining:
                - **Category weight**: identity-based categories (ethnicity, religion, gender)
                  are weighted higher, reflecting research on identity-based harassment causing
                  greater psychological harm than generic insults.
                - **Model confidence**: how certain the classifier is about the category.
                - **Intensity boost**: presence of threatening/absolutist language
                  (e.g. "kill", "always", "everyone").
                """
            )

    st.markdown(
        f"""
        <div style="text-align:center; margin-top: 1.4rem; color:{COLOR_INK_LO}; font-size:0.82rem; font-family:'Inter',sans-serif;">
            This tool is for educational/research demonstration purposes and is not a
            clinical diagnostic instrument.
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
