import streamlit as st

from ui.theme import PALETTE as P


def apply_custom_style():
    st.markdown(f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        html, body, [class*="css"] {{ font-family: 'Inter', system-ui, sans-serif; }}
        .stApp {{ background: {P['bg']}; }}
        .block-container {{ padding-top: 3.2rem; max-width: 1400px; }}
        section[data-testid="stSidebar"] {{ background: {P['surface']}; border-right: 1px solid {P['border']}; }}

        .hero {{
            background: linear-gradient(120deg, {P['primary']} 0%, {P['sky']} 55%, {P['teal']} 100%);
            border-radius: 18px; padding: 22px 28px; color: white; margin-bottom: 18px;
            box-shadow: 0 10px 30px rgba(108, 92, 231, 0.25);
        }}
        .hero h1 {{ color: white; font-size: 1.75rem; font-weight: 700; margin: 0; padding: 0; }}
        .hero p {{ color: rgba(255,255,255,0.9); margin: 4px 0 0 0; font-size: 0.95rem; }}

        div[data-testid="stMetric"] {{
            background: {P['surface']}; border: 1px solid {P['border']}; border-radius: 14px;
            padding: 14px 16px; box-shadow: 0 2px 10px rgba(30, 34, 51, 0.04);
            border-top: 4px solid {P['primary']};
        }}
        div[data-testid="stMetric"] label {{ color: {P['muted']}; font-weight: 500; }}
        {_metric_accents()}

        div[data-testid="stPlotlyChart"] {{
            background: {P['surface']}; border: 1px solid {P['border']}; border-radius: 14px; padding: 6px;
        }}
        .stTabs [data-baseweb="tab-list"] {{ gap: 6px; }}
        .stTabs [data-baseweb="tab"] {{
            background: {P['surface']}; border: 1px solid {P['border']}; border-radius: 999px;
            padding: 6px 16px; height: auto;
        }}
        .stTabs [aria-selected="true"] {{ background: {P['primary']}; color: white; border-color: {P['primary']}; }}
        .stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {{ display: none; }}

        .pill {{ display:inline-block; padding: 3px 12px; border-radius: 999px; font-size: 0.8rem; font-weight: 600; }}
        .pill-good {{ background: #DDF7F4; color: #00897F; }}
        .pill-bad {{ background: #FFE3E3; color: #D63A3A; }}
        .pill-warn {{ background: #FFF1D6; color: #B57500; }}
        .note {{ color: {P['muted']}; font-size: 0.88rem; }}
        </style>
    """, unsafe_allow_html=True)


def _metric_accents() -> str:
    """Gives each KPI card in a row its own accent colour."""
    rules = []
    for i, key in enumerate(["teal", "amber", "coral", "sky"], start=2):
        sel = f'div[data-testid="stHorizontalBlock"] > div:nth-child({i}) div[data-testid="stMetric"]'
        rules.append(f"{sel} {{ border-top-color: {P[key]}; }}")
    return "\n".join(rules)


def render_header(title: str, subtitle: str = ""):
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)


def pill(text: str, kind: str = "good") -> str:
    return f'<span class="pill pill-{kind}">{text}</span>'
