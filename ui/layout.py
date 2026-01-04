import streamlit as st

def apply_custom_style():
    st.markdown("""
        <style>
        /* General styling */
        .main {
            background-color: #0e1117;
            color: #fafafa;
        }
        h1, h2, h3 {
            color: #fafafa;
            font-family: 'Inter', sans-serif;
        }
        .stMetric {
            background-color: #262730;
            padding: 10px;
            border-radius: 5px;
            border: 1px solid #41424b;
        }
        /* KPI Cards */
        div[data-testid="metric-container"] {
            background-color: #1f2937;
            padding: 10px 15px;
            border-radius: 8px;
            border-left: 5px solid #3b82f6; 
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        div[data-testid="metric-container"] > label {
            color: #9ca3af;
            font-size: 0.8rem;
        }
        div[data-testid="metric-container"] > div {
            color: #f3f4f6;
        }
        
        /* Custom Warning/Info Boxes */
        .info-box {
            background-color: #1e3a8a;
            color: #dbeafe;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            border-left: 5px solid #60a5fa;
        }
        .risk-box {
            background-color: #7f1d1d;
            color: #fca5a5;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            border-left: 5px solid #f87171;
        }
        </style>
    """, unsafe_allow_html=True)

def render_header(title: str, subtitle: str = ""):
    st.title(title)
    if subtitle:
        st.markdown(f"*{subtitle}*")
    st.markdown("---")
