import streamlit as st
import sys
import os

# Add the parent directory to sys.path to resolve 'data', 'risk', etc.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from datetime import datetime, timedelta

# Import custom modules
from data.loader import MarketDataLoader
import ui.layout as layout
import ui.panels as panels

# Page Config
st.set_page_config(page_title="Market Risk Terminal", layout="wide", page_icon="📈")

# Apply Styling
layout.apply_custom_style()

# Sidebar
st.sidebar.title("Risk Config")
tickers_input = st.sidebar.text_input("Tickers (comma separated)", "SPY, QQQ, GLD")
tickers = [t.strip().upper() for t in tickers_input.split(",")]

start_date = st.sidebar.date_input("Start Date", datetime.now() - timedelta(days=365*2))
end_date = st.sidebar.date_input("End Date", datetime.now())
interval = st.sidebar.selectbox("Interval", ["1d", "1wk", "1mo"])

# Data Fetching
@st.cache_data
def load_data(tickers, start, end, interval):
    loader = MarketDataLoader()
    prices = loader.fetch_data(tickers, str(start), str(end), interval)
    returns = loader.calculate_returns(prices)
    return prices, returns

data_state = st.empty()

if st.sidebar.button("Load Data", type="primary"):
    with st.spinner("Fetching Market Data..."):
        try:
            prices, returns = load_data(tickers, start_date, end_date, interval)
            st.session_state['prices'] = prices
            st.session_state['returns'] = returns
            st.sidebar.success("Data Loaded!")
        except Exception as e:
            st.sidebar.error(f"Error loading data: {e}")

# Main Layout
layout.render_header("Market Volatility & Stress Testing Terminal")

if 'prices' in st.session_state:
    prices = st.session_state['prices']
    returns = st.session_state['returns']
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Risk Analysis", "Monte Carlo", "Stress Testing"])
    
    with tab1:
        panels.render_overview(prices, returns)
    
    with tab2:
        panels.render_risk_analysis(returns)
        
    with tab3:
        panels.render_simulation(prices)
        
    with tab4:
        panels.render_stress_test(returns)

else:
    st.info("👈 Please enter tickers and click 'Load Data' to begin analysis.")
