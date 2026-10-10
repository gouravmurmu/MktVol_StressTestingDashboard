import os
import sys
from datetime import datetime, timedelta

import streamlit as st

# Make the project root importable when launched with `streamlit run ui/app.py`
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.loader import MarketDataLoader, periods_per_year  # noqa: E402
from data.portfolio import PortfolioManager  # noqa: E402
from risk.metrics import RiskMetrics  # noqa: E402
from ui import layout, panels  # noqa: E402
from ui.theme import register_plotly_template  # noqa: E402

st.set_page_config(page_title="Market Risk Terminal", layout="wide", page_icon="📈")
register_plotly_template()
layout.apply_custom_style()

# ---------- Sidebar ----------
st.sidebar.markdown("## ⚙️ Portfolio setup")
source = st.sidebar.radio("Data source", ["Live (Yahoo Finance)", "Demo data"], horizontal=True,
                          help="Demo data is synthetic and works offline.")
raw = st.sidebar.text_input("Tickers (comma separated)", "SPY, QQQ, TLT, GLD")
tickers = list(dict.fromkeys(t.strip().upper() for t in raw.split(",") if t.strip()))
benchmark = st.sidebar.text_input("Benchmark", "SPY").strip().upper() or "SPY"

c1, c2 = st.sidebar.columns(2)
start_date = c1.date_input("Start", datetime.now() - timedelta(days=365 * 5))
end_date = c2.date_input("End", datetime.now())
interval = st.sidebar.selectbox("Interval", ["1d", "1wk", "1mo"], help="Forecast models assume daily data.")

weights = {}
if tickers:
    with st.sidebar.expander("Weights", expanded=len(tickers) > 1):
        for t in tickers:
            weights[t] = st.slider(t, 0, 100, round(100 / len(tickers)), key=f"w_{t}")


@st.cache_data(show_spinner=False, ttl=3600)
def load_prices(tickers, start, end, interval, source):
    loader = MarketDataLoader()
    if source.startswith("Live"):
        try:
            prices = loader.fetch_data(tickers, str(start), str(end), interval)
            if not prices.empty:
                return prices, None
            note = "Yahoo Finance returned no data"
        except Exception as e:  # noqa: BLE001
            note = f"Yahoo Finance is unreachable ({type(e).__name__})"
        return loader.generate_demo_data(tickers, str(start), str(end)), f"{note}, so demo data is shown."
    return loader.generate_demo_data(tickers, str(start), str(end)), None


if st.sidebar.button("Load data", type="primary", width="stretch"):
    if not tickers:
        st.sidebar.error("Enter at least one ticker.")
    else:
        all_tickers = tickers + ([benchmark] if benchmark not in tickers else [])
        with st.spinner("Fetching market data..."):
            prices, note = load_prices(tuple(all_tickers), start_date, end_date, interval, source)
        missing = [t for t in tickers if t not in prices.columns]
        if prices.empty or len(prices) < 30:
            st.sidebar.error("Not enough data. Check the tickers and date range.")
        else:
            st.session_state.update(prices=prices, note=note, tickers=[t for t in tickers if t in prices.columns],
                                    benchmark=benchmark, interval=interval)
            st.session_state.pop("sim", None)
            if missing:
                st.sidebar.warning(f"No data for: {', '.join(missing)}")

layout.render_header("Market Volatility & Stress Testing Terminal",
                     "Portfolio risk, volatility forecasting, regime detection, Monte Carlo and stress scenarios")

if "prices" not in st.session_state:
    st.info("👈 Pick your tickers and weights, then press **Load data**. "
            "Choose **Demo data** to explore without an internet connection.")
    st.stop()

if st.session_state["note"]:
    st.warning(st.session_state["note"])

all_prices = st.session_state["prices"]
held = st.session_state["tickers"]
bench = st.session_state["benchmark"]
risk = RiskMetrics(periods_per_year(st.session_state["interval"]))

prices = all_prices[held]
returns = MarketDataLoader.calculate_returns(prices)
bench_returns = MarketDataLoader.calculate_returns(all_prices[[bench]])[bench] \
    if bench in all_prices else returns.mean(axis=1)
w = PortfolioManager.normalize_weights(held, {t: weights.get(t, 0) for t in held})
port = PortfolioManager().construct_portfolio(returns, w)

tabs = st.tabs(["📊 Overview", "🛡️ Risk & VaR", "🔮 Forecasts", "🎲 Monte Carlo", "⚡ Stress Testing"])
with tabs[0]:
    panels.render_overview(prices, returns, port, w, risk)
with tabs[1]:
    panels.render_risk_analysis(port, risk)
with tabs[2]:
    panels.render_forecasts(prices.iloc[:, 0], port, risk)
with tabs[3]:
    panels.render_simulation(port, risk)
with tabs[4]:
    panels.render_stress_test(returns, port, w, bench_returns, risk)
