# Market Volatility & Stress Testing Dashboard

A "Finance-grade" interactive risk terminal for exploring market volatility, calculating VaR/CVaR, and running stress tests.

## Overview
This dashboard allows risk managers and investors to:
- **Visualize Risk**: See volatility clustering and regime changes.
- **Quantify Downside**: Calculate Value-at-Risk (VaR) and Expected Shortfall (CVaR).
- **Simulate Futures**: Run Monte Carlo simulations to see potential future price paths.
- **Stress Test**: Apply "what-if" scenarios (e.g., 2008 Crash, 2x Volatility) to your portfolio.

## Key Features
- **Real-time Data**: Fetches data via `yfinance`.
- **Regime Detection**: Identifies Bull/Bear markets automatically.
- **Interactive Risk Explorer**: Adjust parameters on the fly to see how risk metrics change.

## Architecture
- `data/`: Market data fetching and processing.
- `risk/`: Core risk mathematics (VaR, CVaR, etc.).
- `simulation/`: Geometric Brownian Motion engine.
- `regime/`: Market regime detection logic.
- `visualization/`: Plotly-based financial charting.
- `ui/`: Streamlit interface components.

## How to Run
```bash
pip install -r requirements.txt
streamlit run ui/app.py
```
