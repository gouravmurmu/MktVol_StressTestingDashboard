import streamlit as st
import pandas as pd
import numpy as np

from visualization.charts import Visualizer
from risk.metrics import RiskMetrics
from risk.stress import StressTester
from simulation.monte_carlo import MonteCarloSimulator
from regime.detector import RegimeDetector

# Initialize Engines
risk_engine = RiskMetrics()
stress_engine = StressTester()
sim_engine = MonteCarloSimulator()
regime_engine = RegimeDetector()

def render_overview(prices: pd.DataFrame, returns: pd.DataFrame):
    st.subheader("Market Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        total_ret = (prices.iloc[-1] / prices.iloc[0] - 1).mean()
        st.metric("Total Return", f"{total_ret:.2%}")
    with col2:
        vol = returns.std() * np.sqrt(252)
        avg_vol = vol.mean()
        st.metric("Annual Volatility", f"{avg_vol:.2%}")
    with col3:
        # Simple Regime Check (last point)
        regimes = regime_engine.detect_regime(prices.iloc[:, 0]) # Check first asset for regime proxy
        current_regime = "Bull" if regimes['Regime'].iloc[-1] == 1 else "Bear"
        st.metric("Regime (Est.)", current_regime)
    with col4:
        sharpe = (returns.mean() * 252) / (returns.std() * np.sqrt(252))
        st.metric("Sharpe Ratio", f"{sharpe.mean():.2f}")

    # Charts
    st.plotly_chart(Visualizer.plot_price_and_regime(prices.iloc[:, 0]), use_container_width=True)

def render_risk_analysis(returns: pd.DataFrame):
    st.subheader("Risk Metrics (VaR & CVaR)")
    
    # Portfolio Return Series (Equal weighted for now if multi-asset, or just use first)
    if isinstance(returns, pd.DataFrame):
        port_returns = returns.mean(axis=1) # Simplified equal weight
    else:
        port_returns = returns

    col1, col2 = st.columns([1, 3])
    
    with col1:
        conf_level = st.slider("Confidence Level", 0.90, 0.99, 0.95, 0.01)
        lookback = st.slider("Lookback (Days)", 30, 500, 252)
        
        # Recalculate
        recent_returns = port_returns.tail(lookback)
        var_hist = risk_engine.calculate_var_historical(recent_returns, conf_level)
        cvar_hist = risk_engine.calculate_cvar(recent_returns, conf_level)
        
        st.markdown("### Key Metrics")
        st.metric(f"VaR ({conf_level:.0%})", f"{var_hist:.2%}", help="Estimated loss that will not be exceeded with X% confidence.")
        st.metric(f"CVaR ({conf_level:.0%})", f"{cvar_hist:.2%}", help="Expected loss IF the VaR threshold is breached.")

    with col2:
        st.plotly_chart(Visualizer.plot_return_distribution(recent_returns, var_hist, cvar_hist), use_container_width=True)

def render_simulation(prices: pd.DataFrame):
    st.subheader("Monte Carlo Simulation")
    
    # Sim params
    asset = prices.iloc[:, 0] # First asset
    last_price = asset.iloc[-1]
    
    col1, col2 = st.columns([1, 3])
    with col1:
        T_days = st.slider("Horizon (Days)", 10, 252, 30)
        n_sims = st.slider("Num Simulations", 100, 5000, 1000)
        mu_annual = st.number_input("Expected Annual Return", value=0.07, step=0.01)
        vol_annual = st.number_input("Expected Annual Volatility", value=asset.pct_change().std()*np.sqrt(252), step=0.01)
        
    with col2:
        if st.button("Run Simulation"):
            with st.spinner("Simulating..."):
                paths = sim_engine.simulate_gbm(last_price, mu_annual, vol_annual, T_days, n_sims=n_sims)
                stats = sim_engine.calculate_stats(paths[-1], last_price)
                
                st.plotly_chart(Visualizer.plot_simulation(paths), use_container_width=True)
                
                # Stats Row
                c1, c2, c3 = st.columns(3)
                c1.metric("Median Price", f"${np.median(paths[-1]):.2f}")
                c2.metric("99% Worst Case", f"${stats['var_99']*last_price + last_price:.2f}")
                c3.metric("Best Case", f"${stats['best_case']*last_price + last_price:.2f}")

def render_stress_test(returns: pd.DataFrame):
    st.subheader("Stress Testing Playground")
    
    if isinstance(returns, pd.DataFrame):
        port_returns = returns.mean(axis=1)
    else:
        port_returns = returns

    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Scenarios")
        scenario = st.selectbox("Choose Scenario", ["2x Volatility Shock", "Negative Drift (Bear)", "2008 Crash Analog (-20%)"])
        
        stressed_ret = port_returns.copy()
        description = ""
        
        if scenario == "2x Volatility Shock":
            stressed_ret = stress_engine.simulate_shock(port_returns, "volatility", 2.0)
            description = "Doubles the daily volatility. Represents a panic market."
        elif scenario == "Negative Drift (Bear)":
            stressed_ret = stress_engine.simulate_shock(port_returns, "drift", -0.30)
            description = "Adds a -30% annualized drift to the returns."
        elif scenario == "2008 Crash Analog (-20%)":
             # Extremely simplified
            stressed_ret = stress_engine.simulate_shock(port_returns, "drift", -0.50) 
            stressed_ret = stress_engine.simulate_shock(stressed_ret, "volatility", 3.0)
            description = "Combined high volatility (3x) and severe negative drift (-50% annualized)."
        
        st.info(description)
        
    with col2:
        # Compare VaR before and after
        var_normal = risk_engine.calculate_var_historical(port_returns, 0.99)
        var_stress = risk_engine.calculate_var_historical(stressed_ret, 0.99)
        
        # Delta chart (simple bar)
        comp_df = pd.DataFrame({
            "Scenario": ["Normal", "Stressed"],
            "VaR (99%)": [var_normal, var_stress]
        })
        st.bar_chart(comp_df.set_index("Scenario"))
        
        st.metric("New VaR (99%)", f"{var_stress:.2%}", delta=f"{var_normal-var_stress:.2%}", delta_color="inverse")
