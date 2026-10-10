from typing import Dict

import numpy as np
import pandas as pd
import streamlit as st

from models import volatility as volmodels
from regime.detector import RegimeDetector
from risk.metrics import RiskMetrics
from risk.stress import HISTORICAL_SCENARIOS, Scenario, StressTester
from simulation.monte_carlo import MonteCarloSimulator
from ui.layout import pill
from visualization.charts import Visualizer as V

stress_engine = StressTester()
regime_engine = RegimeDetector()


def _money(x: float) -> str:
    """Compact currency, e.g. -$66.9k or $1.25M, so values fit in KPI cards."""
    sign, x = ("-" if x < 0 else ""), abs(x)
    for div, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "k")):
        if x >= div:
            return f"{sign}${x / div:,.{2 if x / div < 10 else 1}f}{suffix}"
    return f"{sign}${x:,.0f}"


# ---------- Cached model runs ----------
@st.cache_data(show_spinner=False)
def _vol_forecasts(returns: pd.Series, horizon: int):
    return {k: (f.annualized(), f.aic, f.persistence) for k, f in volmodels.forecast_all(returns, horizon).items()}


@st.cache_data(show_spinner=False)
def _vol_backtest(returns: pd.Series, horizon: int):
    return volmodels.backtest_vol_models(returns, horizon=horizon)


@st.cache_data(show_spinner=False)
def _garch_var(returns: pd.Series, conf: float):
    return volmodels.garch_var_backtest(returns, conf)


@st.cache_data(show_spinner=False)
def _regimes(returns: pd.Series):
    probs = regime_engine.detect_volatility_regime(returns)
    return probs, probs.attrs["summary"]


# ---------- Tabs ----------
def render_overview(prices: pd.DataFrame, returns: pd.DataFrame, port: pd.Series, weights: Dict[str, float],
                    risk: RiskMetrics):
    c = st.columns(5)
    total = np.expm1(port.sum())
    c[0].metric("Total Return", f"{total:.1%}")
    c[1].metric("Annual Volatility", f"{port.std() * np.sqrt(risk.periods):.1%}")
    c[2].metric("Sharpe", f"{risk.sharpe(port):.2f}")
    c[3].metric("Max Drawdown", f"{risk.max_drawdown(port):.1%}")
    c[4].metric("Sortino", f"{risk.sortino(port):.2f}")

    trend = regime_engine.detect_regime(np.exp(port.cumsum()))
    st.plotly_chart(V.plot_growth(prices, port, trend["Regime"]), width="stretch")
    st.caption("Shaded areas mark bearish trend periods (50-day average below 200-day average).")

    left, right = st.columns(2)
    with left:
        st.plotly_chart(V.plot_drawdown(risk.drawdown(port)), width="stretch")
    with right:
        if returns.shape[1] > 1:
            st.plotly_chart(V.plot_correlation(returns), width="stretch")
        else:
            st.plotly_chart(V.plot_rolling_vol({"30d": risk.calculate_volatility(port, 30),
                                                "EWMA": risk.calculate_ewma_volatility(port)}), width="stretch")

    from data.loader import MarketDataLoader
    from data.portfolio import PortfolioManager
    stats = MarketDataLoader.get_asset_stats(returns, risk.periods)
    stats.insert(0, "Weight", pd.Series(weights))
    if returns.shape[1] > 1:
        stats["Risk Contribution"] = PortfolioManager.risk_contributions(returns, weights)
    st.markdown("#### Holdings")
    st.dataframe(stats.style.format({
        "Weight": "{:.0%}", "Annual Return": "{:.1%}", "Annual Volatility": "{:.1%}", "Sharpe": "{:.2f}",
        "Skewness": "{:.2f}", "Excess Kurtosis": "{:.2f}", "Risk Contribution": "{:.0%}",
    }), width="stretch")


def render_risk_analysis(port: pd.Series, risk: RiskMetrics):
    col1, col2 = st.columns([1, 3])
    with col1:
        conf = st.slider("Confidence level", 0.90, 0.995, 0.99, 0.005, format="%.3f")
        lookback = st.slider("Lookback (periods)", 60, max(61, len(port)), min(500, len(port)))
        horizon = st.select_slider("Holding period (days)", [1, 5, 10, 21], 1,
                                   help="Scaled with the square-root-of-time rule.")
        recent = port.tail(lookback)
        table = risk.var_table(recent, conf) * np.sqrt(horizon)
        cvar = risk.calculate_cvar(recent, conf) * np.sqrt(horizon)
        st.metric(f"Historical VaR ({conf:.1%})", f"{table.loc['Historical', 'VaR']:.2%}")
        st.metric(f"Expected Shortfall ({conf:.1%})", f"{cvar:.2%}")
    with col2:
        st.plotly_chart(V.plot_return_distribution(recent, table.loc["Historical", "VaR"] / np.sqrt(horizon),
                                                   cvar / np.sqrt(horizon)), width="stretch")

    st.markdown("#### VaR by method")
    st.plotly_chart(V.plot_bar(table["VaR"], f"{horizon}-day VaR at {conf:.1%}"), width="stretch")
    st.caption("Student-t and Cornish-Fisher adjust for fat tails and skew, which a Gaussian model misses.")

    st.markdown("#### Does the model hold up? VaR backtest")
    which = st.radio("Model", ["Rolling historical (250d)", "GJR-GARCH-t"], horizontal=True)
    if which.startswith("Rolling"):
        var_series = risk.rolling_var(port, 250, conf)
        bt = pd.DataFrame({"return": port, "var": var_series}).dropna()
    else:
        with st.spinner("Fitting GARCH and forecasting VaR day by day..."):
            bt = _garch_var(port, conf)
    test = risk.kupiec_test(bt["return"], bt["var"], conf)
    verdict = pill("Model passes", "good") if test["p_value"] >= 0.05 else pill("Model rejected", "bad")
    st.markdown(f"{verdict} &nbsp; {test['breaches']} breaches vs {test['expected']:.1f} expected over "
                f"{test['observations']} days · Kupiec p-value {test['p_value']:.3f}", unsafe_allow_html=True)
    st.plotly_chart(V.plot_var_backtest(bt["return"], bt["var"], f"{which} VaR backtest"), width="stretch")


def render_forecasts(prices: pd.Series, port: pd.Series, risk: RiskMetrics):
    st.markdown("#### Volatility forecasting models")
    horizon = st.slider("Forecast horizon (trading days)", 5, 63, 21)
    with st.spinner("Fitting EWMA, HAR-RV, GARCH, GJR-GARCH and EGARCH..."):
        fc = _vol_forecasts(port, horizon)

    cols = st.columns(len(fc))
    for col, (name, (path, _, _)) in zip(cols, fc.items()):
        col.metric(name, f"{path[-1]:.1%}", delta=f"{path[-1] - path[0]:+.1%}", delta_color="inverse",
                   help=f"Annualized volatility forecast for day {horizon}; delta is the change from day 1.")

    history = risk.calculate_ewma_volatility(port)
    st.plotly_chart(V.plot_vol_forecast(history, {k: v[0] for k, v in fc.items()}), width="stretch")

    with st.expander("Which model forecasts best? Out-of-sample scorecard", expanded=True):
        with st.spinner("Backtesting models on held-out data..."):
            board = _vol_backtest(port, 5)
        if board.empty:
            st.info("Load a longer history (at least ~2 years of daily data) to run the backtest.")
        else:
            best = board.index[0]
            st.markdown(f"Best model on the last 30% of the sample: {pill(best)}", unsafe_allow_html=True)
            st.dataframe(board.style.format({"RMSE (ann.)": "{:.2%}", "QLIKE": "{:.3f}"})
                         .highlight_min(subset=["QLIKE", "RMSE (ann.)"], color="#DDF7F4"), width="stretch")
            st.caption("Models are fit on the first 70% of data and forecast 5-day realized volatility every day "
                       "of the remaining 30%. QLIKE is the standard loss for volatility forecasts; lower is better.")

    st.markdown("#### Market regime model")
    try:
        with st.spinner("Fitting Markov-switching model..."):
            probs, summary = _regimes(port)
        current = probs["Regime"].iloc[-1]
        p_turb = probs["Turbulent"].iloc[-1]
        kind = "bad" if current == "Turbulent" else "good"
        st.markdown(f"Current regime: {pill(current, kind)} &nbsp; probability of turbulence {p_turb:.0%}",
                    unsafe_allow_html=True)
        st.plotly_chart(V.plot_regime_probability(np.exp(port.cumsum()) * 100, probs), width="stretch")
        st.dataframe(summary.style.format({"Annual Vol": "{:.1%}", "Expected Duration (days)": "{:.0f}"}),
                     width="stretch")
    except Exception as e:  # noqa: BLE001
        st.warning(f"Regime model did not converge on this sample: {e}")


def render_simulation(port: pd.Series, risk: RiskMetrics):
    col1, col2 = st.columns([1, 3])
    with col1:
        model = st.selectbox("Model", ["GJR-GARCH-t", "Block bootstrap", "Student-t GBM", "GBM"],
                             help="GARCH and bootstrap keep volatility clustering; GBM assumes constant volatility.")
        value = st.number_input("Portfolio value ($)", 1_000, 100_000_000, 100_000, step=10_000)
        T = st.slider("Horizon (days)", 10, 252, 63)
        n = st.slider("Simulations", 500, 10_000, 2_000, step=500)
        hist_mu = port.mean() * risk.periods
        hist_sigma = port.std() * np.sqrt(risk.periods)
        if model in ("GBM", "Student-t GBM"):
            mu = st.number_input("Expected annual return", value=float(round(hist_mu, 3)), step=0.01)
            sigma = st.number_input("Annual volatility", value=float(round(hist_sigma, 3)), step=0.01)
        seed = st.number_input("Random seed", 0, 10_000, 42)
        run = st.button("Run simulation", type="primary", width="stretch")

    with col2:
        if not run and "sim" not in st.session_state:
            st.info("Choose a model and press **Run simulation**.")
            return
        if run:
            sim = MonteCarloSimulator(int(seed))
            with st.spinner("Simulating..."):
                if model == "GBM":
                    paths = sim.simulate_gbm(1.0, mu, sigma, T, n_sims=n)
                elif model == "Student-t GBM":
                    paths = sim.simulate_student_t(1.0, mu, sigma, T, n_sims=n)
                elif model == "Block bootstrap":
                    paths = sim.simulate_bootstrap(1.0, port, T, n_sims=n)
                else:
                    paths = sim.simulate_garch(1.0, port, T, n_sims=n)
            st.session_state["sim"] = (paths * value, value, model)
        paths, v0, used = st.session_state["sim"]
        stats = MonteCarloSimulator.calculate_stats(paths[-1], v0)
        st.plotly_chart(V.plot_simulation(paths, port.index[-1]), width="stretch")
        c = st.columns(4)
        c[0].metric("Median outcome", _money(np.median(paths[-1])), f"{stats['median_return']:+.1%}")
        c[1].metric("Probability of loss", f"{stats['prob_loss']:.0%}")
        c[2].metric("95% VaR", _money(-stats['var_95'] * v0))
        c[3].metric("95% Expected Shortfall", _money(-stats['cvar_95'] * v0))
        st.plotly_chart(V.plot_terminal_distribution(paths[-1] / v0 - 1), width="stretch")
        st.caption(f"Model: {used}")


def render_stress_test(returns: pd.DataFrame, port: pd.Series, weights: Dict[str, float],
                       benchmark: pd.Series, risk: RiskMetrics):
    betas = {a: risk.beta(returns[a], benchmark) for a in returns.columns}
    port_beta = sum(weights[a] * betas[a] for a in weights)
    daily_vol = port.std()

    left, right = st.columns([1, 2])
    with left:
        names = [s.name for s in HISTORICAL_SCENARIOS] + ["Custom scenario"]
        choice = st.selectbox("Scenario", names)
        if choice == "Custom scenario":
            shock = st.slider("Market move", -0.80, 0.0, -0.30, 0.01, format="%.2f")
            volx = st.slider("Volatility multiplier", 1.0, 6.0, 2.5, 0.1)
            days = st.slider("Duration (trading days)", 1, 500, 60)
            scen = Scenario("Custom scenario", shock, volx, days, "Your own what-if shock.")
        else:
            scen = next(s for s in HISTORICAL_SCENARIOS if s.name == choice)
        st.info(scen.description)
        value = st.number_input("Portfolio value ($)", 1_000, 100_000_000, 100_000, step=10_000, key="stress_val")
        st.markdown(f"Portfolio beta to benchmark: **{port_beta:.2f}**")

    impact = stress_engine.scenario_impact(betas, weights, scen)
    total = impact["Contribution"].sum()
    with right:
        c = st.columns(3)
        c[0].metric("Scenario P&L", _money(total * value), f"{total:.1%}")
        c[1].metric("Stressed volatility", f"{daily_vol * scen.vol_multiplier * np.sqrt(252):.0%}",
                    f"x{scen.vol_multiplier:.1f}", delta_color="inverse")
        stressed = stress_engine.simulate_shock(port, "volatility", scen.vol_multiplier)
        c[2].metric("Stressed 99% VaR", f"{risk.calculate_var_historical(stressed, 0.99):.2%}",
                    f"{risk.calculate_var_historical(stressed, 0.99) - risk.calculate_var_historical(port, 0.99):+.2%}",
                    delta_color="inverse")
        paths = stress_engine.scenario_path(scen, daily_vol, port_beta)
        st.plotly_chart(V.plot_stress_paths(paths, value, f"Simulated path through: {scen.name}"), width="stretch")

    a, b = st.columns([3, 2])
    with a:
        st.plotly_chart(V.plot_waterfall(impact["Contribution"]), width="stretch")
    with b:
        st.markdown("#### Reverse stress test")
        loss = st.slider("Loss you can tolerate", 0.05, 0.60, 0.20, 0.01, format="%.2f")
        needed = stress_engine.reverse_stress(betas, weights, loss)
        st.markdown(f"A benchmark move of **{needed:.1%}** would cause a **{-loss:.0%}** portfolio loss.")
        hits = [s.name for s in HISTORICAL_SCENARIOS if s.market_shock <= needed]
        if hits:
            st.markdown("Historical episodes at least that severe: " + ", ".join(hits))
        st.dataframe(impact.style.format({"Weight": "{:.0%}", "Beta": "{:.2f}", "Shocked Return": "{:.1%}",
                                          "Contribution": "{:.1%}"}), width="stretch")
