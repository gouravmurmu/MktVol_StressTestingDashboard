import numpy as np
import pandas as pd

from risk.metrics import RiskMetrics
from risk.stress import HISTORICAL_SCENARIOS, StressTester


def test_var_ordering_and_cvar(returns):
    r = returns["SPY"]
    m = RiskMetrics()
    assert m.calculate_var_historical(r, 0.99) > m.calculate_var_historical(r, 0.95) > 0
    assert m.calculate_cvar(r, 0.99) >= m.calculate_var_historical(r, 0.99)
    table = m.var_table(r, 0.99)
    assert set(table.index) == {"Historical", "Gaussian", "Student-t", "Cornish-Fisher"}
    assert (table["VaR"] > 0).all()


def test_gaussian_var_matches_closed_form():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0, 0.01, 200_000))
    assert abs(RiskMetrics().calculate_var_parametric(r, 0.99) - 0.02326) < 5e-4


def test_drawdown_and_ratios(returns):
    m = RiskMetrics()
    dd = m.drawdown(returns["SPY"])
    assert dd.max() <= 1e-12 and dd.min() < 0
    assert np.isfinite([m.sharpe(returns["SPY"]), m.sortino(returns["SPY"]), m.calmar(returns["SPY"])]).all()
    assert abs(m.beta(returns["SPY"], returns["SPY"]) - 1) < 1e-9


def test_kupiec_accepts_correct_model():
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(0, 0.01, 5000))
    var = pd.Series(0.02326, index=r.index)
    assert RiskMetrics.kupiec_test(r, var, 0.99)["p_value"] > 0.05
    assert RiskMetrics.kupiec_test(r, var / 2, 0.99)["p_value"] < 0.01


def test_scenarios_and_reverse_stress():
    st = StressTester()
    scen = HISTORICAL_SCENARIOS[0]
    impact = st.scenario_impact({"A": 1.0, "B": 0.5}, {"A": 0.5, "B": 0.5}, scen)
    assert abs(impact["Contribution"].sum() - 0.75 * scen.market_shock) < 1e-12
    paths = st.scenario_path(scen, 0.01, beta=1.0, n_paths=50, recovery_days=0)
    np.testing.assert_allclose(paths[-1], 1 + scen.market_shock, rtol=1e-9)
    assert abs(st.reverse_stress({"A": 2.0}, {"A": 1.0}, 0.2) + 0.1) < 1e-12
