import numpy as np

from models import volatility as vm
from regime.detector import RegimeDetector
from simulation.monte_carlo import MonteCarloSimulator


def test_forecast_all_models(returns):
    fc = vm.forecast_all(returns["SPY"], horizon=10)
    assert {"EWMA (λ=0.94)", "HAR-RV", "GARCH(1,1)", "GJR-GARCH(1,1)"} <= set(fc)
    for f in fc.values():
        ann = f.annualized()
        assert len(ann) == 10 and np.all((ann > 0.01) & (ann < 2))


def test_backtest_scorecard(returns):
    board = vm.backtest_vol_models(returns["SPY"], horizon=5)
    assert not board.empty and (board["QLIKE"] >= 0).all()


def test_garch_var_backtest(returns):
    bt = vm.garch_var_backtest(returns["SPY"], 0.99)
    assert (bt["var"].dropna() > 0).all()


def test_markov_regimes(returns):
    probs = RegimeDetector().detect_volatility_regime(returns["SPY"])
    assert np.allclose(probs[["Calm", "Turbulent"]].sum(axis=1), 1)
    summary = probs.attrs["summary"]
    assert summary.loc["Turbulent", "Annual Vol"] > summary.loc["Calm", "Annual Vol"]


def test_simulators_shapes(returns):
    sim = MonteCarloSimulator(0)
    for paths in (sim.simulate_gbm(100, 0.05, 0.2, 20, n_sims=100),
                  sim.simulate_student_t(100, 0.05, 0.2, 20, n_sims=100),
                  sim.simulate_bootstrap(100, returns["SPY"], 20, n_sims=100),
                  sim.simulate_garch(100, returns["SPY"], 20, n_sims=100)):
        assert paths.shape == (21, 100) and np.all(paths[0] == 100) and np.all(paths > 0)
    stats = sim.calculate_stats(paths[-1], 100)
    assert stats["cvar_95"] <= stats["var_95"]
