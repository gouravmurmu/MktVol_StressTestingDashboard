import sys
import os

# Add current dir to path so we can import modules
sys.path.append(os.getcwd())

from data.loader import MarketDataLoader
from risk.metrics import RiskMetrics
from simulation.monte_carlo import MonteCarloSimulator
from regime.detector import RegimeDetector
import pandas as pd

def test_pipeline():
    print("1. Testing Data Loader...")
    loader = MarketDataLoader()
    # Use a safe ticker and short duration
    try:
        prices = loader.fetch_data(["SPY"], "2023-01-01", "2023-06-01")
        if prices.empty:
            print("WARNING: No data fetched. Check internet or ticker.")
            return
        returns = loader.calculate_returns(prices)
        print(f"   Data Fetched: {len(prices)} rows.")
    except Exception as e:
        print(f"   FAILED: {e}")
        return

    print("2. Testing Risk Metrics...")
    metrics = RiskMetrics()
    try:
        vol = metrics.calculate_volatility(returns).iloc[-1].item()
        var = metrics.calculate_var_historical(returns['SPY'])
        print(f"   Vol (Ann): {vol:.2%}")
        print(f"   VaR (95%): {var:.2%}")
    except Exception as e:
        print(f"   FAILED: {e}")
        return

    print("3. Testing Simulation...")
    sim = MonteCarloSimulator()
    try:
        last_price = prices.iloc[-1].item()
        paths = sim.simulate_gbm(last_price, 0.07, 0.15, 30, n_sims=10)
        print(f"   Simulated 30 days. Final mean price: {paths[-1].mean():.2f}")
    except Exception as e:
        # Check if it's just a numpy/pandas version mismatch or shape issue
        print(f"   FAILED: {e}")
        return
        
    print("4. Testing Regime Detection...")
    regime = RegimeDetector()
    try:
        signals = regime.detect_regime(prices['SPY'], short_window=10, long_window=30)
        curr = signals.iloc[-1]['Regime']
        print(f"   Regime detected: {curr}")
    except Exception as e:
        print(f"   FAILED: {e}")
        return

    print("SUCCESS: All core modules verified.")

if __name__ == "__main__":
    test_pipeline()
