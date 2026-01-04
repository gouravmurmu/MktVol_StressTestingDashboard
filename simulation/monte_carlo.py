import numpy as np
import pandas as pd

class MonteCarloSimulator:
    def __init__(self):
        pass

    def simulate_gbm(self, last_price: float, mu: float, sigma: float, T: int, dt: float = 1/252, n_sims: int = 1000) -> np.ndarray:
        """
        Simulate Geometric Brownian Motion.
        Returns: (T+1, n_sims) array of projected prices.
        """
        # S_t = S_0 * exp((mu - 0.5*sigma^2)*t + sigma*W_t)
        
        # Increments: (mu - 0.5 * sigma^2) * dt + sigma * sqrt(dt) * Z
        # Z ~ N(0, 1)
        
        # Drift component (constant)
        drift = (mu - 0.5 * sigma**2) * dt
        
        # Stochastic component
        # Generate N(0,1) matrix
        Z = np.random.normal(0, 1, (T, n_sims))
        
        shock = sigma * np.sqrt(dt) * Z
        
        # Log returns
        log_returns = drift + shock
        
        # Accumulate log returns
        cum_log_returns = np.cumsum(log_returns, axis=0)
        
        # Prepend 0 for the start
        cum_log_returns = np.vstack([np.zeros((1, n_sims)), cum_log_returns])
        
        # Convert to prices
        simulated_prices = last_price * np.exp(cum_log_returns)
        
        return simulated_prices
    
    def calculate_stats(self, final_prices: np.ndarray, initial_price: float) -> dict:
        """
        Calculate summary stats for the simulation.
        """
        returns = (final_prices - initial_price) / initial_price
        return {
            "mean_return": np.mean(returns),
            "median_return": np.median(returns),
            "var_95": np.percentile(returns, 5),    # 5th percentile
            "var_99": np.percentile(returns, 1),    # 1st percentile
            "worst_case": np.min(returns),
            "best_case": np.max(returns)
        }
