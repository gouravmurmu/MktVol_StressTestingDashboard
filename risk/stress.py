import pandas as pd
import numpy as np
from scipy.stats import t

class StressTester:
    def __init__(self):
        pass

    def simulate_shock(self, returns: pd.Series, 
                       shock_type: str = "volatility", 
                       magnitude: float = 2.0) -> pd.Series:
        """
        Applies a synthetic shock to the historical returns.
        
        shock_type options:
        - 'volatility': Multiply returns by magnitude (e.g., 2x daily moves).
        - 'drift': Add negative drift (magnitude as annualized % drop).
        - 'crash': Replay a specific historical drawdown shape (custom logic needed, simplified here as percentage drop).
        """
        stressed_returns = returns.copy()

        if shock_type == "volatility":
            # Simple scalar multiplication of volatility
            # We assume mean is close to 0, so just scaling returns works
            stressed_returns = returns * magnitude

        elif shock_type == "drift":
            # Magnitude is annualized negative drift (e.g. -0.20 for -20%)
            # Convert to daily drift
            daily_drift = magnitude / 252.0
            stressed_returns = returns + daily_drift

        return stressed_returns

    def generate_heavy_tails(self, n_samples: int, df: int = 3, vol: float = 0.20) -> np.ndarray:
        """
        Generates synthetic Student-t distributed returns (fatter tails than Normal).
        Used for Monte Carlo or "What-if" distributions.
        """
        # Student-t standardized
        daily_vol = vol / np.sqrt(252)
        random_t = t.rvs(df, size=n_samples) 
        # Scale to desired vol. t-dist variance is df/(df-2), so we adjust
        std_t = np.sqrt(df / (df - 2))
        scaled_returns = (random_t / std_t) * daily_vol
        return scaled_returns
