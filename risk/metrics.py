import numpy as np
import pandas as pd
from scipy.stats import norm

class RiskMetrics:
    def __init__(self):
        pass

    def calculate_volatility(self, returns: pd.Series, window: int = 30, annualized: bool = True) -> pd.Series:
        """
        Calculates rolling standard deviation.
        """
        scale_factor = np.sqrt(252) if annualized else 1.0
        return returns.rolling(window=window).std() * scale_factor

    def calculate_ewma_volatility(self, returns: pd.Series, lambda_param: float = 0.94, annualized: bool = True) -> pd.Series:
        """
        Calculates Exponentially Weighted Moving Average (EWMA) volatility.
        RiskMetrics standard lambda is 0.94.
        """
        scale_factor = np.sqrt(252) if annualized else 1.0
        # Calculate EWMA variance
        var_ewma = returns.ewm(alpha=1-lambda_param, adjust=False).var()
        return np.sqrt(var_ewma) * scale_factor

    def calculate_var_historical(self, returns: pd.Series, confidence_level: float = 0.95) -> float:
        """
        Calculates Historical Value at Risk.
        Metric is positive (loss value).
        """
        if returns.empty: return 0.0
        # Quantile is 1 - confidence (e.g., 5% for 95% confidence)
        var = -np.percentile(returns, (1 - confidence_level) * 100)
        return max(0.0, var)

    def calculate_var_parametric(self, returns: pd.Series, confidence_level: float = 0.95) -> float:
        """
        Calculates Parametric (Gaussian) Value at Risk.
        Average return is typically assumed to be 0 for daily VaR, but we can include mean.
        """
        if returns.empty: return 0.0
        mu = returns.mean()
        sigma = returns.std()
        
        # Z-score for cut-off
        z_score = norm.ppf(1 - confidence_level)
        var = -(mu + z_score * sigma)
        return max(0.0, var)

    def calculate_cvar(self, returns: pd.Series, confidence_level: float = 0.95) -> float:
        """
        Calculates Conditional Value at Risk (Expected Shortfall).
        Average loss strictly exceeding the VaR.
        """
        if returns.empty: return 0.0
        cutoff = np.percentile(returns, (1 - confidence_level) * 100)
        tail_losses = returns[returns <= cutoff]
        if tail_losses.empty:
            return 0.0
        cvar = -tail_losses.mean()
        return max(0.0, cvar)
