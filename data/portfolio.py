from typing import Dict

import numpy as np
import pandas as pd


class PortfolioManager:
    @staticmethod
    def normalize_weights(columns, weights: Dict[str, float]) -> Dict[str, float]:
        """Normalizes weights to sum to 1. Falls back to equal weight if all are zero."""
        clean = {c: max(float(weights.get(c, 0.0)), 0.0) for c in columns}
        total = sum(clean.values())
        if total == 0:
            return {c: 1.0 / len(columns) for c in columns}
        return {c: w / total for c, w in clean.items()}

    def construct_portfolio(self, returns: pd.DataFrame, weights: Dict[str, float]) -> pd.Series:
        """
        Constructs a daily-rebalanced portfolio return series from asset log returns.
        Log returns are converted to simple returns for aggregation, then back to log.
        """
        if returns.empty:
            return pd.Series(dtype=float, name="Portfolio")

        w = self.normalize_weights(returns.columns, weights)
        weight_vector = np.array([w[c] for c in returns.columns])
        simple = np.expm1(returns)
        port = np.log1p(simple.dot(weight_vector))
        port.name = "Portfolio"
        return port

    @staticmethod
    def risk_contributions(returns: pd.DataFrame, weights: Dict[str, float]) -> pd.Series:
        """Percentage contribution of each asset to total portfolio variance."""
        w = np.array([weights[c] for c in returns.columns])
        cov = returns.cov().values
        port_var = w @ cov @ w
        if port_var <= 0:
            return pd.Series(0.0, index=returns.columns)
        contrib = w * (cov @ w) / port_var
        return pd.Series(contrib, index=returns.columns)
