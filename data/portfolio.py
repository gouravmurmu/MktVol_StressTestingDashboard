import pandas as pd
import numpy as np
from typing import Dict, List

class PortfolioManager:
    def __init__(self):
        pass

    def construct_portfolio(self, returns: pd.DataFrame, weights: Dict[str, float]) -> pd.Series:
        """
        Constructs a portfolio return series based on weights.
        Returns: pd.Series of portfolio returns.
        """
        if returns.empty:
            return pd.Series(dtype=float)

        # Ensure weights match the columns in returns
        # Normalize weights to sum to 1 if they don't
        total_weight = sum(weights.values())
        if total_weight == 0:
             # Default to equal weight if invalid
             normalized_weights = {col: 1.0/len(returns.columns) for col in returns.columns}
        else:
             normalized_weights = {k: v/total_weight for k, v in weights.items()}

        # Create a weight vector aligned with columns
        weight_vector = np.array([normalized_weights.get(col, 0.0) for col in returns.columns])
        
        # Calculate weighted sum
        portfolio_returns = returns.dot(weight_vector)
        portfolio_returns.name = "Portfolio"
        
        return portfolio_returns
