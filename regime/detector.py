import pandas as pd
import numpy as np

class RegimeDetector:
    def __init__(self):
        pass

    def detect_regime(self, prices: pd.Series, short_window: int = 50, long_window: int = 200) -> pd.DataFrame:
        """
        Detects market regime based on SMA Crossover.
        Bull (1) if Short > Long.
        Bear (-1) if Short < Long.
        Returns DataFrame with 'Regime' column (-1, 1).
        """
        signals = pd.DataFrame(index=prices.index)
        signals['Signal'] = 0.0
        
        # Create rolling means
        signals['Short_MA'] = prices.rolling(window=short_window, min_periods=1, center=False).mean()
        signals['Long_MA'] = prices.rolling(window=long_window, min_periods=1, center=False).mean()
        
        # Signal: 1 if Short > Long, -1 otherwise (technically 0 in boolean but we map to regimes)
        signals['Regime'] = np.where(signals['Short_MA'] > signals['Long_MA'], 1.0, -1.0)
        
        return signals

    def get_regime_stats(self, returns: pd.Series, regimes: pd.Series) -> pd.DataFrame:
        """
        Calculates volatility/return stats per regime.
        """
        df = pd.DataFrame({'Returns': returns, 'Regime': regimes}).dropna()
        stats = df.groupby('Regime')['Returns'].agg(['mean', 'std', 'count'])
        # Annualize
        stats['mean'] = stats['mean'] * 252
        stats['std'] = stats['std'] * np.sqrt(252)
        return stats
