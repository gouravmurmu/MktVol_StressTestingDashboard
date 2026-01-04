import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, Union, Dict

class MarketDataLoader:
    def __init__(self):
        pass

    def fetch_data(self, tickers: List[str], start_date: str, end_date: str = None, interval: str = "1d") -> pd.DataFrame:
        """
        Fetches historical data for given tickers.
        Returns a DataFrame of Adjusted Close prices.
        """
        if not tickers:
            return pd.DataFrame()
        
        # yfinance expects space-separated string for multiple tickers or list
        data = yf.download(tickers, start=start_date, end=end_date, interval=interval, progress=False)
        
        # Handle multi-index columns if multiple tickers
        if len(tickers) > 1:
            prices = data['Adj Close']
        else:
             # If single ticker, it might come as Series or DF depending on yfinance version
            prices = data['Adj Close'] if 'Adj Close' in data else data
            if isinstance(prices, pd.Series):
                 prices = prices.to_frame(name=tickers[0])
            
        return prices.dropna()

    def calculate_returns(self, prices: pd.DataFrame, log_returns: bool = True) -> pd.DataFrame:
        """
        Calculates returns (log or simple) from prices.
        """
        if log_returns:
            returns = np.log(prices / prices.shift(1))
        else:
            returns = prices.pct_change()
            
        return returns.dropna()

    def get_asset_stats(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Returns basic stats: Annualized Return, Volatility, Skew, Kurtosis.
        Assuming daily data (252 days).
        """
        stats = pd.DataFrame(index=returns.columns)
        stats['Annual Return'] = returns.mean() * 252
        stats['Annual Volatility'] = returns.std() * np.sqrt(252)
        stats['Skewness'] = returns.skew()
        stats['Kurtosis'] = returns.kurtosis()
        return stats
