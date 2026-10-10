import logging
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TRADING_DAYS = 252

# (beta to the market factor, idiosyncratic annual vol) for common tickers, so demo data
# shows realistic diversification. Unknown tickers get a random equity-like profile.
DEMO_PROFILES = {
    "SPY": (1.0, 0.02), "VOO": (1.0, 0.02), "IVV": (1.0, 0.02), "QQQ": (1.2, 0.08), "IWM": (1.15, 0.10),
    "DIA": (0.95, 0.04), "TLT": (-0.25, 0.13), "IEF": (-0.15, 0.06), "AGG": (-0.05, 0.04),
    "GLD": (0.05, 0.14), "SLV": (0.25, 0.25), "USO": (0.6, 0.30), "VNQ": (1.0, 0.12), "EEM": (0.9, 0.12),
    "AAPL": (1.2, 0.20), "MSFT": (1.1, 0.18), "NVDA": (1.7, 0.35), "TSLA": (1.8, 0.45), "AMZN": (1.25, 0.22),
}


class MarketDataLoader:
    """Fetches prices from Yahoo Finance, with a synthetic fallback for offline/demo use."""

    def fetch_data(self, tickers: List[str], start_date: str, end_date: Optional[str] = None,
                   interval: str = "1d") -> pd.DataFrame:
        """
        Fetches historical prices for the given tickers.
        Returns a DataFrame of (dividend/split adjusted) close prices, one column per ticker.
        """
        tickers = [t.strip().upper() for t in tickers if t and t.strip()]
        if not tickers:
            return pd.DataFrame()

        import yfinance as yf

        # auto_adjust=True makes 'Close' the adjusted close. Newer yfinance versions no longer
        # return an 'Adj Close' column by default, which broke the previous implementation.
        data = yf.download(tickers, start=start_date, end=end_date, interval=interval,
                           auto_adjust=True, progress=False, group_by="column")
        return self._extract_close(data, tickers)

    @staticmethod
    def _extract_close(data: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
        if data is None or data.empty:
            return pd.DataFrame()

        if isinstance(data.columns, pd.MultiIndex):
            level0 = data.columns.get_level_values(0)
            field = "Close" if "Close" in level0 else "Adj Close"
            prices = data[field]
        else:
            field = "Close" if "Close" in data.columns else "Adj Close"
            prices = data[[field]].rename(columns={field: tickers[0]})

        if isinstance(prices, pd.Series):
            prices = prices.to_frame(name=tickers[0])

        prices = prices.dropna(axis=1, how="all").dropna()
        prices.columns = [str(c) for c in prices.columns]
        prices.index = pd.to_datetime(prices.index).tz_localize(None)
        return prices

    @staticmethod
    def generate_demo_data(tickers: List[str], start_date: str, end_date: Optional[str] = None,
                           seed: int = 7) -> pd.DataFrame:
        """
        Generates realistic synthetic prices: a common market factor with regime-switching
        volatility (calm/stressed) plus idiosyncratic noise per asset. Used when the
        network is unavailable or when 'Demo data' is selected.
        """
        rng = np.random.default_rng(seed)
        idx = pd.bdate_range(start=start_date, end=end_date or pd.Timestamp.today().normalize())
        n = len(idx)
        if n < 2 or not tickers:
            return pd.DataFrame()

        # Two-state Markov chain for market regime: 0 = calm, 1 = stressed
        p_stay = np.array([0.99, 0.95])
        state = np.zeros(n, dtype=int)
        for i in range(1, n):
            stay = rng.random() < p_stay[state[i - 1]]
            state[i] = state[i - 1] if stay else 1 - state[i - 1]

        mkt_vol = np.where(state == 0, 0.12, 0.38) / np.sqrt(TRADING_DAYS)
        mkt_mu = np.where(state == 0, 0.12, -0.25) / TRADING_DAYS
        market = mkt_mu + mkt_vol * rng.standard_t(5, n) / np.sqrt(5 / 3)

        prices = {}
        for i, ticker in enumerate(tickers):
            t_rng = np.random.default_rng(seed + 101 * (i + 1) + sum(map(ord, ticker)))
            beta, idio = DEMO_PROFILES.get(ticker, (t_rng.uniform(0.6, 1.4), t_rng.uniform(0.10, 0.25)))
            idio_vol = idio / np.sqrt(TRADING_DAYS)
            alpha = t_rng.uniform(-0.03, 0.05) / TRADING_DAYS
            rets = alpha + beta * market + idio_vol * t_rng.standard_normal(n)
            rets[0] = 0.0
            prices[ticker] = t_rng.uniform(40, 400) * np.exp(np.cumsum(rets))

        return pd.DataFrame(prices, index=idx)

    @staticmethod
    def calculate_returns(prices: pd.DataFrame, log_returns: bool = True) -> pd.DataFrame:
        """Calculates log or simple returns from prices."""
        if log_returns:
            returns = np.log(prices / prices.shift(1))
        else:
            returns = prices.pct_change()
        return returns.dropna(how="all").dropna()

    @staticmethod
    def get_asset_stats(returns: pd.DataFrame, periods: int = TRADING_DAYS) -> pd.DataFrame:
        """Annualized return/volatility, Sharpe, skew and excess kurtosis per asset."""
        stats = pd.DataFrame(index=returns.columns)
        stats["Annual Return"] = returns.mean() * periods
        stats["Annual Volatility"] = returns.std() * np.sqrt(periods)
        stats["Sharpe"] = stats["Annual Return"] / stats["Annual Volatility"].replace(0, np.nan)
        stats["Skewness"] = returns.skew()
        stats["Excess Kurtosis"] = returns.kurtosis()
        return stats


def periods_per_year(interval: str) -> int:
    return {"1d": 252, "1wk": 52, "1mo": 12}.get(interval, 252)
