import logging
import warnings

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class RegimeDetector:
    def detect_regime(self, prices: pd.Series, short_window: int = 50, long_window: int = 200) -> pd.DataFrame:
        """
        Trend regime from an SMA crossover. Bull (1) if short > long, Bear (-1) otherwise.
        """
        signals = pd.DataFrame(index=prices.index)
        signals["Short_MA"] = prices.rolling(window=short_window, min_periods=1).mean()
        signals["Long_MA"] = prices.rolling(window=long_window, min_periods=1).mean()
        signals["Regime"] = np.where(signals["Short_MA"] > signals["Long_MA"], 1.0, -1.0)
        return signals

    def detect_volatility_regime(self, returns: pd.Series, k_regimes: int = 2) -> pd.DataFrame:
        """
        Markov-switching model (Hamilton 1989) with regime-dependent mean and variance.
        Returns smoothed probabilities per regime, ordered from calmest to most turbulent,
        plus the most likely regime label each day.
        """
        from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

        r = returns.dropna() * 100
        model = MarkovRegression(r, k_regimes=k_regimes, trend="c", switching_variance=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = model.fit(disp=False, search_reps=5)

        sigmas = np.array([res.params[f"sigma2[{i}]"] for i in range(k_regimes)])
        order = np.argsort(sigmas)
        probs = res.smoothed_marginal_probabilities
        probs = pd.DataFrame(np.asarray(probs)[:, order], index=r.index)
        names = ["Calm", "Turbulent"] if k_regimes == 2 else ["Calm", "Normal", "Turbulent"][:k_regimes]
        probs.columns = names
        probs["Regime"] = probs[names].idxmax(axis=1)

        trans = np.asarray(res.regime_transition)[:, :, 0] if np.asarray(res.regime_transition).ndim == 3 \
            else np.asarray(res.regime_transition)
        trans = trans[np.ix_(order, order)]
        stay = np.diag(trans)
        probs.attrs["summary"] = pd.DataFrame({
            "Annual Vol": np.sqrt(sigmas[order]) / 100 * np.sqrt(252),
            "Expected Duration (days)": 1 / np.clip(1 - stay, 1e-6, None),
        }, index=names)
        return probs

    @staticmethod
    def get_regime_stats(returns: pd.Series, regimes: pd.Series, periods: int = 252) -> pd.DataFrame:
        """Annualized return/vol per regime."""
        df = pd.DataFrame({"Returns": returns, "Regime": regimes}).dropna()
        stats = df.groupby("Regime")["Returns"].agg(["mean", "std", "count"])
        stats["mean"] *= periods
        stats["std"] *= np.sqrt(periods)
        return stats.rename(columns={"mean": "Annual Return", "std": "Annual Vol", "count": "Days"})
