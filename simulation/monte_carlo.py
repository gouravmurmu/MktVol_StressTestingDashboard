from typing import Optional

import numpy as np
import pandas as pd


class MonteCarloSimulator:
    """Price path simulators. All return arrays shaped (T + 1, n_sims) starting at `last_price`."""

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)

    @staticmethod
    def _to_prices(last_price: float, log_returns: np.ndarray) -> np.ndarray:
        cum = np.vstack([np.zeros((1, log_returns.shape[1])), np.cumsum(log_returns, axis=0)])
        return last_price * np.exp(cum)

    def simulate_gbm(self, last_price: float, mu: float, sigma: float, T: int,
                     dt: float = 1 / 252, n_sims: int = 1000) -> np.ndarray:
        """Geometric Brownian Motion with annualized drift `mu` and volatility `sigma`."""
        z = self.rng.standard_normal((T, n_sims))
        log_r = (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z
        return self._to_prices(last_price, log_r)

    def simulate_student_t(self, last_price: float, mu: float, sigma: float, T: int,
                           df: float = 4.0, dt: float = 1 / 252, n_sims: int = 1000) -> np.ndarray:
        """GBM with Student-t shocks (unit variance) for fatter tails."""
        z = self.rng.standard_t(df, (T, n_sims)) / np.sqrt(df / (df - 2))
        log_r = (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z
        return self._to_prices(last_price, log_r)

    def simulate_bootstrap(self, last_price: float, returns: pd.Series, T: int,
                           n_sims: int = 1000, block: int = 5) -> np.ndarray:
        """Stationary block bootstrap of historical log returns, which keeps volatility clustering."""
        r = returns.dropna().values
        n = len(r)
        n_blocks = int(np.ceil(T / block))
        starts = self.rng.integers(0, max(1, n - block), size=(n_blocks, n_sims))
        offsets = np.arange(block)
        idx = (starts[:, None, :] + offsets[None, :, None]).reshape(n_blocks * block, n_sims)[:T]
        return self._to_prices(last_price, r[idx])

    def simulate_garch(self, last_price: float, returns: pd.Series, T: int, n_sims: int = 1000) -> np.ndarray:
        """Simulates from a fitted GJR-GARCH(1,1)-t, so volatility evolves along each path."""
        import warnings

        from arch import arch_model

        r = returns.dropna() * 100
        am = arch_model(r, mean="Constant", vol="GARCH", p=1, o=1, q=1, dist="t")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = am.fit(disp="off", show_warning=False)
            fc = res.forecast(horizon=T, method="simulation", simulations=n_sims, reindex=False)
        sims = fc.simulations.values[-1] / 100  # (n_sims, T)
        return self._to_prices(last_price, sims.T)

    @staticmethod
    def calculate_stats(final_prices: np.ndarray, initial_price: float) -> dict:
        returns = final_prices / initial_price - 1
        var95 = np.percentile(returns, 5)
        return {
            "mean_return": float(np.mean(returns)),
            "median_return": float(np.median(returns)),
            "prob_loss": float(np.mean(returns < 0)),
            "var_95": float(var95),
            "var_99": float(np.percentile(returns, 1)),
            "cvar_95": float(returns[returns <= var95].mean()),
            "worst_case": float(np.min(returns)),
            "best_case": float(np.max(returns)),
        }
