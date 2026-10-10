from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import t


@dataclass(frozen=True)
class Scenario:
    name: str
    market_shock: float      # benchmark peak-to-trough move, e.g. -0.55
    vol_multiplier: float    # how much daily volatility rises during the episode
    duration_days: int       # trading days from peak to trough
    description: str


# Approximate S&P 500 peak-to-trough moves for well known stress episodes.
HISTORICAL_SCENARIOS: List[Scenario] = [
    Scenario("2008 Global Financial Crisis", -0.55, 3.0, 355,
             "Lehman collapse and credit freeze. S&P 500 fell ~55% from Oct 2007 to Mar 2009."),
    Scenario("2020 COVID Crash", -0.34, 4.0, 23,
             "Fastest bear market on record. S&P 500 fell ~34% in 23 trading days."),
    Scenario("2000-02 Dot-com Bust", -0.49, 1.8, 640,
             "Tech bubble unwinds. S&P 500 fell ~49% over two and a half years."),
    Scenario("1987 Black Monday", -0.20, 6.0, 1,
             "Single-day crash of ~20% in the S&P 500."),
    Scenario("2022 Rate Shock", -0.25, 1.6, 195,
             "Fastest Fed hiking cycle in decades. S&P 500 fell ~25% in 2022."),
    Scenario("2011 Euro Debt Crisis", -0.19, 2.2, 110,
             "US downgrade and European sovereign stress. S&P 500 fell ~19%."),
]


class StressTester:
    def simulate_shock(self, returns: pd.Series, shock_type: str = "volatility",
                       magnitude: float = 2.0) -> pd.Series:
        """
        Applies a synthetic shock to historical returns.
        - 'volatility': scales deviations from the mean by `magnitude`.
        - 'drift': adds an annualized drift of `magnitude` (e.g. -0.30).
        """
        if shock_type == "volatility":
            mu = returns.mean()
            return mu + (returns - mu) * magnitude
        if shock_type == "drift":
            return returns + magnitude / 252.0
        return returns.copy()

    @staticmethod
    def scenario_impact(betas: Dict[str, float], weights: Dict[str, float], scenario: Scenario) -> pd.DataFrame:
        """
        Projects a benchmark shock onto each holding through its beta:
        asset move = beta * market move, floored at -100%.
        """
        rows = []
        for asset, w in weights.items():
            b = betas.get(asset, 1.0)
            move = max(-1.0, b * scenario.market_shock)
            rows.append({"Asset": asset, "Weight": w, "Beta": b, "Shocked Return": move, "Contribution": w * move})
        df = pd.DataFrame(rows).set_index("Asset")
        return df

    @staticmethod
    def scenario_path(scenario: Scenario, daily_vol: float, beta: float = 1.0,
                      n_paths: int = 200, recovery_days: int = 60, seed: int = 42) -> np.ndarray:
        """
        Simulates a stressed price path (starting at 1.0) that reaches the scenario's
        drawdown over its duration with elevated volatility, followed by a recovery window.
        Returns an array of shape (duration + recovery + 1, n_paths).
        """
        rng = np.random.default_rng(seed)
        d = max(1, scenario.duration_days)
        target = np.log1p(max(-0.99, beta * scenario.market_shock))
        stressed_vol = daily_vol * scenario.vol_multiplier
        noise = rng.standard_normal((d, n_paths)) * stressed_vol
        # Brownian bridge: noise that sums to zero, so every path hits the target drawdown.
        noise -= noise.mean(axis=0, keepdims=True)
        crash = target / d + noise
        recovery = rng.standard_normal((recovery_days, n_paths)) * daily_vol * max(1.0, scenario.vol_multiplier / 2)
        log_path = np.vstack([np.zeros((1, n_paths)), np.cumsum(np.vstack([crash, recovery]), axis=0)])
        return np.exp(log_path)

    @staticmethod
    def reverse_stress(betas: Dict[str, float], weights: Dict[str, float], loss_threshold: float) -> float:
        """Market move needed for the portfolio to lose `loss_threshold` (e.g. 0.20 for -20%)."""
        port_beta = sum(weights[a] * betas.get(a, 1.0) for a in weights)
        if port_beta <= 0:
            return np.nan
        return -loss_threshold / port_beta

    @staticmethod
    def generate_heavy_tails(n_samples: int, df: int = 3, vol: float = 0.20, seed: int = None) -> np.ndarray:
        """Student-t returns scaled to an annualized volatility."""
        rng = np.random.default_rng(seed)
        daily_vol = vol / np.sqrt(252)
        draws = t.rvs(df, size=n_samples, random_state=rng)
        return draws / np.sqrt(df / (df - 2)) * daily_vol
