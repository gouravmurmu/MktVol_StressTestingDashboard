import numpy as np
import pandas as pd
from scipy.stats import chi2, norm, t


class RiskMetrics:
    """Risk statistics on a return series. VaR/CVaR are reported as positive loss fractions."""

    def __init__(self, periods: int = 252):
        self.periods = periods

    # ---------- Volatility ----------
    def calculate_volatility(self, returns: pd.Series, window: int = 30, annualized: bool = True) -> pd.Series:
        """Rolling standard deviation."""
        scale = np.sqrt(self.periods) if annualized else 1.0
        return returns.rolling(window=window).std() * scale

    def calculate_ewma_volatility(self, returns: pd.Series, lambda_param: float = 0.94,
                                  annualized: bool = True) -> pd.Series:
        """RiskMetrics EWMA volatility (lambda 0.94 by default)."""
        scale = np.sqrt(self.periods) if annualized else 1.0
        var_ewma = (returns ** 2).ewm(alpha=1 - lambda_param, adjust=False).mean()
        return np.sqrt(var_ewma) * scale

    # ---------- Value at Risk ----------
    def calculate_var_historical(self, returns: pd.Series, confidence_level: float = 0.95) -> float:
        if returns.empty:
            return 0.0
        return max(0.0, -np.percentile(returns, (1 - confidence_level) * 100))

    def calculate_var_parametric(self, returns: pd.Series, confidence_level: float = 0.95) -> float:
        """Gaussian VaR."""
        if returns.empty:
            return 0.0
        z = norm.ppf(1 - confidence_level)
        return max(0.0, -(returns.mean() + z * returns.std()))

    def calculate_var_student_t(self, returns: pd.Series, confidence_level: float = 0.95) -> float:
        """VaR from a Student-t fit, which captures fat tails."""
        if len(returns) < 10:
            return 0.0
        df, loc, scale = t.fit(returns)
        return max(0.0, -t.ppf(1 - confidence_level, df, loc, scale))

    def calculate_var_cornish_fisher(self, returns: pd.Series, confidence_level: float = 0.95) -> float:
        """Modified VaR adjusting the Gaussian quantile for skew and excess kurtosis."""
        if len(returns) < 10:
            return 0.0
        z = norm.ppf(1 - confidence_level)
        s, k = returns.skew(), returns.kurtosis()
        z_cf = (z + (z ** 2 - 1) * s / 6 + (z ** 3 - 3 * z) * k / 24 - (2 * z ** 3 - 5 * z) * s ** 2 / 36)
        return max(0.0, -(returns.mean() + z_cf * returns.std()))

    def calculate_cvar(self, returns: pd.Series, confidence_level: float = 0.95) -> float:
        """Expected Shortfall: average loss beyond the historical VaR."""
        if returns.empty:
            return 0.0
        cutoff = np.percentile(returns, (1 - confidence_level) * 100)
        tail = returns[returns <= cutoff]
        return 0.0 if tail.empty else max(0.0, -tail.mean())

    def var_table(self, returns: pd.Series, confidence_level: float = 0.95) -> pd.DataFrame:
        """All VaR methods side by side."""
        rows = {
            "Historical": self.calculate_var_historical(returns, confidence_level),
            "Gaussian": self.calculate_var_parametric(returns, confidence_level),
            "Student-t": self.calculate_var_student_t(returns, confidence_level),
            "Cornish-Fisher": self.calculate_var_cornish_fisher(returns, confidence_level),
        }
        return pd.DataFrame({"VaR": rows})

    def rolling_var(self, returns: pd.Series, window: int = 250, confidence_level: float = 0.99) -> pd.Series:
        """Rolling historical VaR, shifted one day so it is a true out-of-sample forecast."""
        q = returns.rolling(window).quantile(1 - confidence_level)
        return (-q).shift(1)

    # ---------- Performance & drawdown ----------
    @staticmethod
    def drawdown(returns: pd.Series) -> pd.Series:
        wealth = np.exp(returns.cumsum())
        return wealth / wealth.cummax() - 1

    def max_drawdown(self, returns: pd.Series) -> float:
        return float(self.drawdown(returns).min()) if not returns.empty else 0.0

    def sharpe(self, returns: pd.Series, rf: float = 0.0) -> float:
        vol = returns.std() * np.sqrt(self.periods)
        return float((returns.mean() * self.periods - rf) / vol) if vol > 0 else 0.0

    def sortino(self, returns: pd.Series, rf: float = 0.0) -> float:
        downside = returns[returns < 0].std() * np.sqrt(self.periods)
        return float((returns.mean() * self.periods - rf) / downside) if downside > 0 else 0.0

    def calmar(self, returns: pd.Series) -> float:
        mdd = abs(self.max_drawdown(returns))
        return float(returns.mean() * self.periods / mdd) if mdd > 0 else 0.0

    @staticmethod
    def beta(asset: pd.Series, benchmark: pd.Series) -> float:
        df = pd.concat([asset, benchmark], axis=1).dropna()
        if len(df) < 2 or df.iloc[:, 1].var() == 0:
            return 1.0
        return float(df.cov().iloc[0, 1] / df.iloc[:, 1].var())

    # ---------- Backtesting ----------
    @staticmethod
    def kupiec_test(returns: pd.Series, var_forecast: pd.Series, confidence_level: float) -> dict:
        """
        Kupiec proportion-of-failures test. A p-value below 0.05 means the VaR model's
        breach rate is statistically inconsistent with the chosen confidence level.
        """
        df = pd.concat([returns, var_forecast], axis=1, keys=["r", "var"]).dropna()
        n = len(df)
        if n == 0:
            return {"observations": 0, "breaches": 0, "expected": 0.0, "breach_rate": 0.0, "p_value": np.nan}
        breaches = int((df["r"] < -df["var"]).sum())
        p = 1 - confidence_level
        rate = breaches / n
        if breaches in (0, n):
            lr = -2 * (n * np.log(1 - p) if breaches == 0 else n * np.log(p))
        else:
            lr = -2 * ((n - breaches) * np.log(1 - p) + breaches * np.log(p)
                       - (n - breaches) * np.log(1 - rate) - breaches * np.log(rate))
        return {
            "observations": n,
            "breaches": breaches,
            "expected": n * p,
            "breach_rate": rate,
            "p_value": float(1 - chi2.cdf(lr, 1)),
        }
