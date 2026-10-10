"""
Volatility forecasting models.

All models work on daily log returns. Internally returns are scaled to percent, which
keeps the GARCH optimizer well conditioned. Outputs are returned as decimal volatility.
"""
import logging
import warnings
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

GARCH_SPECS = {
    "GARCH(1,1)": dict(vol="GARCH", p=1, o=0, q=1),
    "GJR-GARCH(1,1)": dict(vol="GARCH", p=1, o=1, q=1),
    "EGARCH(1,1)": dict(vol="EGARCH", p=1, o=1, q=1),
}


@dataclass
class VolForecast:
    model: str
    daily_vol_path: np.ndarray       # forecast daily vol for each step of the horizon
    params: Dict[str, float]
    aic: Optional[float] = None
    persistence: Optional[float] = None

    def annualized(self, periods: int = 252) -> np.ndarray:
        return self.daily_vol_path * np.sqrt(periods)


def _arch_model(returns_pct: pd.Series, spec: dict, dist: str):
    from arch import arch_model
    return arch_model(returns_pct, mean="Constant", dist=dist, **spec)


def fit_garch(returns: pd.Series, model: str = "GJR-GARCH(1,1)", dist: str = "t",
              horizon: int = 21) -> VolForecast:
    """Fits a GARCH-family model and forecasts daily volatility over `horizon` steps."""
    r = returns.dropna() * 100
    am = _arch_model(r, GARCH_SPECS[model], dist)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = am.fit(disp="off", show_warning=False)
    # EGARCH has no closed-form multi-step forecast, so it is simulated.
    method = "simulation" if model.startswith("EGARCH") and horizon > 1 else "analytic"
    fc = res.forecast(horizon=horizon, reindex=False, method=method)
    path = np.sqrt(fc.variance.values[-1]) / 100
    params = {k: float(v) for k, v in res.params.items()}
    persistence = params.get("alpha[1]", 0) + params.get("beta[1]", 0) + 0.5 * params.get("gamma[1]", 0)
    if model.startswith("EGARCH"):
        persistence = params.get("beta[1]", np.nan)
    return VolForecast(model, path, params, float(res.aic), float(persistence))


def ewma_forecast(returns: pd.Series, lam: float = 0.94, horizon: int = 21) -> VolForecast:
    """RiskMetrics EWMA. The forecast is flat across the horizon by construction."""
    var = float((returns.dropna() ** 2).ewm(alpha=1 - lam, adjust=False).mean().iloc[-1])
    return VolForecast("EWMA (λ=0.94)", np.full(horizon, np.sqrt(var)), {"lambda": lam})


def _realized_vol(returns: pd.Series, window: int) -> pd.Series:
    return np.sqrt((returns ** 2).rolling(window).mean())


def har_features(returns: pd.Series) -> pd.DataFrame:
    """Heterogeneous AutoRegressive (Corsi 2009) features: daily, weekly and monthly realized vol."""
    return pd.DataFrame({
        "rv_d": returns.abs(),
        "rv_w": _realized_vol(returns, 5),
        "rv_m": _realized_vol(returns, 22),
    })


def har_forecast(returns: pd.Series, horizon: int = 21) -> VolForecast:
    """HAR-RV regression of future `horizon`-day realized vol on lagged daily/weekly/monthly RV."""
    r = returns.dropna()
    X = har_features(r)
    target = _realized_vol(r, horizon).shift(-horizon)
    df = pd.concat([X, target.rename("y")], axis=1).dropna()
    if len(df) < 60:
        return ewma_forecast(r, horizon=horizon)
    A = np.column_stack([np.ones(len(df)), df[["rv_d", "rv_w", "rv_m"]].values])
    coef, *_ = np.linalg.lstsq(A, df["y"].values, rcond=None)
    x_last = np.concatenate([[1.0], X.iloc[-1].values])
    pred = max(float(x_last @ coef), 1e-6)
    params = dict(zip(["const", "beta_d", "beta_w", "beta_m"], map(float, coef)))
    return VolForecast("HAR-RV", np.full(horizon, pred), params)


def forecast_all(returns: pd.Series, horizon: int = 21) -> Dict[str, VolForecast]:
    """Runs every model, skipping any that fail to converge."""
    out = {"EWMA (λ=0.94)": ewma_forecast(returns, horizon=horizon),
           "HAR-RV": har_forecast(returns, horizon=horizon)}
    for name in GARCH_SPECS:
        try:
            out[name] = fit_garch(returns, name, horizon=horizon)
        except Exception as e:  # noqa: BLE001 - keep the dashboard usable if one model fails
            logger.warning("%s failed: %s", name, e)
    return out


def backtest_vol_models(returns: pd.Series, test_fraction: float = 0.3, horizon: int = 5) -> pd.DataFrame:
    """
    Out-of-sample comparison. Parameters are estimated on the first (1 - test_fraction)
    of the sample, then each model forecasts the next `horizon`-day realized volatility
    every day of the test window. Scored by RMSE and QLIKE (lower is better for both).
    """
    r = returns.dropna()
    n = len(r)
    split = int(n * (1 - test_fraction))
    if split < 120 or n - split < 30:
        return pd.DataFrame()

    realized = _realized_vol(r, horizon).shift(-horizon)
    preds = {}

    ewma_var = (r ** 2).ewm(alpha=0.06, adjust=False).mean()
    preds["EWMA (λ=0.94)"] = np.sqrt(ewma_var)

    X = har_features(r)
    train = pd.concat([X, realized.rename("y")], axis=1).iloc[:split].dropna()
    A = np.column_stack([np.ones(len(train)), train[["rv_d", "rv_w", "rv_m"]].values])
    coef, *_ = np.linalg.lstsq(A, train["y"].values, rcond=None)
    preds["HAR-RV"] = (np.column_stack([np.ones(len(X)), X.values]) @ coef).clip(1e-6)
    preds["HAR-RV"] = pd.Series(preds["HAR-RV"], index=X.index)

    r_pct = r * 100
    for name, spec in GARCH_SPECS.items():
        try:
            am = _arch_model(r_pct, spec, "t")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = am.fit(last_obs=r.index[split], disp="off", show_warning=False)
                method = "simulation" if name.startswith("EGARCH") else "analytic"
                fc = res.forecast(horizon=horizon, start=r.index[split], reindex=True, method=method,
                                  simulations=200)
            avg_var = fc.variance.mean(axis=1)
            preds[name] = np.sqrt(avg_var) / 100
        except Exception as e:  # noqa: BLE001
            logger.warning("Backtest for %s failed: %s", name, e)

    rows = []
    test_idx = r.index[split:]
    for name, p in preds.items():
        df = pd.concat([p.rename("f"), realized.rename("y")], axis=1).loc[test_idx].dropna()
        df = df[(df["f"] > 0) & (df["y"] > 0)]
        if df.empty:
            continue
        rmse = np.sqrt(((df["f"] - df["y"]) ** 2).mean())
        ratio = df["y"] ** 2 / df["f"] ** 2
        qlike = (ratio - np.log(ratio) - 1).mean()
        rows.append({"Model": name, "RMSE (ann.)": rmse * np.sqrt(252), "QLIKE": qlike, "Obs": len(df)})

    return pd.DataFrame(rows).sort_values("QLIKE").set_index("Model")


def garch_var_backtest(returns: pd.Series, confidence_level: float = 0.99,
                       test_fraction: float = 0.3) -> pd.DataFrame:
    """One-day-ahead GJR-GARCH-t VaR forecasts over the test window (positive loss numbers)."""
    from scipy.stats import t as student_t
    r = returns.dropna()
    split = int(len(r) * (1 - test_fraction))
    am = _arch_model(r * 100, GARCH_SPECS["GJR-GARCH(1,1)"], "t")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = am.fit(last_obs=r.index[split], disp="off", show_warning=False)
        fc = res.forecast(horizon=1, start=r.index[split], reindex=True)
    nu = res.params["nu"]
    q = student_t.ppf(1 - confidence_level, nu) * np.sqrt((nu - 2) / nu)
    mu = fc.mean.iloc[:, 0]
    sigma = np.sqrt(fc.variance.iloc[:, 0])
    # Forecast made at t applies to t+1
    var = (-(mu + q * sigma) / 100).shift(1)
    return pd.DataFrame({"return": r, "var": var}).iloc[split + 1:]
