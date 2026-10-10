from typing import Dict, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ui.theme import GAIN, LOSS, SERIES
from ui.theme import PALETTE as P


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


class Visualizer:

    @staticmethod
    def plot_growth(prices: pd.DataFrame, portfolio: Optional[pd.Series] = None,
                    regime: Optional[pd.Series] = None) -> go.Figure:
        """Growth of $100 for each asset and the portfolio, with turbulent regimes shaded."""
        fig = go.Figure()
        norm = prices / prices.iloc[0] * 100
        for i, col in enumerate(norm.columns):
            fig.add_trace(go.Scatter(x=norm.index, y=norm[col], name=col, mode="lines",
                                     line=dict(width=1.6, color=SERIES[(i + 1) % len(SERIES)])))
        if portfolio is not None:
            wealth = 100 * np.exp(portfolio.cumsum())
            fig.add_trace(go.Scatter(x=wealth.index, y=wealth, name="Portfolio", mode="lines",
                                     line=dict(width=3, color=P["primary"])))
        if regime is not None:
            _shade_regimes(fig, regime)
        fig.update_layout(title="Growth of $100", yaxis_title="Value", hovermode="x unified", height=420)
        return fig

    @staticmethod
    def plot_drawdown(drawdown: pd.Series) -> go.Figure:
        fig = go.Figure(go.Scatter(x=drawdown.index, y=drawdown, fill="tozeroy", mode="lines",
                                   line=dict(color=LOSS, width=1.5), fillcolor=_rgba(LOSS, 0.18), name="Drawdown"))
        fig.update_layout(title="Underwater Chart (Drawdown)", yaxis_tickformat=".0%", height=300, showlegend=False)
        return fig

    @staticmethod
    def plot_correlation(returns: pd.DataFrame) -> go.Figure:
        corr = returns.corr()
        fig = go.Figure(go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.index, zmin=-1, zmax=1, text=corr.round(2).values,
            texttemplate="%{text}", colorscale=[[0, P["coral"]], [0.5, "#FFFFFF"], [1, P["primary"]]],
        ))
        fig.update_layout(title="Correlation Matrix", height=360)
        return fig

    @staticmethod
    def plot_rolling_vol(vols: Dict[str, pd.Series]) -> go.Figure:
        fig = go.Figure()
        for i, (name, s) in enumerate(vols.items()):
            fig.add_trace(go.Scatter(x=s.index, y=s, name=name, mode="lines", line=dict(width=2, color=SERIES[i])))
        fig.update_layout(title="Rolling Volatility (annualized)", yaxis_tickformat=".0%", hovermode="x unified",
                          height=340)
        return fig

    @staticmethod
    def plot_return_distribution(returns: pd.Series, var: float, cvar: float) -> go.Figure:
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=returns, nbinsx=80, histnorm="probability density", name="Returns",
                                   marker=dict(color=_rgba(P["primary"], 0.65), line=dict(color="white", width=0.5))))
        fig.add_vrect(x0=returns.min(), x1=-var, fillcolor=_rgba(LOSS, 0.12), line_width=0)
        fig.add_vline(x=-var, line_width=2.5, line_dash="dash", line_color=P["amber"],
                      annotation_text=f"VaR {-var:.2%}", annotation_position="top left")
        fig.add_vline(x=-cvar, line_width=2.5, line_dash="dot", line_color=LOSS,
                      annotation_text=f"CVaR {-cvar:.2%}", annotation_position="bottom left")
        fig.update_layout(title="Return Distribution & Tail Risk", xaxis_tickformat=".1%", showlegend=False, height=400)
        return fig

    @staticmethod
    def plot_var_backtest(returns: pd.Series, var: pd.Series, title: str) -> go.Figure:
        df = pd.concat([returns.rename("r"), var.rename("v")], axis=1).dropna()
        breach = df[df["r"] < -df["v"]]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=df.index, y=df["r"], name="Daily return", marker_color=_rgba(P["sky"], 0.55)))
        fig.add_trace(go.Scatter(x=df.index, y=-df["v"], name="VaR forecast", line=dict(color=P["amber"], width=2)))
        fig.add_trace(go.Scatter(x=breach.index, y=breach["r"], mode="markers", name="Breach",
                                 marker=dict(color=LOSS, size=8, symbol="x")))
        fig.update_layout(title=title, yaxis_tickformat=".1%", height=360, bargap=0)
        return fig

    @staticmethod
    def plot_vol_forecast(history: pd.Series, forecasts: Dict[str, np.ndarray]) -> go.Figure:
        """Recent conditional volatility plus each model's forward term structure."""
        fig = go.Figure()
        hist = history.tail(120)
        fig.add_trace(go.Scatter(x=hist.index, y=hist, name="EWMA history", line=dict(color=P["muted"], width=1.5)))
        future = pd.bdate_range(hist.index[-1], periods=len(next(iter(forecasts.values()))) + 1)[1:]
        for i, (name, path) in enumerate(forecasts.items()):
            fig.add_trace(go.Scatter(x=future, y=path, name=name, mode="lines",
                                     line=dict(width=2.5, color=SERIES[i % len(SERIES)])))
        fig.add_vline(x=hist.index[-1], line_dash="dot", line_color=P["border"])
        fig.update_layout(title="Volatility Forecast (annualized)", yaxis_tickformat=".0%", hovermode="x unified",
                          height=400)
        return fig

    @staticmethod
    def plot_regime_probability(prices: pd.Series, probs: pd.DataFrame) -> go.Figure:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4], vertical_spacing=0.06)
        fig.add_trace(go.Scatter(x=prices.index, y=prices, name="Price", line=dict(color=P["primary"], width=2)),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=probs.index, y=probs["Turbulent"], name="P(Turbulent)", fill="tozeroy",
                                 line=dict(color=LOSS, width=1.5), fillcolor=_rgba(LOSS, 0.2)), row=2, col=1)
        _shade_regimes(fig, probs["Regime"], row=1)
        fig.update_yaxes(tickformat=".0%", range=[0, 1], row=2, col=1)
        fig.update_layout(title="Markov-Switching Volatility Regimes", height=480, hovermode="x unified")
        return fig

    @staticmethod
    def plot_simulation(sim_paths: np.ndarray, start_date: Optional[pd.Timestamp] = None) -> go.Figure:
        """Fan chart: percentile bands plus a handful of sample paths."""
        T = sim_paths.shape[0]
        x = pd.bdate_range(start_date, periods=T) if start_date is not None else np.arange(T)
        pct = np.percentile(sim_paths, [5, 25, 50, 75, 95], axis=1)
        fig = go.Figure()
        for i in range(min(25, sim_paths.shape[1])):
            fig.add_trace(go.Scatter(x=x, y=sim_paths[:, i], mode="lines", showlegend=False, hoverinfo="skip",
                                     line=dict(width=0.6, color=_rgba(P["muted"], 0.25))))
        fig.add_trace(go.Scatter(x=x, y=pct[4], line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=x, y=pct[0], fill="tonexty", fillcolor=_rgba(P["sky"], 0.18),
                                 line=dict(width=0), name="5–95%"))
        fig.add_trace(go.Scatter(x=x, y=pct[3], line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=x, y=pct[1], fill="tonexty", fillcolor=_rgba(P["primary"], 0.25),
                                 line=dict(width=0), name="25–75%"))
        fig.add_trace(go.Scatter(x=x, y=pct[2], name="Median", line=dict(width=3, color=P["primary"])))
        fig.update_layout(title="Monte Carlo Fan Chart", yaxis_title="Projected value", hovermode="x unified",
                          height=440)
        return fig

    @staticmethod
    def plot_terminal_distribution(final_returns: np.ndarray) -> go.Figure:
        hist, edges = np.histogram(final_returns, bins=60)
        centers = (edges[:-1] + edges[1:]) / 2
        fig = go.Figure(go.Bar(x=centers, y=hist, marker_color=np.where(centers < 0, LOSS, GAIN), opacity=0.85))
        fig.update_layout(title="Terminal Return Distribution", xaxis_tickformat=".0%", height=320, bargap=0.02,
                          showlegend=False)
        return fig

    @staticmethod
    def plot_stress_paths(paths: np.ndarray, portfolio_value: float, title: str) -> go.Figure:
        pct = np.percentile(paths, [5, 50, 95], axis=1) * portfolio_value
        x = np.arange(paths.shape[0])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=pct[2], line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=x, y=pct[0], fill="tonexty", fillcolor=_rgba(LOSS, 0.15), line=dict(width=0),
                                 name="5–95%"))
        fig.add_trace(go.Scatter(x=x, y=pct[1], name="Median path", line=dict(color=LOSS, width=3)))
        fig.add_hline(y=portfolio_value, line_dash="dot", line_color=P["muted"])
        fig.update_layout(title=title, xaxis_title="Trading days", yaxis_title="Portfolio value", height=400,
                          hovermode="x unified")
        return fig

    @staticmethod
    def plot_waterfall(contrib: pd.Series) -> go.Figure:
        fig = go.Figure(go.Waterfall(
            x=list(contrib.index) + ["Total"], y=list(contrib.values) + [contrib.sum()],
            measure=["relative"] * len(contrib) + ["total"],
            decreasing=dict(marker=dict(color=LOSS)), increasing=dict(marker=dict(color=GAIN)),
            totals=dict(marker=dict(color=P["primary"])), texttemplate="%{y:.1%}", textposition="outside",
        ))
        fig.update_layout(title="Loss Attribution by Holding", yaxis_tickformat=".0%", height=380, showlegend=False)
        return fig

    @staticmethod
    def plot_bar(series: pd.Series, title: str, fmt: str = ".2%") -> go.Figure:
        fig = go.Figure(go.Bar(x=series.index, y=series.values, marker_color=SERIES[: len(series)],
                               texttemplate=f"%{{y:{fmt}}}", textposition="outside"))
        fig.update_layout(title=title, yaxis_tickformat=fmt, height=340, showlegend=False)
        return fig


def _shade_regimes(fig: go.Figure, regime: pd.Series, row: Optional[int] = None) -> None:
    """Shades contiguous 'Turbulent' (or -1/bear) periods."""
    bad = regime.isin(["Turbulent", -1, -1.0])
    if not bad.any():
        return
    groups = (bad != bad.shift()).cumsum()
    for _, seg in regime[bad].groupby(groups[bad]):
        kwargs = dict(x0=seg.index[0], x1=seg.index[-1], fillcolor=_rgba(LOSS, 0.08), line_width=0, layer="below")
        if row is not None:
            kwargs.update(row=row, col=1)
        fig.add_vrect(**kwargs)
