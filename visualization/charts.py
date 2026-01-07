import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

class Visualizer:
    
    @staticmethod
    def plot_price_and_regime(prices: pd.Series, regime_col: pd.Series = None) -> go.Figure:
        """
        Plots price history. If regime is provided, backgrounds the plot.
        """
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=prices.index, y=prices, mode='lines', name='Price', line=dict(color='#636EFA', width=2)))
        
        if regime_col is not None:
            # Map regimes to colors (1: Bull=Green, -1: Bear=Red)
            # This is complex to do efficiently with huge data, so we might just overlay simple shapes or colored lines
            # For simplicity in this dashboard, we might overlay a secondary heatmap or scatter
            pass
            
        fig.update_layout(title="Asset Price History", xaxis_title="Date", yaxis_title="Price", template="plotly_white")
        return fig

    @staticmethod
    def plot_volatility(volatility: pd.Series) -> go.Figure:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=volatility.index, y=volatility, mode='lines', name='Annualized Vol', 
                                 line=dict(color='orange')))
        fig.update_layout(title="Rolling Volatility (Annualized)", template="plotly_white")
        return fig

    @staticmethod
    def plot_return_distribution(returns: pd.Series, var: float, cvar: float) -> go.Figure:
        """
        Histogram of returns with VaR and CVaR lines.
        """
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=returns, nbinsx=100, name='Returns', histnorm='probability density', marker_color='blue', opacity=0.6))
        
        # VaR Line
        fig.add_vline(x=-var, line_width=3, line_dash="dash", line_color="red", annotation_text=f"VaR: {-var:.2%}")
        
        # CVaR Line
        fig.add_vline(x=-cvar, line_width=3, line_dash="dot", line_color="darkred", annotation_text=f"CVaR: {-cvar:.2%}")
        
        fig.update_layout(title="Return Distribution & Tail Risk", template="plotly_white")
        return fig

    @staticmethod
    def plot_simulation(sim_paths: np.ndarray) -> go.Figure:
        """
        Plots first 100 paths of simulation.
        """
        fig = go.Figure()
        
        # Downsample if too many
        n_paths = min(sim_paths.shape[1], 100)
        
        for i in range(n_paths):
            fig.add_trace(go.Scatter(y=sim_paths[:, i], mode='lines', line=dict(width=0.5, color='gray'), opacity=0.3, showlegend=False))
            
        # Add mean path
        mean_path = np.mean(sim_paths, axis=1)
        fig.add_trace(go.Scatter(y=mean_path, mode='lines', name='Mean Path', line=dict(width=3, color='blue')))
        
        fig.update_layout(title="Monte Carlo Simulation Paths", xaxis_title="Days", yaxis_title="Projected Price", template="plotly_white")
        return fig
