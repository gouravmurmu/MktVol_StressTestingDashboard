import numpy as np
import pandas as pd

from data.loader import MarketDataLoader
from data.portfolio import PortfolioManager


def test_demo_data_is_deterministic(prices):
    again = MarketDataLoader.generate_demo_data(["SPY", "QQQ", "TLT"], "2018-01-01", "2023-12-31")
    pd.testing.assert_frame_equal(prices, again)
    assert (prices > 0).all().all()


def test_extract_close_handles_new_yfinance_multiindex():
    idx = pd.date_range("2024-01-01", periods=3)
    cols = pd.MultiIndex.from_product([["Close", "Open"], ["SPY"]], names=["Price", "Ticker"])
    raw = pd.DataFrame(np.arange(6.0).reshape(3, 2) + 1, index=idx, columns=cols)
    out = MarketDataLoader._extract_close(raw, ["SPY"])
    assert list(out.columns) == ["SPY"]
    assert out["SPY"].tolist() == [1.0, 3.0, 5.0]


def test_extract_close_flat_columns():
    idx = pd.date_range("2024-01-01", periods=2)
    raw = pd.DataFrame({"Open": [1.0, 2.0], "Close": [3.0, 4.0]}, index=idx)
    assert MarketDataLoader._extract_close(raw, ["AAPL"])["AAPL"].tolist() == [3.0, 4.0]


def test_portfolio_weights_normalize_and_aggregate(returns):
    pm = PortfolioManager()
    w = pm.normalize_weights(returns.columns, {"SPY": 2, "QQQ": 1, "TLT": 1})
    assert abs(sum(w.values()) - 1) < 1e-12 and w["SPY"] == 0.5
    single = pm.construct_portfolio(returns, {"SPY": 1})
    np.testing.assert_allclose(single.values, returns["SPY"].values)
    rc = pm.risk_contributions(returns, w)
    assert abs(rc.sum() - 1) < 1e-9
