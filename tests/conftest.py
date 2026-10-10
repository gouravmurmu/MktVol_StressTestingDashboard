import pytest

from data.loader import MarketDataLoader


@pytest.fixture(scope="session")
def prices():
    return MarketDataLoader.generate_demo_data(["SPY", "QQQ", "TLT"], "2018-01-01", "2023-12-31")


@pytest.fixture(scope="session")
def returns(prices):
    return MarketDataLoader.calculate_returns(prices)
