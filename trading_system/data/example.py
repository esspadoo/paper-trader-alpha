"""Example usage for the market data stream and universe selector."""

from __future__ import annotations

import asyncio

from trading_system.data.market_data import MarketDataStream
from trading_system.data.providers.yfinance import YFinanceMarketDataProvider
from trading_system.data.schemas import UniverseSelectionCriteria
from trading_system.data.universe import UniverseSelector


async def main() -> None:
    """Fetch intraday bars and build a filtered daily universe.

    This example requires the optional `market-data` dependencies:
    `pandas` and `yfinance`.
    """

    provider = YFinanceMarketDataProvider()
    stream = MarketDataStream(provider)
    selector = UniverseSelector(
        stream,
        default_criteria=UniverseSelectionCriteria(
            selection_size=50,
            min_price=10.0,
            min_average_volume=1_000_000.0,
            min_daily_volatility=0.02,
            lookback_days=20,
        ),
    )

    async with stream:
        intraday = await stream.fetch_intraday_ohlcv(["AAPL", "MSFT", "NVDA"], period="1d")
        universe = await selector.select(["AAPL", "MSFT", "NVDA", "META", "AMZN", "TSLA"])

    print(intraday.head())
    print(universe.head())


if __name__ == "__main__":
    asyncio.run(main())
