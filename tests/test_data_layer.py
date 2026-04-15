"""Offline tests for the market data layer."""

from __future__ import annotations

import importlib.util
import unittest
from datetime import datetime, timezone

from trading_system.data import MarketDataProvider, MarketDataStream, UniverseSelectionCriteria, UniverseSelector
from trading_system.data.schemas import BarRequest

PANDAS_AVAILABLE = importlib.util.find_spec("pandas") is not None

if PANDAS_AVAILABLE:
    import pandas as pd


class FakeMarketDataProvider(MarketDataProvider):
    """Provider stub used to test caching and universe selection."""

    def __init__(self, intraday_frame: "pd.DataFrame", daily_frame: "pd.DataFrame") -> None:
        """Store canned market data responses."""

        self._intraday_frame = intraday_frame
        self._daily_frame = daily_frame
        self._connected = False
        self.fetch_calls: list[str] = []

    @property
    def provider_name(self) -> str:
        """Return the stub provider identifier."""

        return "fake"

    async def connect(self) -> None:
        """Mark the provider as connected."""

        self._connected = True

    async def disconnect(self) -> None:
        """Mark the provider as disconnected."""

        self._connected = False

    async def fetch_bars(self, request: BarRequest) -> "pd.DataFrame":
        """Return the canned frame matching the interval."""

        self.fetch_calls.append(request.interval)
        if request.interval == "5m":
            return self._intraday_frame.copy(deep=True)
        return self._daily_frame.copy(deep=True)


@unittest.skipUnless(PANDAS_AVAILABLE, "pandas is required for the data-layer tests")
class DataLayerTests(unittest.IsolatedAsyncioTestCase):
    """Verify caching and universe selection behavior."""

    def setUp(self) -> None:
        """Create reusable test frames."""

        intraday_index = pd.MultiIndex.from_tuples(
            [
                ("AAPL", pd.Timestamp("2026-04-13 13:30:00+00:00")),
                ("AAPL", pd.Timestamp("2026-04-13 13:35:00+00:00")),
                ("MSFT", pd.Timestamp("2026-04-13 13:30:00+00:00")),
                ("MSFT", pd.Timestamp("2026-04-13 13:35:00+00:00")),
            ],
            names=["symbol", "timestamp"],
        )
        self.intraday_frame = pd.DataFrame(
            {
                "open": [190.0, 190.5, 410.0, 411.0],
                "high": [191.0, 191.2, 412.0, 412.5],
                "low": [189.8, 190.2, 409.0, 410.5],
                "close": [190.8, 191.0, 411.5, 412.0],
                "volume": [300_000, 280_000, 220_000, 240_000],
            },
            index=intraday_index,
        )

        daily_index = pd.MultiIndex.from_tuples(
            [
                ("AAPL", pd.Timestamp("2026-04-07 00:00:00+00:00")),
                ("AAPL", pd.Timestamp("2026-04-08 00:00:00+00:00")),
                ("AAPL", pd.Timestamp("2026-04-09 00:00:00+00:00")),
                ("AAPL", pd.Timestamp("2026-04-10 00:00:00+00:00")),
                ("AAPL", pd.Timestamp("2026-04-13 00:00:00+00:00")),
                ("MSFT", pd.Timestamp("2026-04-07 00:00:00+00:00")),
                ("MSFT", pd.Timestamp("2026-04-08 00:00:00+00:00")),
                ("MSFT", pd.Timestamp("2026-04-09 00:00:00+00:00")),
                ("MSFT", pd.Timestamp("2026-04-10 00:00:00+00:00")),
                ("MSFT", pd.Timestamp("2026-04-13 00:00:00+00:00")),
                ("CHEAP", pd.Timestamp("2026-04-07 00:00:00+00:00")),
                ("CHEAP", pd.Timestamp("2026-04-08 00:00:00+00:00")),
                ("CHEAP", pd.Timestamp("2026-04-09 00:00:00+00:00")),
                ("CHEAP", pd.Timestamp("2026-04-10 00:00:00+00:00")),
                ("CHEAP", pd.Timestamp("2026-04-13 00:00:00+00:00")),
                ("STABLE", pd.Timestamp("2026-04-07 00:00:00+00:00")),
                ("STABLE", pd.Timestamp("2026-04-08 00:00:00+00:00")),
                ("STABLE", pd.Timestamp("2026-04-09 00:00:00+00:00")),
                ("STABLE", pd.Timestamp("2026-04-10 00:00:00+00:00")),
                ("STABLE", pd.Timestamp("2026-04-13 00:00:00+00:00")),
            ],
            names=["symbol", "timestamp"],
        )
        self.daily_frame = pd.DataFrame(
            {
                "open": [
                    12.0,
                    12.4,
                    12.1,
                    12.7,
                    12.8,
                    30.0,
                    31.0,
                    29.5,
                    32.0,
                    32.5,
                    8.0,
                    8.1,
                    8.2,
                    8.0,
                    8.3,
                    50.0,
                    50.0,
                    50.1,
                    50.1,
                    50.1,
                ],
                "high": [
                    12.5,
                    12.6,
                    12.8,
                    12.9,
                    13.1,
                    31.5,
                    31.7,
                    32.0,
                    33.0,
                    33.5,
                    8.2,
                    8.3,
                    8.4,
                    8.2,
                    8.5,
                    50.1,
                    50.2,
                    50.2,
                    50.3,
                    50.2,
                ],
                "low": [
                    11.9,
                    12.1,
                    11.8,
                    12.4,
                    12.5,
                    29.8,
                    29.9,
                    29.0,
                    31.8,
                    32.2,
                    7.8,
                    7.9,
                    8.0,
                    7.8,
                    8.1,
                    49.9,
                    49.9,
                    49.9,
                    50.0,
                    50.0,
                ],
                "close": [
                    12.3,
                    12.0,
                    12.7,
                    12.5,
                    13.0,
                    31.2,
                    29.3,
                    32.0,
                    31.0,
                    33.0,
                    8.1,
                    8.0,
                    8.3,
                    8.1,
                    8.4,
                    50.0,
                    50.1,
                    50.0,
                    50.1,
                    50.0,
                ],
                "volume": [
                    1_600_000,
                    1_500_000,
                    1_700_000,
                    1_550_000,
                    1_650_000,
                    2_400_000,
                    2_100_000,
                    2_300_000,
                    2_500_000,
                    2_600_000,
                    3_000_000,
                    2_900_000,
                    3_100_000,
                    2_800_000,
                    3_000_000,
                    5_000_000,
                    5_200_000,
                    5_100_000,
                    5_300_000,
                    5_100_000,
                ],
            },
            index=daily_index,
        )

    async def test_market_data_stream_caches_repeated_intraday_requests(self) -> None:
        """Repeated intraday fetches should reuse the cache."""

        provider = FakeMarketDataProvider(self.intraday_frame, self.daily_frame)
        stream = MarketDataStream(provider)

        async with stream:
            first = await stream.fetch_intraday_ohlcv(["AAPL", "MSFT"], period="1d")
            second = await stream.fetch_intraday_ohlcv(["AAPL", "MSFT"], period="1d")

        self.assertEqual(provider.fetch_calls, ["5m"])
        self.assertEqual(first.index.names, ["symbol", "timestamp"])
        self.assertTrue(first.equals(second))

    async def test_universe_selector_filters_and_ranks_candidates(self) -> None:
        """The selector should keep only liquid, volatile symbols and rank by dollar volume."""

        provider = FakeMarketDataProvider(self.intraday_frame, self.daily_frame)
        stream = MarketDataStream(provider)
        selector = UniverseSelector(
            stream,
            default_criteria=UniverseSelectionCriteria(
                selection_size=2,
                min_price=10.0,
                min_average_volume=1_000_000.0,
                min_daily_volatility=0.02,
                lookback_days=5,
            ),
        )

        async with stream:
            universe = await selector.select(
                ["AAPL", "MSFT", "CHEAP", "STABLE"],
                as_of=datetime(2026, 4, 13, tzinfo=timezone.utc),
            )

        self.assertEqual(provider.fetch_calls, ["1d"])
        self.assertEqual(universe["symbol"].tolist(), ["MSFT", "AAPL"])
        self.assertEqual(universe["rank"].tolist(), [1, 2])
