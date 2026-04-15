"""Abstract provider contracts for market data adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from trading_system.data.schemas import BarRequest

if TYPE_CHECKING:
    import pandas as pd


class MarketDataProvider(ABC):
    """Abstract adapter interface for market data vendors."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider identifier used for logging and routing."""

    @abstractmethod
    async def connect(self) -> None:
        """Initialize provider resources before fetching data."""

    @abstractmethod
    async def disconnect(self) -> None:
        """Release provider resources."""

    @abstractmethod
    async def fetch_bars(self, request: BarRequest) -> "pd.DataFrame":
        """Return normalized OHLCV bars as a MultiIndex DataFrame.

        The returned frame must use a `MultiIndex` of `('symbol', 'timestamp')`
        and the columns `open`, `high`, `low`, `close`, and `volume`.
        """
