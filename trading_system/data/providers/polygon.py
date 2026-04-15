"""Production-oriented scaffold for a Polygon market data adapter."""

from __future__ import annotations

from typing import TYPE_CHECKING

from trading_system.data.providers.base import MarketDataProvider
from trading_system.data.schemas import BarRequest

if TYPE_CHECKING:
    import pandas as pd


class PolygonMarketDataProvider(MarketDataProvider):
    """Scaffold for a Polygon.io market data adapter.

    This class defines the interface and constructor shape expected by the
    rest of the market data layer. Live REST or websocket integration can be
    added later without changing the consumer-facing APIs.
    """

    def __init__(self, *, api_key: str, base_url: str = "https://api.polygon.io") -> None:
        """Store provider configuration for future implementation."""

        self._api_key = api_key
        self._base_url = base_url

    @property
    def provider_name(self) -> str:
        """Return the provider identifier."""

        return "polygon"

    async def connect(self) -> None:
        """Prepare provider resources for future REST or websocket sessions."""

    async def disconnect(self) -> None:
        """Release provider resources."""

    async def fetch_bars(self, request: BarRequest) -> "pd.DataFrame":
        """Fetch normalized OHLCV bars from Polygon."""

        raise NotImplementedError("PolygonMarketDataProvider is a production scaffold and has not been wired yet")
