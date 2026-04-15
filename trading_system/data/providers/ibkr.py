"""Production-oriented scaffold for an IBKR market data adapter."""

from __future__ import annotations

from typing import TYPE_CHECKING

from trading_system.data.providers.base import MarketDataProvider
from trading_system.data.schemas import BarRequest

if TYPE_CHECKING:
    import pandas as pd


class IBKRMarketDataProvider(MarketDataProvider):
    """Scaffold for an Interactive Brokers market data adapter.

    This class reserves the constructor and method surface for a future
    `ib_insync` or native API implementation while keeping the higher-level
    market data interfaces stable.
    """

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 7497,
        client_id: int = 1,
        read_only: bool = True,
    ) -> None:
        """Store connection parameters for future implementation."""

        self._host = host
        self._port = port
        self._client_id = client_id
        self._read_only = read_only

    @property
    def provider_name(self) -> str:
        """Return the provider identifier."""

        return "ibkr"

    async def connect(self) -> None:
        """Prepare provider resources for a future IBKR session."""

    async def disconnect(self) -> None:
        """Release provider resources."""

    async def fetch_bars(self, request: BarRequest) -> "pd.DataFrame":
        """Fetch normalized OHLCV bars from Interactive Brokers."""

        raise NotImplementedError("IBKRMarketDataProvider is a production scaffold and has not been wired yet")
